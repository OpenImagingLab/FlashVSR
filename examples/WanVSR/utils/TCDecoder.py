#!/usr/bin/env python3
"""
Tiny AutoEncoder for Hunyuan Video (Decoder-only, pruned)
- Encoder removed
- Transplant/widening helpers removed
- Deepening (IdentityConv2d+ReLU) is now built into the decoder structure itself
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from collections import namedtuple
from einops import rearrange
import torch.nn.init as init

try:
    from diffsynth.perf_stats import record as _perf_record
except Exception:
    def _perf_record(name, count=1):  # noqa: ARG001
        return None

DecoderResult = namedtuple("DecoderResult", ("frame", "memory"))
TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))

# ---------------------------------------------------------------------------
# Hopper TCDecoder acceleration: channels_last (NHWC) memory format.
#
# The TCDecoder is a pure Conv2d (TAEHV) graph. With the default contiguous
# (NCHW) layout, cuDNN inserts nchwToNhwc/nhwcToNchw conversions around every
# bf16 conv on Hopper (~226 ms / ~9% of denoise GPU time @768x1408) and the
# convs themselves run ~1.5x slower. Running the whole decoder in channels_last
# removes the layout churn and speeds up the convs, with bit-identical math.
#
# Knob (opt-in, default OFF; set 1 to enable the NHWC fast path):
#   FLASHVSR_TCDECODER_CHANNELS_LAST = 1 | 0
# ---------------------------------------------------------------------------

_TCDEC_CHANNELS_LAST = os.environ.get("FLASHVSR_TCDECODER_CHANNELS_LAST", "0") != "0"

# A MemBlock's recurrent state is its input tensor from the previous timestep.
# Rebinding the state keeps that tensor alive and removes a full-frame D2D
# copy after every non-initial MemBlock invocation. Arithmetic and state order
# are unchanged. Opt-in until the E2E bit-equality/performance gate passes.
_TCDEC_POINTER_STATE = os.environ.get(
    "FLASHVSR_TCDECODER_POINTER_STATE", "0") != "0"

_TCDEC_FUSE_POINTWISE = os.environ.get(
    "FLASHVSR_TCDECODER_FUSE_POINTWISE", "0") != "0"
_TCDEC_FUSE_POINTWISE_FAILED = False
_TCDEC_UPSAMPLE = os.environ.get(
    "FLASHVSR_TCDECODER_UPSAMPLE", "0") != "0"
_TCDEC_UPSAMPLE_FAILED = False
_TCDEC_CONCAT = os.environ.get(
    "FLASHVSR_TCDECODER_CONCAT", "0") != "0"
_TCDEC_CONCAT_FAILED = False
# Algebraic reorder of the three `Upsample(2x) -> TGrow(1x1)` pairs: run the
# bias-free 1x1 TGrow conv at LOW resolution (4x fewer FLOPs), then unpack the
# temporal channel groups and nearest-upsample them in one Triton pass.
# Nearest-neighbor duplication commutes exactly with a pointwise conv; the only
# numeric caveat is that cuDNN may pick a different reduction split for the
# low-res GEMM, so this path is gated on the >=49 dB PSNR budget (measured far
# above; see PHASE_BENCH_LOG) rather than the bit-exact gate.
_TCDEC_TGROW_UP = os.environ.get(
    "FLASHVSR_TCDECODER_TGROW_UP", "0") != "0"
_TCDEC_TGROW_UP_FAILED = False
# cuDNN runtime-fused Conv+Bias(+Add)+ReLU for the MemBlock chain and the
# standalone Conv->ReLU pairs. QUALITY-GATED (not bit-exact): the fused engine
# skips the separate BF16 materialization between conv/bias/residual/ReLU, so
# ~10-14% of values move by 1 BF16 ULP (~70 dB isolated PSNR, gate >= 49 dB).
_TCDEC_CUDNN_FUSED = os.environ.get(
    "FLASHVSR_TCDECODER_CUDNN_FUSED", "0") != "0"
_TCDEC_CUDNN_FUSED_FAILED = False
# Phase 7-C: the first MemBlock convolution consumes cat([current, past]).
# Split its weights once and evaluate the two halves separately so the recurrent
# concat allocation/copy disappears. cuDNN's add+ReLU path keeps the current
# contribution fused with the second accumulation. This changes BF16 reduction
# order, so it is quality-gated and only supplements CUDNN_FUSED.
_TCDEC_SPLITK_CONV = os.environ.get(
    "FLASHVSR_TCDECODER_SPLITK_CONV", "0") != "0"
_TCDEC_SPLITK_CONV_FAILED = False
try:
    from .triton_tcdecoder_ops import (
        bias_relu as _bias_relu,
        bias_residual_relu as _bias_residual_relu,
        upsample2x_channels_last as _upsample2x,
        concat_channels_last as _concat_channels_last,
        tgrow_upsample2x_channels_last as _tgrow_upsample2x,
    )
except Exception:
    try:
        from utils.triton_tcdecoder_ops import (
            bias_relu as _bias_relu,
            bias_residual_relu as _bias_residual_relu,
            upsample2x_channels_last as _upsample2x,
            concat_channels_last as _concat_channels_last,
            tgrow_upsample2x_channels_last as _tgrow_upsample2x,
        )
    except Exception:
        _bias_relu = None
        _bias_residual_relu = None
        _upsample2x = None
        _concat_channels_last = None
        _tgrow_upsample2x = None

def _tcdec_channels_last_enabled(device=None):
    if not _TCDEC_CHANNELS_LAST:
        return False
    try:
        return torch.cuda.is_available()
    except Exception:
        return False

# ----------------------------
# Utility / building blocks
# ----------------------------

class IdentityConv2d(nn.Conv2d):
    """Same-shape Conv2d initialized to identity (Dirac)."""
    def __init__(self, C, kernel_size=3, bias=False):
        pad = kernel_size // 2
        super().__init__(C, C, kernel_size, padding=pad, bias=bias)
        with torch.no_grad():
            init.dirac_(self.weight)
            if self.bias is not None:
                self.bias.zero_()

def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3

class MemBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(
            conv(n_in * 2, n_out), nn.ReLU(inplace=True),
            conv(n_out, n_out), nn.ReLU(inplace=True),
            conv(n_out, n_out)
        )
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)
    def forward(self, x, past):
        global _TCDEC_FUSE_POINTWISE_FAILED, _TCDEC_CONCAT_FAILED, \
            _TCDEC_CUDNN_FUSED_FAILED, _TCDEC_SPLITK_CONV_FAILED
        if (_TCDEC_SPLITK_CONV and _TCDEC_CUDNN_FUSED
                and not _TCDEC_SPLITK_CONV_FAILED
                and isinstance(self.skip, nn.Identity)):
            # W * cat([x, past]) = W_current * x + W_past * past. Cache the
            # two layout-contiguous views: slicing the input-channel dimension
            # otherwise leaves a strided weight that defeats cuDNN's fast path.
            try:
                conv0, conv1, conv2 = self.conv[0], self.conv[2], self.conv[4]
                channels = x.shape[1]
                version = conv0.weight._version
                cached = getattr(self, "_splitk_weights", None)
                key = (conv0.weight.data_ptr(), version, channels)
                if cached is None or cached[0] != key:
                    current_weight = conv0.weight[:, :channels].contiguous(
                        memory_format=torch.channels_last)
                    past_weight = conv0.weight[:, channels:].contiguous(
                        memory_format=torch.channels_last)
                    self._splitk_weights = (key, current_weight, past_weight)
                else:
                    _, current_weight, past_weight = cached
                past_y = F.conv2d(
                    past, past_weight, None, conv0.stride, conv0.padding,
                    conv0.dilation, conv0.groups)
                y = torch.ops.aten.cudnn_convolution_add_relu(
                    x, current_weight, past_y, 1.0, conv0.bias, conv0.stride,
                    conv0.padding, conv0.dilation, conv0.groups)
                y = torch.ops.aten.cudnn_convolution_relu(
                    y, conv1.weight, conv1.bias, conv1.stride,
                    conv1.padding, conv1.dilation, conv1.groups)
                y = torch.ops.aten.cudnn_convolution_add_relu(
                    y, conv2.weight, x, 1.0, conv2.bias, conv2.stride,
                    conv2.padding, conv2.dilation, conv2.groups)
                _perf_record("decoder_memblock_splitk")
                return y
            except Exception as exc:
                _TCDEC_SPLITK_CONV_FAILED = True
                try:
                    from diffsynth.perf_stats import record_error
                    record_error("decoder_memblock_splitk", exc)
                except Exception:
                    pass
        if (_TCDEC_CONCAT and not _TCDEC_CONCAT_FAILED
                and _concat_channels_last is not None):
            try:
                merged = _concat_channels_last(x, past)
                _perf_record("decoder_concat_triton")
            except Exception as exc:
                _TCDEC_CONCAT_FAILED = True
                try:
                    from diffsynth.perf_stats import record_error
                    record_error("decoder_concat_triton", exc)
                except Exception:
                    pass
                merged = torch.cat([x, past], 1)
                _perf_record("decoder_concat_native")
        else:
            merged = torch.cat([x, past], 1)
            _perf_record("decoder_concat_native")
        if (_TCDEC_CUDNN_FUSED and not _TCDEC_CUDNN_FUSED_FAILED
                and isinstance(self.skip, nn.Identity)):
            # Quality-gated cuDNN fused engines: conv+bias+ReLU twice, then
            # conv+bias+residual+ReLU in a single kernel each. Removes the
            # separate epilogue kernels entirely.
            try:
                conv0, conv1, conv2 = self.conv[0], self.conv[2], self.conv[4]
                y = torch.ops.aten.cudnn_convolution_relu(
                    merged, conv0.weight, conv0.bias, conv0.stride,
                    conv0.padding, conv0.dilation, conv0.groups)
                y = torch.ops.aten.cudnn_convolution_relu(
                    y, conv1.weight, conv1.bias, conv1.stride,
                    conv1.padding, conv1.dilation, conv1.groups)
                y = torch.ops.aten.cudnn_convolution_add_relu(
                    y, conv2.weight, x, 1.0, conv2.bias, conv2.stride,
                    conv2.padding, conv2.dilation, conv2.groups)
                _perf_record("decoder_memblock_cudnn")
                return y
            except Exception as exc:
                _TCDEC_CUDNN_FUSED_FAILED = True
                try:
                    from diffsynth.perf_stats import record_error
                    record_error("decoder_memblock_cudnn", exc)
                except Exception:
                    pass
        if (_TCDEC_FUSE_POINTWISE and not _TCDEC_FUSE_POINTWISE_FAILED
                and _bias_relu is not None and _bias_residual_relu is not None
                and isinstance(self.skip, nn.Identity)):
            try:
                conv0, conv1, conv2 = self.conv[0], self.conv[2], self.conv[4]
                y = F.conv2d(
                    merged, conv0.weight, None, conv0.stride, conv0.padding,
                    conv0.dilation, conv0.groups)
                y = _bias_relu(y, conv0.bias)
                y = F.conv2d(
                    y, conv1.weight, None, conv1.stride, conv1.padding,
                    conv1.dilation, conv1.groups)
                y = _bias_relu(y, conv1.bias)
                y = F.conv2d(
                    y, conv2.weight, None, conv2.stride, conv2.padding,
                    conv2.dilation, conv2.groups)
                y = _bias_residual_relu(y, conv2.bias, x)
                _perf_record("decoder_memblock_fused")
                return y
            except Exception as exc:
                _TCDEC_FUSE_POINTWISE_FAILED = True
                try:
                    from diffsynth.perf_stats import record_error
                    record_error("decoder_memblock_fused", exc)
                except Exception:
                    pass
        _perf_record("decoder_memblock_native")
        return self.act(self.conv(merged) + self.skip(x))

class TPool(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f*stride, n_f, 1, bias=False)
    def forward(self, x):
        _NT, C, H, W = x.shape
        return self.conv(x.reshape(-1, self.stride * C, H, W))

class TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f*stride, 1, bias=False)
    def forward(self, x):
        _NT, C, H, W = x.shape
        x = self.conv(x)
        if self.stride > 1:
            _perf_record("decoder_tgrow_native")
        return x.reshape(-1, C, H, W)

class PixelShuffle3d(nn.Module):
    def __init__(self, ff, hh, ww):
        super().__init__()
        self.ff = ff
        self.hh = hh
        self.ww = ww
    def forward(self, x):
        # x: (B, C, F, H, W)
        B, C, F, H, W = x.shape
        if F % self.ff != 0:
            first_frame = x[:, :, 0:1, :, :].repeat(1, 1, self.ff - F % self.ff, 1, 1)
            x = torch.cat([first_frame, x], dim=2)
        return rearrange(
            x,
            'b c (f ff) (h hh) (w ww) -> b (c ff hh ww) f h w',
            ff=self.ff, hh=self.hh, ww=self.ww
        ).transpose(1, 2)

# ----------------------------
# Generic NTCHW graph executor (kept; used by decoder)
# ----------------------------

def apply_model_with_memblocks(model, x, parallel, show_progress_bar, mem=None,
                               output=None, output_trim=0):
    """
    Apply a sequential model with memblocks to the given input.
    Args:
    - model: nn.Sequential of blocks to apply
    - x: input data, of dimensions NTCHW
    - parallel: if True, parallelize over timesteps (fast but uses O(T) memory)
        if False, each timestep will be processed sequentially (slow but uses O(1) memory)
    - show_progress_bar: if True, enables tqdm progressbar display

    Returns NTCHW tensor of output data.
    """
    global _TCDEC_UPSAMPLE_FAILED, _TCDEC_TGROW_UP_FAILED, \
        _TCDEC_CUDNN_FUSED_FAILED
    assert x.ndim == 5, f"TAEHV operates on NTCHW tensors, but got {x.ndim}-dim tensor"
    N, T, C, H, W = x.shape
    if parallel:
        x = x.reshape(N*T, C, H, W)
        for b in tqdm(model, disable=not show_progress_bar):
            if isinstance(b, MemBlock):
                NT, C, H, W = x.shape
                T = NT // N
                _x = x.reshape(N, T, C, H, W)
                mem = F.pad(_x, (0,0,0,0,0,0,1,0), value=0)[:,:T].reshape(x.shape)
                x = b(x, mem)
            else:
                x = b(x)
        NT, C, H, W = x.shape
        T = NT // N
        x = x.view(N, T, C, H, W)
    else:
        out = []
        _cl = _tcdec_channels_last_enabled()
        work_queue = [
            TWorkItem(xt.contiguous(memory_format=torch.channels_last) if _cl else xt, 0)
            for t, xt in enumerate(x.reshape(N, T * C, H, W).chunk(T, dim=1))
        ]
        progress_bar = tqdm(range(T), disable=not show_progress_bar)
        while work_queue:
            xt, i = work_queue.pop(0)
            if i == 0:
                progress_bar.update(1)
            if i == len(model):
                out.append(xt)
            else:
                b = model[i]
                if isinstance(b, MemBlock):
                    if mem[i] is None:
                        xt_new = b(xt, xt * 0)
                        mem[i] = xt
                    else:
                        xt_new = b(xt, mem[i])
                        if _TCDEC_POINTER_STATE:
                            mem[i] = xt
                            _perf_record("decoder_state_pointer_updates")
                        else:
                            mem[i].copy_(xt)
                            _perf_record("decoder_state_copies")
                    work_queue.insert(0, TWorkItem(xt_new, i+1))
                elif isinstance(b, TPool):
                    if mem[i] is None:
                        mem[i] = []
                    mem[i].append(xt)
                    if len(mem[i]) > b.stride:
                        raise ValueError("TPool internal state invalid.")
                    elif len(mem[i]) == b.stride:
                        N_, C_, H_, W_ = xt.shape
                        xt = b(torch.cat(mem[i], 1).view(N_*b.stride, C_, H_, W_))
                        mem[i] = []
                        work_queue.insert(0, TWorkItem(xt, i+1))
                elif isinstance(b, TGrow):
                    xt = b(xt)
                    NT, C_, H_, W_ = xt.shape
                    for xt_next in reversed(xt.view(
                            N, b.stride * C_, H_, W_).chunk(b.stride, 1)):
                        work_queue.insert(0, TWorkItem(xt_next, i+1))
                elif isinstance(b, nn.Upsample):
                    fused_tgrow = False
                    if (_TCDEC_TGROW_UP and not _TCDEC_TGROW_UP_FAILED
                            and _tgrow_upsample2x is not None
                            and b.scale_factor == 2.0
                            and i + 1 < len(model)
                            and isinstance(model[i + 1], TGrow)):
                        # Reordered pair: 1x1 TGrow conv at LOW resolution,
                        # then one kernel unpacks channel groups into temporal
                        # frames and nearest-upsamples them. The Upsample node
                        # at i and the TGrow node at i+1 are both consumed;
                        # frames continue at layer i+2 in temporal order
                        # (neither layer carries mem state).
                        tgrow = model[i + 1]
                        try:
                            grown = F.conv2d(
                                xt, tgrow.conv.weight, tgrow.conv.bias)
                            frames = _tgrow_upsample2x(grown, tgrow.stride)
                            _perf_record("decoder_tgrow_fused")
                            for xt_next in reversed(
                                    frames.chunk(tgrow.stride, 0)):
                                work_queue.insert(
                                    0, TWorkItem(xt_next, i + 2))
                            fused_tgrow = True
                        except Exception as exc:
                            # Nothing observable mutated yet (xt and all mem
                            # entries untouched) -> native path below replays
                            # this Upsample node safely.
                            _TCDEC_TGROW_UP_FAILED = True
                            try:
                                from diffsynth.perf_stats import record_error
                                record_error("decoder_tgrow_fused", exc)
                            except Exception:
                                pass
                    if not fused_tgrow:
                        if (_TCDEC_UPSAMPLE and not _TCDEC_UPSAMPLE_FAILED
                                and _upsample2x is not None
                                and b.scale_factor == 2.0):
                            try:
                                xt = _upsample2x(xt)
                                _perf_record("decoder_upsample_triton")
                            except Exception as exc:
                                _TCDEC_UPSAMPLE_FAILED = True
                                try:
                                    from diffsynth.perf_stats import record_error
                                    record_error("decoder_upsample_triton", exc)
                                except Exception:
                                    pass
                                xt = b(xt)
                                _perf_record("decoder_upsample_native")
                        else:
                            xt = b(xt)
                            _perf_record("decoder_upsample_native")
                        work_queue.insert(0, TWorkItem(xt, i+1))
                elif (isinstance(b, nn.Conv2d) and _TCDEC_CUDNN_FUSED
                      and not _TCDEC_CUDNN_FUSED_FAILED
                      and i + 1 < len(model)
                      and isinstance(model[i + 1], nn.ReLU)):
                    # Quality-gated fused Conv+Bias+ReLU pair (consumes the
                    # ReLU node at i+1). Covers the latent-stage conv, the
                    # deepening IdentityConv2d layers and the full-res tail.
                    try:
                        xt = torch.ops.aten.cudnn_convolution_relu(
                            xt, b.weight, b.bias, b.stride, b.padding,
                            b.dilation, b.groups)
                        _perf_record("decoder_conv_relu_cudnn")
                        work_queue.insert(0, TWorkItem(xt, i + 2))
                    except Exception as exc:
                        _TCDEC_CUDNN_FUSED_FAILED = True
                        try:
                            from diffsynth.perf_stats import record_error
                            record_error("decoder_conv_relu_cudnn", exc)
                        except Exception:
                            pass
                        xt = b(xt)
                        work_queue.insert(0, TWorkItem(xt, i + 1))
                else:
                    if isinstance(b, nn.ReLU):
                        _perf_record("decoder_relu_native")
                    xt = b(xt)
                    work_queue.insert(0, TWorkItem(xt, i+1))
        progress_bar.close()
        if output is None:
            x = torch.stack(out, 1)
        else:
            selected = out[output_trim:]
            expected = (N, len(selected), selected[0].shape[1],
                        selected[0].shape[2], selected[0].shape[3])
            if tuple(output.shape) != expected:
                raise ValueError(
                    f"TCDecoder output shape mismatch: expected {expected}, "
                    f"got {tuple(output.shape)}")
            torch.stack(selected, 1, out=output)
            x = output
    return x, mem

# ----------------------------
# Decoder-only TAEHV
# ----------------------------

class TAEHV(nn.Module):
    image_channels = 3
    def __init__(
        self,
        checkpoint_path="taehv.pth",
        decoder_time_upscale=(True, True),
        decoder_space_upscale=(True, True, True),
        channels = [256, 128, 64, 64],
        latent_channels = 16
    ):
        """Initialize TAEHV (decoder-only) with built-in deepening after every ReLU.
        Deepening config: how_many_each=1, k=3 (fixed as requested).
        """
        super().__init__()
        self.latent_channels = latent_channels
        n_f = channels
        self.frames_to_trim = 2**sum(decoder_time_upscale) - 1

        # Build the decoder "skeleton"
        base_decoder = nn.Sequential(
            Clamp(), conv(self.latent_channels, n_f[0]), nn.ReLU(inplace=True),

            MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1),
            TGrow(n_f[0], 1),
            conv(n_f[0], n_f[1], bias=False),

            MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1),
            TGrow(n_f[1], 2 if decoder_time_upscale[0] else 1),
            conv(n_f[1], n_f[2], bias=False),

            MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1),
            TGrow(n_f[2], 2 if decoder_time_upscale[1] else 1),
            conv(n_f[2], n_f[3], bias=False),

            nn.ReLU(inplace=True), conv(n_f[3], TAEHV.image_channels),
        )

        # Inline deepening: insert (IdentityConv2d(k=3) + ReLU) after every ReLU
        self.decoder = self._apply_identity_deepen(base_decoder, how_many_each=1, k=3)

        self.pixel_shuffle = PixelShuffle3d(4, 8, 8)

        if checkpoint_path is not None:
            missing_keys = self.load_state_dict(
                self.patch_tgrow_layers(torch.load(checkpoint_path, map_location="cpu", weights_only=True)),
                strict=False
            )
            print('missing_keys', missing_keys)

        # Initialize decoder mem state
        self.mem = [None] * len(self.decoder)

    @staticmethod
    def _apply_identity_deepen(decoder: nn.Sequential, how_many_each=1, k=3) -> nn.Sequential:
        """Return a new Sequential where every nn.ReLU is followed by how_many_each*(IdentityConv2d(k)+ReLU)."""
        new_layers = []
        for b in decoder:
            new_layers.append(b)
            if isinstance(b, nn.ReLU):
                # Deduce channel count from preceding layer
                C = None
                if len(new_layers) >= 2 and isinstance(new_layers[-2], nn.Conv2d):
                    C = new_layers[-2].out_channels
                elif len(new_layers) >= 2 and isinstance(new_layers[-2], MemBlock):
                    C = new_layers[-2].conv[-1].out_channels
                if C is not None:
                    for _ in range(how_many_each):
                        new_layers.append(IdentityConv2d(C, kernel_size=k, bias=False))
                        new_layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*new_layers)

    def patch_tgrow_layers(self, sd):
        """Patch TGrow layers to use a smaller kernel if needed (decoder-only)."""
        new_sd = self.state_dict()
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, TGrow):
                key = f"decoder.{i}.conv.weight"
                if key in sd and sd[key].shape[0] > new_sd[key].shape[0]:
                    sd[key] = sd[key][-new_sd[key].shape[0]:]
        return sd

    def _maybe_to_channels_last(self):
        """Lazily convert the conv2d decoder weights to channels_last (NHWC).

        Done after weights are loaded; idempotent. cuDNN then keeps the whole
        conv graph in NHWC on Hopper, removing layout-conversion kernels.
        """
        if getattr(self, "_cl_done", False):
            return
        if _tcdec_channels_last_enabled():
            self.decoder.to(memory_format=torch.channels_last)
        self._cl_done = True

    def decode_video(self, x, parallel=True, show_progress_bar=False, cond=None,
                     out=None):
        """Decode a sequence of frames from latents.
        x: NTCHW latent tensor; returns NTCHW RGB in ~[0, 1].
        """
        if out is not None and parallel:
            raise ValueError("TCDecoder out requires parallel=False")
        self._maybe_to_channels_last()
        trim_flag = self.mem[-8] is None  # keeps original relative check

        if cond is not None:
            x = torch.cat([self.pixel_shuffle(cond), x], dim=2)

        trim = self.frames_to_trim if trim_flag else 0
        x, self.mem = apply_model_with_memblocks(
            self.decoder, x, parallel, show_progress_bar, mem=self.mem,
            output=out, output_trim=trim if out is not None else 0)

        if trim_flag and out is None:
            return x[:, self.frames_to_trim:]
        return x

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Decoder-only model: call decode_video(...) instead.")

    def clean_mem(self):
        self.mem = [None] * len(self.decoder)

class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

class TAEW2_1DiffusersWrapper(nn.Module):
    def __init__(self, pretrained_path=None, channels = [256, 128, 64, 64]):
        super().__init__()
        self.dtype = torch.bfloat16
        self.device = "cuda"
        self.taehv = TAEHV(pretrained_path, channels = channels).to(self.dtype)
        self.temperal_downsample = [True, True, False]  # [sic]
        self.config = DotDict(scaling_factor=1.0, latents_mean=torch.zeros(16), z_dim=16, latents_std=torch.ones(16))

    def decode(self, latents, return_dict=None):
        n, c, t, h, w = latents.shape
        return (self.taehv.decode_video(latents.transpose(1, 2), parallel=False).transpose(1, 2).mul_(2).sub_(1),)

    def stream_decode_with_cond(self, latents, tiled=False, cond=None):
        n, c, t, h, w = latents.shape
        return self.taehv.decode_video(latents.transpose(1, 2), parallel=False, cond=cond).transpose(1, 2).mul_(2).sub_(1)

    def clean_mem(self):
        self.taehv.clean_mem()

# ----------------------------
# Simplified builder (no small, no transplant, no post-hoc deepening)
# ----------------------------

def build_tcdecoder(new_channels = [512, 256, 128, 128],
                                  device="cuda",
                                  dtype=torch.bfloat16,
                                  new_latent_channels=None):
    """
    构建“更宽”的 decoder；深度增强（IdentityConv2d+ReLU）已在 TAEHV 内部完成。
    - 不创建 small / 不做移植
    - base_ckpt_path 参数保留但不使用（接口兼容）

    返回：big （单个模型）
    """
    if new_latent_channels is not None:
        big = TAEHV(checkpoint_path=None, channels=new_channels, latent_channels=new_latent_channels).to(device).to(dtype).train()
    else:
        big = TAEHV(checkpoint_path=None, channels=new_channels).to(device).to(dtype).train()

    big.clean_mem()
    return big
