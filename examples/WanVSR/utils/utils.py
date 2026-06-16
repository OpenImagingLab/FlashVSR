from einops import rearrange, repeat

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time


CACHE_T = 2


# ---------------------------------------------------------------------------
# Hopper (sm_90 / GH200) 3D-conv acceleration: im2col + bf16 GEMM (tensor core)
#
# cuDNN's 3D-conv path for the LQ-projector kernels (4x3x3, stride (2,1,1),
# large channels) fails to pick a tensor-core implicit-GEMM and runs at
# ~28 TFLOP/s (~2.8% peak) on GH200. Reformulating the same convolution as an
# explicit im2col (unfold) + bf16 matmul saturates the Hopper WGMMA tensor
# cores (~400+ TFLOP/s, ~15-17x faster), with bit-for-bit identical math.
#
# Knob (opt-in, default = current nn.Conv3d behaviour):
#   FLASHVSR_CONV3D_BACKEND = auto | gemm
# Guarded to sm_90; falls back to nn.Conv3d on any error / OOM.
# See docs/hopper_conv3d_acceleration.md.
# ---------------------------------------------------------------------------

_CONV3D_BACKEND = os.environ.get("FLASHVSR_CONV3D_BACKEND", "auto").lower()

# Phase 2: per-chunk im2col patch memory budget (GB). The full im2col patch
# tensor for conv2 is ~9.5 GB at 1920x2560; we tile the output-H axis so the
# transient patch stays within this budget (default ~2 GB) with no measurable
# speed loss and bit-identical math. Set <=0 to disable chunking (Phase-1
# single-shot behaviour).
_CONV3D_IM2COL_BUDGET_GB = float(
    os.environ.get("FLASHVSR_CONV3D_IM2COL_BUDGET_GB", "2.0")
)


def _is_hopper(device):
    try:
        if device is None or device.type != "cuda":
            return False
        return torch.cuda.get_device_capability(device) == (9, 0)
    except Exception:
        return False


def _im2col_gemm_rows(x, weight, bias, stride, h0, h1):
    """im2col + GEMM for output rows [h0, h1). Returns (N, Cout, To, h1-h0, Wo)."""
    N, Cin, T, _, W = x.shape
    Cout = weight.shape[0]
    kt, kh, kw = weight.shape[2], weight.shape[3], weight.shape[4]
    st, sh, sw = stride
    # input rows feeding output rows [h0,h1): [h0*sh : (h1-1)*sh + kh)
    xs = x[:, :, :, h0 * sh:(h1 - 1) * sh + kh, :]
    patches = (
        xs.unfold(2, kt, st)
        .unfold(3, kh, sh)
        .unfold(4, kw, sw)
    )
    To, Ho, Wo = patches.shape[2], patches.shape[3], patches.shape[4]
    patches = patches.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(
        N * To * Ho * Wo, Cin * kt * kh * kw
    )
    wmat = weight.reshape(Cout, Cin * kt * kh * kw).t()
    out = torch.addmm(bias, patches, wmat) if bias is not None else patches @ wmat
    return out.reshape(N, To, Ho, Wo, Cout).permute(0, 4, 1, 2, 3)


def _conv3d_gemm(x, weight, bias, stride):
    """Core (padding-free) 3D convolution as im2col + GEMM.

    `x` must already be padded by the caller (CausalConv3d applies the causal
    replicate-pad + streaming cache before calling this). This routine only
    performs the strided convolution arithmetic, identical to
    F.conv3d(x, weight, bias, stride=stride, padding=0).

    Phase 2: the im2col patch is built in chunks along the output-H axis so the
    transient memory stays bounded (~_CONV3D_IM2COL_BUDGET_GB) regardless of
    resolution. Math is bit-identical to the single-shot path.
    """
    N, Cin, T, H, W = x.shape
    Cout, _, kt, kh, kw = weight.shape
    st, sh, sw = stride

    To = (T - kt) // st + 1
    Ho = (H - kh) // sh + 1
    Wo = (W - kw) // sw + 1

    # Choose chunk height so that one chunk's patch tensor fits the budget.
    # patch bytes per output row = N*To*Wo * (Cin*kt*kh*kw) * itemsize
    elems_per_row = N * To * Wo * (Cin * kt * kh * kw)
    bytes_per_row = elems_per_row * x.element_size()
    if _CONV3D_IM2COL_BUDGET_GB <= 0 or bytes_per_row == 0:
        rows = Ho
    else:
        budget = int(_CONV3D_IM2COL_BUDGET_GB * 1e9)
        rows = max(1, min(Ho, budget // max(1, bytes_per_row)))

    if rows >= Ho:
        # single shot (Phase-1 path)
        return _im2col_gemm_rows(x, weight, bias, stride, 0, Ho).contiguous()

    out = torch.empty(N, Cout, To, Ho, Wo, device=x.device, dtype=x.dtype)
    for h0 in range(0, Ho, rows):
        h1 = min(h0 + rows, Ho)
        out[:, :, :, h0:h1, :] = _im2col_gemm_rows(x, weight, bias, stride, h0, h1)
    return out


class RMS_norm(nn.Module):

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        return F.normalize(
            x, dim=(1 if self.channel_first else
                    -1)) * self.scale * self.gamma + self.bias

class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            # print(cache_x.shape, x.shape)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
            # print('cache!')
        x = F.pad(x, padding, mode='replicate') # mode='replicate'
        # print(x[0,0,:,0,0])

        # Hopper im2col+GEMM backend (opt-in, guarded, with fallback). The
        # causal replicate-pad + streaming cache above is already applied; only
        # the core padding-free conv is rerouted to a tensor-core GEMM.
        if _CONV3D_BACKEND == "gemm" and _is_hopper(x.device):
            try:
                return _conv3d_gemm(x, self.weight, self.bias, self.stride)
            except Exception:
                torch.cuda.empty_cache()
                # fall through to the standard cuDNN path

        return super().forward(x)
    
class PixelShuffle3d(nn.Module):
    def __init__(self, ff, hh, ww):
        super().__init__()
        self.ff = ff
        self.hh = hh
        self.ww = ww

    def forward(self, x):
        # x: (B, C, F, H, W)
        return rearrange(x, 
                         'b c (f ff) (h hh) (w ww) -> b (c ff hh ww) f h w',
                         ff=self.ff, hh=self.hh, ww=self.ww)

class Buffer_LQ4x_Proj(nn.Module):

    def __init__(self, in_dim, out_dim, layer_num=30):
        super().__init__()
        self.ff = 1
        self.hh = 16
        self.ww = 16
        self.hidden_dim1 = 2048
        self.hidden_dim2 = 3072
        self.layer_num = layer_num

        self.pixel_shuffle = PixelShuffle3d(self.ff, self.hh, self.ww)

        self.conv1 = CausalConv3d(in_dim*self.ff*self.hh*self.ww, self.hidden_dim1, (4, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)) # f -> f/2 h -> h w -> w
        self.norm1 = RMS_norm(self.hidden_dim1, images=False)
        self.act1 = nn.SiLU()

        self.conv2 = CausalConv3d(self.hidden_dim1, self.hidden_dim2, (4, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)) # f -> f/2 h -> h w -> w
        self.norm2 = RMS_norm(self.hidden_dim2, images=False)
        self.act2 = nn.SiLU()

        self.linear_layers = nn.ModuleList([nn.Linear(self.hidden_dim2, out_dim) for _ in range(layer_num)])

        self.clip_idx = 0

    def forward(self, video):
        self.clear_cache()
        # x: (B, C, F, H, W)
        
        t = video.shape[2]
        iter_ = 1 + (t - 1) // 4
        first_frame = video[:, :, :1, :, :].repeat(1, 1, 3, 1, 1)
        video = torch.cat([first_frame, video], dim=2)
        # print(video.shape)

        out_x = []
        for i in range(iter_):
            x = self.pixel_shuffle(video[:,:,i*4:(i+1)*4,:,:])
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache['conv1'] = cache1_x
            x = self.conv1(x, self.cache['conv1'])
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache['conv2'] = cache2_x
            if i == 0:
                continue
            x = self.conv2(x, self.cache['conv2'])
            x = self.norm2(x)
            x = self.act2(x)
            out_x.append(x)
        out_x = torch.cat(out_x, dim = 2)
        # print(out_x.shape)
        out_x = rearrange(out_x, 'b c f h w -> b (f h w) c')
        outputs = []
        for i in range(self.layer_num):
            outputs.append(self.linear_layers[i](out_x))
        return outputs

    def clear_cache(self):
        self.cache = {}
        self.cache['conv1'] = None
        self.cache['conv2'] = None
        self.clip_idx = 0
    
    def stream_forward(self, video_clip):
        if self.clip_idx == 0:
            # self.clear_cache()
            first_frame = video_clip[:, :, :1, :, :].repeat(1, 1, 3, 1, 1)
            video_clip = torch.cat([first_frame, video_clip], dim=2)
            x = self.pixel_shuffle(video_clip)
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache['conv1'] = cache1_x
            x = self.conv1(x, self.cache['conv1'])
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache['conv2'] = cache2_x
            self.clip_idx += 1
            return None
        else:
            x = self.pixel_shuffle(video_clip)
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache['conv1'] = cache1_x
            x = self.conv1(x, self.cache['conv1'])
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache['conv2'] = cache2_x
            x = self.conv2(x, self.cache['conv2'])
            x = self.norm2(x)
            x = self.act2(x)
            out_x = rearrange(x, 'b c f h w -> b (f h w) c')
            outputs = []
            for i in range(self.layer_num):
                outputs.append(self.linear_layers[i](out_x))
            self.clip_idx += 1
            return outputs

class Causal_LQ4x_Proj(nn.Module):

    def __init__(self, in_dim, out_dim, layer_num=30):
        super().__init__()
        self.ff = 1
        self.hh = 16
        self.ww = 16
        self.hidden_dim1 = 2048
        self.hidden_dim2 = 3072
        self.layer_num = layer_num

        self.pixel_shuffle = PixelShuffle3d(self.ff, self.hh, self.ww)

        self.conv1 = CausalConv3d(in_dim*self.ff*self.hh*self.ww, self.hidden_dim1, (4, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)) # f -> f/2 h -> h w -> w
        self.norm1 = RMS_norm(self.hidden_dim1, images=False)
        self.act1 = nn.SiLU()

        self.conv2 = CausalConv3d(self.hidden_dim1, self.hidden_dim2, (4, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)) # f -> f/2 h -> h w -> w
        self.norm2 = RMS_norm(self.hidden_dim2, images=False)
        self.act2 = nn.SiLU()

        self.linear_layers = nn.ModuleList([nn.Linear(self.hidden_dim2, out_dim) for _ in range(layer_num)])

        self.clip_idx = 0

    def forward(self, video):
        self.clear_cache()
        # x: (B, C, F, H, W)
        
        t = video.shape[2]
        iter_ = 1 + (t - 1) // 4
        first_frame = video[:, :, :1, :, :].repeat(1, 1, 3, 1, 1)
        video = torch.cat([first_frame, video], dim=2)
        # print(video.shape)

        out_x = []
        for i in range(iter_):
            x = self.pixel_shuffle(video[:,:,i*4:(i+1)*4,:,:])
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv1(x, self.cache['conv1'])
            self.cache['conv1'] = cache1_x
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            if i == 0:
                self.cache['conv2'] = cache2_x
                continue
            x = self.conv2(x, self.cache['conv2'])
            self.cache['conv2'] = cache2_x
            x = self.norm2(x)
            x = self.act2(x)
            out_x.append(x)
        out_x = torch.cat(out_x, dim = 2)
        out_x = rearrange(out_x, 'b c f h w -> b (f h w) c')
        outputs = []
        for i in range(self.layer_num):
            outputs.append(self.linear_layers[i](out_x))
        return outputs

    def clear_cache(self):
        self.cache = {}
        self.cache['conv1'] = None
        self.cache['conv2'] = None
        self.clip_idx = 0
    
    def stream_forward(self, video_clip):
        if self.clip_idx == 0:
            # self.clear_cache()
            first_frame = video_clip[:, :, :1, :, :].repeat(1, 1, 3, 1, 1)
            video_clip = torch.cat([first_frame, video_clip], dim=2)
            x = self.pixel_shuffle(video_clip)
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv1(x, self.cache['conv1'])
            self.cache['conv1'] = cache1_x
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache['conv2'] = cache2_x
            self.clip_idx += 1
            return None
        else:
            x = self.pixel_shuffle(video_clip)
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv1(x, self.cache['conv1'])
            self.cache['conv1'] = cache1_x
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv2(x, self.cache['conv2'])
            self.cache['conv2'] = cache2_x
            x = self.norm2(x)
            x = self.act2(x)
            out_x = rearrange(x, 'b c f h w -> b (f h w) c')
            outputs = []
            for i in range(self.layer_num):
                outputs.append(self.linear_layers[i](out_x))
            self.clip_idx += 1
            return outputs