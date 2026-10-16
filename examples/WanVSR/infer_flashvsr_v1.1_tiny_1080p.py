#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime
from typing import Optional, Tuple, Dict
import torch.nn.functional as F
import os, re, time, subprocess, shutil
import numpy as np
from PIL import Image
import imageio
from tqdm import tqdm
import torch
from einops import rearrange

from diffsynth import ModelManager, FlashVSRTinyPipeline
from utils.utils import Causal_LQ4x_Proj
from utils.TCDecoder import build_tcdecoder


TARGET_LONG_EDGE = 1920
TARGET_SHORT_EDGE = 1080

# 视频保存配置
USE_FFMPEG_SAVE = True
FFMPEG_ENCODER = "auto"       # auto / h264_nvenc / hevc_nvenc / libx264 ...
FFMPEG_PRESET = None          # None 表示使用对应编码器的默认 preset
FFMPEG_PIX_FMT = "yuv420p"    # 常用输出像素格式，兼容性较好
FFMPEG_THREADS = None         # None 表示由 ffmpeg 自行调度线程
FFMPEG_ENCODER_RESOLVED: Optional[str] = None


def adjust_frames_to_resolution_array(frames: np.ndarray, target_size):
    if target_size is None:
        return frames, None
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("adjust_frames_to_resolution_array 仅支持形状为 (T,H,W,3) 的数组")
    target_w, target_h = target_size
    T, H, W, C = frames.shape
    if W == target_w and H == target_h:
        return frames, None
    out = np.zeros((T, target_h, target_w, C), dtype=frames.dtype)
    applied_mode = None
    for idx in range(T):
        frame = frames[idx]
        h_i, w_i = frame.shape[:2]
        if w_i <= target_w and h_i <= target_h:
            pad_left = (target_w - w_i) // 2
            pad_top = (target_h - h_i) // 2
            pad_right = target_w - w_i - pad_left
            pad_bottom = target_h - h_i - pad_top
            canvas = out[idx]
            if pad_top:
                canvas[:pad_top, :, :] = 0
            if pad_bottom:
                canvas[-pad_bottom:, :, :] = 0
            if pad_left or pad_right:
                canvas[:, :pad_left, :] = 0
                canvas[:, -pad_right:, :] = 0
            canvas[pad_top:pad_top + h_i, pad_left:pad_left + w_i, :] = frame
            applied_mode = "pad"
        else:
            left = max((w_i - target_w) // 2, 0)
            top = max((h_i - target_h) // 2, 0)
            cropped = frame[top:top + target_h, left:left + target_w, :]
            if cropped.shape[0] != target_h or cropped.shape[1] != target_w:
                tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                resized = F.interpolate(tensor, size=(target_h, target_w), mode='bicubic', align_corners=False)
                resized = (resized.squeeze(0).permute(1, 2, 0) * 255.0).clamp(0, 255).byte().numpy()
                out[idx] = resized
                applied_mode = "crop+resize"
            else:
                out[idx] = cropped
                applied_mode = "crop"
    return out, applied_mode

def tensor_to_uint8_array(frames: torch.Tensor) -> np.ndarray:
    frames = rearrange(frames, "C T H W -> T H W C").contiguous()
    frames = ((frames.float() + 1) * 127.5).clamp(0, 255).round()
    return frames.to(dtype=torch.uint8).cpu().numpy()

def tensor2video(frames):
    frames = rearrange(frames, "C T H W -> T H W C")
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    frames = [Image.fromarray(frame) for frame in frames]
    return frames

def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', os.path.basename(name))]

def list_images_natural(folder: str):
    exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]
    fs.sort(key=natural_key)
    return fs

def largest_8n1_leq(n):  # 8n+1
    return 0 if n < 1 else ((n - 1)//8)*8 + 1

def is_video(path):
    return os.path.isfile(path) and path.lower().endswith(('.mp4','.mov','.avi','.mkv'))

def pil_to_tensor_neg1_1(img: Image.Image, dtype=torch.bfloat16, device='cuda'):
    t = torch.from_numpy(np.asarray(img, np.uint8)).to(device=device, dtype=torch.float32)  # HWC
    t = t.permute(2,0,1) / 255.0 * 2.0 - 1.0                                              # CHW in [-1,1]
    return t.to(dtype)

_FFMPEG_ENCODER_CACHE: Dict[Tuple[str, str], bool] = {}

def _resolve_ffmpeg_path() -> str:
    for name in ("ffmpeg", "ffmpeg.exe"):
        path = shutil.which(name)
        if path:
            return path
    raise FileNotFoundError("未在 PATH 中找到 ffmpeg，可安装后再启用 --fast-video-save")

def _ffmpeg_supports_encoder(ffmpeg_bin: str, encoder: str) -> bool:
    key = (ffmpeg_bin, encoder)
    if key in _FFMPEG_ENCODER_CACHE:
        return _FFMPEG_ENCODER_CACHE[key]
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel", "error",
        "-f", "lavfi",
        "-i", "color=c=black:s=2x2:d=0.1",
        "-frames:v", "1",
        "-c:v", encoder,
        "-f", "null",
        "-"
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        support = False
    else:
        support = result.returncode == 0
    _FFMPEG_ENCODER_CACHE[key] = support
    return support

def _select_ffmpeg_encoder(ffmpeg_bin: str, encoder_hint: str) -> str:
    global FFMPEG_ENCODER_RESOLVED
    if encoder_hint != "auto":
        if not _ffmpeg_supports_encoder(ffmpeg_bin, encoder_hint):
            raise RuntimeError(f"ffmpeg 不支持编码器 {encoder_hint}")
        FFMPEG_ENCODER_RESOLVED = encoder_hint
        return encoder_hint

    if FFMPEG_ENCODER_RESOLVED:
        return FFMPEG_ENCODER_RESOLVED

    candidates = []
    if torch.cuda.is_available():
        candidates.extend(["h264_nvenc", "hevc_nvenc"])
    candidates.extend(["libx264", "libx265"])
    for cand in candidates:
        if _ffmpeg_supports_encoder(ffmpeg_bin, cand):
            FFMPEG_ENCODER_RESOLVED = cand
            return cand
    raise RuntimeError("无法找到 ffmpeg 支持的编码器，请在配置中手动指定 FFMPEG_ENCODER")

def init_ffmpeg_encoder_global():
    if not USE_FFMPEG_SAVE:
        return
    try:
        ffmpeg_bin = _resolve_ffmpeg_path()
        encoder_used = _select_ffmpeg_encoder(ffmpeg_bin, FFMPEG_ENCODER)
    except Exception as exc:
        print(f"[Warn] FFmpeg 编码器预热失败: {exc}")
        return
    
def _quality_to_crf(q: int) -> int:
    q = int(round(q))
    q = max(1, min(10, q))
    crf = int(round(38 - 2.5 * q))
    return max(0, min(51, crf))

def _quality_to_nvenc_cq(q: int) -> int:
    q = int(round(q))
    q = max(1, min(10, q))
    cq = int(round(35 - 2.5 * q))
    return max(0, min(51, cq))

def save_video(frames, save_path, fps=30, quality=5):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    writer_kwargs = dict(fps=fps, quality=quality)
    try:
        w = imageio.get_writer(save_path, macro_block_size=None, **writer_kwargs)
    except TypeError:
        w = imageio.get_writer(save_path, **writer_kwargs)
    iterable = frames if isinstance(frames, (list, tuple)) else frames
    if isinstance(frames, np.ndarray):
        for f in tqdm(frames, desc=f"Saving {os.path.basename(save_path)}"):
            w.append_data(f)
    else:
        for f in tqdm(iterable, desc=f"Saving {os.path.basename(save_path)}"):
            w.append_data(np.array(f))
    w.close()

def save_video_ffmpeg(
    frames,
    save_path: str,
    fps: int = 30,
    quality: int = 6,
    *,
    encoder: str = "auto",
    preset: Optional[str] = None,
    pix_fmt: Optional[str] = "yuv420p",
    threads: Optional[int] = None,
):
    if isinstance(frames, np.ndarray):
        if frames.size == 0:
            raise ValueError("save_video_ffmpeg 收到空帧数组")
    else:
        if not frames:
            raise ValueError("save_video_ffmpeg 收到空帧列表")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ffmpeg_bin = _resolve_ffmpeg_path()
    array_input = isinstance(frames, np.ndarray)
    if array_input:
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("save_video_ffmpeg 仅支持 (T,H,W,3) 的 numpy 数组")
        frames = np.ascontiguousarray(frames, dtype=np.uint8)
        _, height, width, _ = frames.shape
    else:
        width, height = frames[0].size

    if not array_input:
        if any(f.size != frames[0].size for f in frames):
            raise ValueError("所有帧必须保持一致分辨率后再写入 ffmpeg")

    encoder_used = _select_ffmpeg_encoder(ffmpeg_bin, encoder)

    cmd = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel", "error",
    ]
    if threads is not None and threads > 0:
        cmd += ["-threads", str(threads)]
    cmd += [
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-c:v", encoder_used,
    ]

    if encoder_used.endswith("_nvenc"):
        cq = _quality_to_nvenc_cq(quality)
        cmd += ["-preset", preset or "p5", "-rc", "vbr", "-cq", str(cq), "-b:v", "0"]
    elif encoder_used.startswith("libx26"):
        crf = _quality_to_crf(quality)
        cmd += ["-preset", preset or "medium", "-crf", str(crf)]
    elif preset:
        cmd += ["-preset", preset]

    if pix_fmt:
        cmd += ["-pix_fmt", pix_fmt]

    cmd.append(save_path)

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    error_message = ""
    try:
        if array_input:
            try:
                proc.stdin.write(frames.tobytes())
            except BrokenPipeError:
                pass
        else:
            for f in tqdm(frames, desc=f"FFmpeg 保存 {os.path.basename(save_path)}"):
                if f.mode != "RGB":
                    frame_rgb = f.convert("RGB")
                else:
                    frame_rgb = f
                frame_array = np.asarray(frame_rgb, dtype=np.uint8)
                try:
                    proc.stdin.write(frame_array.tobytes())
                except BrokenPipeError:
                    break
        proc.stdin.close()
        error_message = proc.stderr.read().decode("utf-8", errors="ignore")
        return_code = proc.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg 编码失败，返回码 {return_code}，信息：{error_message}")
    finally:
        if proc.poll() is None:
            proc.kill()
        if proc.stderr:
            proc.stderr.close()
        if proc.stdin:
            proc.stdin.close()

    return encoder_used

def list_videos_recursive(folder: str):
    """递归获取所有视频路径并自然排序"""
    exts = ('.mp4', '.avi', '.mov', '.mkv')
    fs = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(exts):
                fs.append(os.path.join(root, f))
    fs.sort(key=natural_key)
    return fs

def compute_scaled_and_target_dims(w0: int, h0: int, scale: float = 4.0, multiple: int = 128):
    if w0 <= 0 or h0 <= 0:
        raise ValueError("Invalid original size")
    if scale <= 0:
        raise ValueError("scale must be > 0")

    long_edge = 1920
    short_edge = 1920*(h0/w0) if w0 >= h0 else 1920*(w0/h0)
    # 向上取128的整数倍
    short_edge = int(np.ceil(short_edge / multiple) * multiple)

    sW = long_edge if w0 >= h0 else short_edge
    sH = short_edge if h0 < w0 else long_edge

    tW = (sW // multiple) * multiple
    tH = (sH // multiple) * multiple

    if tW == 0 or tH == 0:
        raise ValueError(
            f"Scaled size too small ({sW}x{sH}) for multiple={multiple}. "
            f"Increase scale (got {scale})."
        )

    return sW, sH, tW, tH


def prepare_input_tensor_gpu(path: str, scale: float = 4, dtype=torch.bfloat16, device: str = 'cuda', *, chunk_size: int = 0):
    """
    GPU-oriented variant that keeps frames as uint8, batches them into a single tensor,
    and performs the expensive resizing/cropping in torch (optionally on GPU) to reduce
    repeated CPU-side work.
    """
    base_name = os.path.basename(path.rstrip('/\\'))

    if os.path.isdir(path):
        paths0 = list_images_natural(path)
        if not paths0:
            raise FileNotFoundError(f"No images in {path}")
        frames_np = []
        for p in paths0:
            with Image.open(p).convert('RGB') as img:
                frames_np.append(np.asarray(img, dtype=np.uint8))
        fps = 30
    elif is_video(path):
        rdr = imageio.get_reader(path)
        frames_np = []
        meta = {}
        try:
            try:
                meta = rdr.get_meta_data()
            except Exception:
                pass
            for frame in rdr:
                if frame.ndim == 2:
                    frame = np.stack([frame]*3, axis=-1)
                elif frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                frames_np.append(frame)
        finally:
            try:
                rdr.close()
            except Exception:
                pass
        fps_val = meta.get('fps', 30) if isinstance(meta, dict) else 30
        fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 30
    else:
        raise ValueError(f"Unsupported input: {path}")

    if not frames_np:
        raise RuntimeError(f"No frames decoded from {path}")

    h0, w0 = frames_np[0].shape[:2]
    
    original_frames = len(frames_np)
    print(f"[{base_name}] (fast) Original Resolution: {w0}x{h0} | Original Frames: {original_frames} | FPS: {fps}")

    sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
    print(f"[{base_name}] (fast) Scaled (x{scale:.2f}): {sW}x{sH} -> Target (128-multiple): {tW}x{tH}")

    frames_np.extend([frames_np[-1].copy() for _ in range(4)])
    target_frames = largest_8n1_leq(len(frames_np))
    if target_frames == 0:
        raise RuntimeError(f"Not enough frames after padding in {path}. Got {len(frames_np)}.")
    frames_np = frames_np[:target_frames]
    print(f"[{base_name}] (fast) Target Frames (8n-3): {target_frames-4}")

    frames_stack = np.stack(frames_np, axis=0)  # (F, H, W, C)
    frames_np = None  # release list references

    frames_tensor = torch.from_numpy(frames_stack).permute(0, 3, 1, 2).contiguous()  # (F, C, H, W)
    frames_stack = None

    if chunk_size and chunk_size > 0:
        chunks = []
        for start in range(0, frames_tensor.shape[0], chunk_size):
            chunk = frames_tensor[start:start+chunk_size].to(device=device, dtype=torch.float32, non_blocking=True) / 255.0
            chunk = F.interpolate(chunk, size=(sH, sW), mode='bicubic', align_corners=False)
            if sH != tH or sW != tW:
                top = (sH - tH) // 2
                left = (sW - tW) // 2
                chunk = chunk[:, :, top:top+tH, left:left+tW]
            chunk = (chunk * 2.0 - 1.0).to(dtype=dtype)
            chunks.append(chunk)
        frames_tensor = torch.cat(chunks, dim=0)
    else:
        frames_tensor = frames_tensor.to(device=device, dtype=torch.float32, non_blocking=True) / 255.0
        frames_tensor = F.interpolate(frames_tensor, size=(sH, sW), mode='bicubic', align_corners=False)
        if sH != tH or sW != tW:
            top = (sH - tH) // 2
            left = (sW - tW) // 2
            frames_tensor = frames_tensor[:, :, top:top+tH, left:left+tW]
        frames_tensor = (frames_tensor * 2.0 - 1.0).to(dtype=dtype)

    video = frames_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # 1 C F H W
    num_frames = frames_tensor.shape[0]
    short_edge = int(1920 * h0 / w0) if h0<w0 else int(1920 * w0 / h0)
    # if not divisible by 2, make it so
    if short_edge % 2 == 1:
        short_edge -= 1
    return video, tH, tW, num_frames, fps, short_edge


def init_pipeline():
    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_models([
        "./FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors"
    ])
    pipe = FlashVSRTinyPipeline.from_model_manager(mm, device="cuda")
    pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to("cuda", dtype=torch.bfloat16)
    LQ_proj_in_path = "./FlashVSR-v1.1/LQ_proj_in.ckpt"
    if os.path.exists(LQ_proj_in_path):
        pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(LQ_proj_in_path, map_location="cpu"), strict=True)
    pipe.denoising_model().LQ_proj_in.to('cuda')

    multi_scale_channels = [512, 256, 128, 128]
    pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, new_latent_channels=16+768)
    mis = pipe.TCDecoder.load_state_dict(torch.load("./FlashVSR-v1.1/TCDecoder.ckpt"), strict=False)

    pipe.to('cuda'); pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv();
    pipe.load_models_to_device(["dit","vae"])

    return pipe

def main():
    INPUT_ROOT = "/data/cjh/VSR/FlashVSR/examples/WanVSR/Archive_2"
    RESULT_ROOT = "./results_Archive_2"
    os.makedirs(RESULT_ROOT, exist_ok=True)

    inputs = list_videos_recursive(INPUT_ROOT)

    if USE_FFMPEG_SAVE:
        init_ffmpeg_encoder_global()

    seed, scale, dtype, device = 0, 1.5, torch.bfloat16, 'cuda'
    sparse_ratio = 2.0      # Recommended: 1.5 or 2.0. 1.5 → faster; 2.0 → more stable.
    pipe = init_pipeline()

    for p in inputs:
        torch.cuda.empty_cache(); torch.cuda.ipc_collect()
        name = os.path.basename(p.rstrip('/'))
        if name.startswith('.'):
            continue
        LQ, th, tw, F, fps, target_short = prepare_input_tensor_gpu(p, scale=scale, dtype=dtype, device=device, chunk_size=0)

        video = pipe(
            prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed,
            LQ_video=LQ, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
            topk_ratio=sparse_ratio*768*1280/(th*tw), 
            kv_ratio=3.0,
            local_range=11,  # Recommended: 9 or 11. local_range=9 → sharper details; 11 → more stable results.
            color_fix = True
        )
        save_path = os.path.join(RESULT_ROOT, f"FlashVSR_Tiny_{name.split('.')[0]}_seed{seed}.mp4")
        save_start = time.time()
        video_array = tensor_to_uint8_array(video)
        height, width = video_array.shape[1:3]
        target_resolution = (target_short, TARGET_LONG_EDGE) if height > width else (TARGET_LONG_EDGE, target_short)
        video_array, adjust_mode = adjust_frames_to_resolution_array(video_array, target_resolution)
        encoder_note = "imageio"
        if USE_FFMPEG_SAVE:
            encoder_used = save_video_ffmpeg(
                video_array,
                save_path,
                fps=fps,
                quality=6,
                encoder=FFMPEG_ENCODER,
                preset=FFMPEG_PRESET,
                pix_fmt=FFMPEG_PIX_FMT,
                threads=FFMPEG_THREADS,
            )
            encoder_note = f"ffmpeg:{encoder_used}"
        else:
            save_video(video_array, save_path, fps=fps, quality=6)
    print("Done.")

if __name__ == "__main__":
    main()
