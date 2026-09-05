# ⚡ FlashVSR

**Towards Real-Time Diffusion-Based Streaming Video Super-Resolution**

**Authors:** Junhao Zhuang, Shi Guo, Xin Cai, Xiaohui Li, Yihao Liu, Chun Yuan, Tianfan Xue

<a href='http://zhuang2002.github.io/FlashVSR'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
<a href="https://huggingface.co/JunhaoZhuang/FlashVSR"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20(v1)-blue"></a> &nbsp;
<a href="https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20(v1.1)-blue"></a> &nbsp;
<a href="https://huggingface.co/datasets/JunhaoZhuang/VSR-120K"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-orange"></a> &nbsp;
<a href="https://arxiv.org/abs/2510.12747"><img src="https://img.shields.io/badge/arXiv-2510.12747-b31b1b.svg"></a>

**Your star means a lot for us to develop this project!** :star:

<img src="./examples/WanVSR/assets/teaser.png" />

---

### 🌟 Abstract

Diffusion models have recently advanced video restoration, but applying them to real-world video super-resolution (VSR) remains challenging due to high latency, prohibitive computation, and poor generalization to ultra-high resolutions. Our goal in this work is to make diffusion-based VSR practical by achieving **efficiency, scalability, and real-time performance**. To this end, we propose **FlashVSR**, the first diffusion-based one-step streaming framework towards real-time VSR. **FlashVSR runs at ∼17 FPS for 768 × 1408 videos on a single A100 GPU** by combining three complementary innovations: (i) a train-friendly three-stage distillation pipeline that enables streaming super-resolution, (ii) locality-constrained sparse attention that cuts redundant computation while bridging the train–test resolution gap, and (iii) a tiny conditional decoder that accelerates reconstruction without sacrificing quality. To support large-scale training, we also construct **VSR-120K**, a new dataset with 120k videos and 180k images. Extensive experiments show that FlashVSR scales reliably to ultra-high resolutions and achieves **state-of-the-art performance with up to ∼12× speedup** over prior one-step diffusion VSR models.

---

### 📰 News

- **Nov 2025 — 🎉 [FlashVSR v1.1](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1) released:** enhanced stability + fidelity  
- **Oct 2025 — [FlashVSR v1](https://huggingface.co/JunhaoZhuang/FlashVSR)  (initial release)**: Inference code and model weights are available now! 🎉  
- **Bug Fix (October 21, 2025):** Fixed `local_attention_mask` update logic to prevent artifacts when switching between different aspect ratios during continuous inference.  
- **Coming Soon:** Dataset release (**VSR-120K**) for large-scale training.

---
### 🌐 Community Integrations

Thanks to the community for the fast adoption of FlashVSR! Below are some third-party integrations:

**ComfyUI Support**
- **[smthemex/ComfyUI_FlashVSR](https://github.com/smthemex/ComfyUI_FlashVSR)** — closer to the official implementation
- **[tl2012tl/TE-Speed-FlashVSR](https://github.com/tl2012tl/TE-Speed-FlashVSR)** — adds motion‑aware dynamic acceleration, SpargeAttn sparse‑attention support and an optimized video‑combining encoder
- **[lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast](https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast)** — modified attention behavior, easier installation, and added `tile_dit`; I have not personally tested this version
- **[naxci1/ComfyUI-FlashVSR_Stable](https://github.com/naxci1/ComfyUI-FlashVSR_Stable)** — community-maintained stable ComfyUI implementation with VRAM optimizations
- **WanVideoWrapper** — integrated support but currently has known issues  
  https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/1441

**Cloud / API Deployments**  
(These third-party services offer ready-to-use online inference, making it easy to try FlashVSR without any setup or GPU requirements. However, it’s unclear whether they run v1 or v1.1 or whether the full pipeline is implemented, so results may differ from the official version. 🤷‍♂️ For the most accurate and complete reproduction, we recommend using the official repository when possible.)

- fal.ai: https://fal.ai/models/fal-ai/flashvsr/upscale/video  
- WaveSpeed AI: https://wavespeed.ai/models/wavespeed-ai/flashvsr  
- Segmind: https://www.segmind.com/models/flashvsr  
- Genbo AI: https://genbo.ai/models/toVideo/Flash-VSR
- JAI Portal: https://www.jaiportal.com/model/flashvsr
- cnaps.ai: https://cnaps.ai
- FlashVSR Online Service (third-party): https://flashvsr.org  
- GigapixelAI Video Upscaler (FlashVSR option): https://gigapixelai.com/ai-video-upscaler
---

### 📢 Important Quality Note (ComfyUI & other third-party implementations)

First of all, huge thanks to the community for the fast adoption, feedback, and contributions to FlashVSR! 🙌  
During community testing, we noticed that some third-party implementations of FlashVSR (e.g. early ComfyUI versions) do **not include our Locality-Constrained Sparse Attention (LCSA)** module and instead fall back to **dense attention**. This may lead to **noticeable quality degradation**, especially at higher resolutions.  
Community discussion: https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/1441

Below is a comparison example provided by a community member:

| Fig.1 – LR Input Video | Fig.2 – 3rd-party (no LCSA) | Fig.3 – Official FlashVSR |
|------------------|-----------------------------------------------|--------------------------------------|
| <video src="https://github.com/user-attachments/assets/ea12a191-48d5-47c0-a8e5-e19ad13581a9" controls width="260"></video> | <video src="https://github.com/user-attachments/assets/c8e53bd5-7eca-420d-9cc6-2b9c06831047" controls width="260"></video> | <video src="https://github.com/user-attachments/assets/a4d80618-d13d-4346-8e37-38d2fabf9827" controls width="260"></video> |

✅ The **official FlashVSR pipeline (this repository)**:
- **Better preserves fine structures and details**
- **Effectively avoids texture aliasing and visual artifacts**

Thanks again to the community for actively testing and helping improve FlashVSR together! 🚀

---

### 📋 TODO

- ✅ Release inference code and model weights  
- ⬜ Release dataset (VSR-120K)

---

### 🚀 Getting Started

Follow these steps to set up and run **FlashVSR** on your local machine:

> ⚠️ **Note:** This project is primarily designed and optimized for **4× video super-resolution**.  
> We **strongly recommend** using the **4× SR setting** to achieve better results and stability. ✅

#### 1️⃣ Clone the Repository

```bash
git clone https://github.com/OpenImagingLab/FlashVSR
cd FlashVSR
````

#### 2️⃣ Set Up the Python Environment

Create and activate the environment (**Python 3.11.13**):

```bash
conda create -n flashvsr python=3.11.13
conda activate flashvsr
```

Install project dependencies:

```bash
pip install -e .
pip install -r requirements.txt
```

#### 3️⃣ Install Block-Sparse Attention (Required)

FlashVSR relies on the **Block-Sparse Attention** backend to enable flexible and dynamic attention masking for efficient inference.

> **⚠️ Note:**
>
> * The Block-Sparse Attention build process can be memory-intensive, especially when compiling in parallel with multiple `ninja` jobs. It is recommended to keep sufficient memory available during compilation to avoid OOM errors. Once the build is complete, runtime memory usage is stable and not an issue.
> * Based on our testing, the Block-Sparse Attention backend works correctly on **NVIDIA A100 and A800** (Ampere) with **ideal acceleration performance**, and it also runs correctly on **H200** (Hopper) but the acceleration is limited due to hardware scheduling differences and sparse kernel behavior. **Compatibility and performance on other GPUs (e.g., RTX 40/50 series or H800) are currently unknown**. For more details, please refer to the official documentation: https://github.com/mit-han-lab/Block-Sparse-Attention


```bash
# ✅ Recommended: clone and install in a separate clean folder (outside the FlashVSR repo)
git clone https://github.com/mit-han-lab/Block-Sparse-Attention
cd Block-Sparse-Attention
pip install packaging
pip install ninja
python setup.py install
```

#### 4️⃣ Download Model Weights from Hugging Face

FlashVSR provides both **v1** and **v1.1** model weights on Hugging Face (via **Git LFS**).  
Please install Git LFS first:

```bash
# From the repo root
cd examples/WanVSR

# Install Git LFS (once per machine)
git lfs install

# Clone v1 (original) or v1.1 (recommended)
git lfs clone https://huggingface.co/JunhaoZhuang/FlashVSR          # v1
# or
git lfs clone https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1      # v1.1
```

After cloning, you should have one of the following folders:

```
./examples/WanVSR/FlashVSR/          # v1
./examples/WanVSR/FlashVSR-v1.1/     # v1.1
│
├── LQ_proj_in.ckpt
├── TCDecoder.ckpt
├── Wan2.1_VAE.pth
├── diffusion_pytorch_model_streaming_dmd.safetensors
└── README.md
```

> Inference scripts automatically load weights from the corresponding folder.

---

#### 5️⃣ Run Inference

```bash
# From the repo root
cd examples/WanVSR

# v1 (original)
python infer_flashvsr_full.py
# or
python infer_flashvsr_tiny.py
# or
python infer_flashvsr_tiny_long_video.py

# v1.1 (recommended)
python infer_flashvsr_v1.1_full.py
# or
python infer_flashvsr_v1.1_tiny.py
# or
python infer_flashvsr_v1.1_tiny_long_video.py
```

---

### ⚡ Hopper Acceleration (optional, GH200 / sm_90)

FlashVSR ships a set of **opt-in** fast paths for NVIDIA Hopper GPUs (e.g. GH200, `sm_90`). They are controlled by environment variables and are **all OFF by default**: with no variables set, the output is **bit-for-bit identical** to the standard path and Ampere / A100 are unaffected. Recoverable kernel/setup failures fall back to the original path and are exposed through telemetry. Direct-output failures after stateful decoding begins fail closed because replaying mutated decoder state is unsafe.

On a GH200 at 768x1408, the current production stack reaches **57.26 FPS core
E2E** for v1.1 Tiny (F=81): the bit-identical stack reaches 54.61 FPS
(+10.9% over the 49.23 FPS Phase-5 baseline), and the quality-gated
`TCDECODER_CUDNN_FUSED` path (55.4 dB PSNR vs the lossless stack, gate 49 dB)
plus the Phase-7 stack (49.59 dB PSNR vs the prior production set, gate 49 dB)
add the remaining gain.

| Env var | Values | Default | Effect |
|---|---|---|---|
| `FLASHVSR_CONV3D_BACKEND` | `auto`, `gemm` | `auto` | `gemm` = im2col + WGMMA conv3d for the LQ projector (largest single win) |
| `FLASHVSR_TCDECODER_CHANNELS_LAST` | `0`, `1` | `0` | NHWC TCDecoder (bit-identical) |
| `FLASHVSR_FUSE_NORM` | `0`, `1` | `0` | fuse norm / modulate / gate via `torch.compile` |
| `FLASHVSR_DIT_ROW_FUSION` | `0`, `1` | `0` | Triton affine-free LayerNorm + AdaLN and residual-gate fusion — quality-gated |
| `FLASHVSR_ATTN_BACKEND` | `sparse`, `triton`, `triton2`, `auto`, `dense` | `sparse` | `triton2` = warp-specialized Hopper block-sparse kernel (same mask) |
| `FLASHVSR_ATTN_TMA` | `0`, `1` | `1` | TMA bulk loads (only used by the `triton` backend) |
| `FLASHVSR_CONV3D_IM2COL_BUDGET_GB` | float | `2.0` | chunked im2col memory budget for the `gemm` backend |
| `FLASHVSR_CACHE_MOD` | `0`, `1` | `0` | cache step-invariant modulation (bit-identical) |
| `FLASHVSR_CACHE_MASK_BIAS` | `0`, `1` | `0` | cache the geometry-only attention bias (bit-identical) |
| `FLASHVSR_FUSE_ROPE` | `0`, `1` | `0` | fused single-kernel RoPE apply, same fp64 math (bit-identical) |
| `FLASHVSR_KV_RINGBUF` | `0`, `1` | `0` | preallocated KV-cache arena, removes the per-chunk KV concat (bit-identical; retains a little extra memory, see `FLASHVSR_KV_RINGBUF_SPARE`) |
| `FLASHVSR_KV_RINGBUF_SPARE` | int | `2` | arena spare slots: higher = rarer compaction copies, more retained memory |
| `FLASHVSR_ATTN_STRIDED_IO` | `0`, `1` | `0` | strided q/k/v/out for the `triton` backend — removes all attention-path transpose copies (bit-identical) |
| `FLASHVSR_MASKGEN_LEAN` | `0`, `1` | `0` | mask-generation cleanup (kthvalue select, no repeat copy, cached seqlens) — exact same mask (bit-identical) |
| `FLASHVSR_MASKGEN_THRESHOLD_CACHE` | `0`, `1` | `0` | reuse each DiT block's previous steady-chunk threshold — quality-gated |
| `FLASHVSR_LQPROJ_LEAN` | `0`, `1` | `0` | LQ projector: single-materialization causal pad + no cache clones (bit-identical) |
| `FLASHVSR_CACHE_ROPE_FREQS` | `0`, `1` | `0` | assemble RoPE freqs on-device in a cached buffer (bit-identical; useful when the CPU is loaded) |
| `FLASHVSR_CONV3D_PACKER` | `eager`, `triton` | `eager` | exact Triton im2col packing; retains the existing `torch.addmm` numerics |
| `FLASHVSR_TCDECODER_POINTER_STATE` | `0`, `1` | `0` | rotate recurrent-state tensor references instead of copying them |
| `FLASHVSR_TCDECODER_DIRECT_OUTPUT` | `0`, `1` | `0` | write overlapped decoder chunks directly into the final output |
| `FLASHVSR_TCDECODER_FUSE_POINTWISE` | `0`, `1` | `0` | exact BF16 MemBlock bias/ReLU/residual Triton fusion |
| `FLASHVSR_TCDECODER_UPSAMPLE` | `0`, `1` | `0` | exact channels-last nearest-neighbor Triton kernel |
| `FLASHVSR_TCDECODER_CONCAT` | `0`, `1` | `0` | exact channels-last recurrent concat Triton kernel |
| `FLASHVSR_TCDECODER_TGROW_UP` | `0`, `1` | `0` | run the 1x1 TGrow conv at low resolution and fuse temporal unpack + nearest upsample into one Triton kernel (measured bit-identical E2E) |
| `FLASHVSR_TCDECODER_CUDNN_FUSED` | `0`, `1` | `0` | cuDNN runtime-fused Conv+Bias(+Add)+ReLU decoder engines — **quality-gated** (~55 dB PSNR vs the lossless stack), not bit-exact |
| `FLASHVSR_TCDECODER_SPLITK_CONV` | `0`, `1` | `0` | avoid recurrent MemBlock concat via split-weight cuDNN convs; requires `TCDECODER_CUDNN_FUSED=1`, quality-gated |

**Recommended full-speed config** (run from `examples/WanVSR`):

```bash
FLASHVSR_CONV3D_BACKEND=gemm \
FLASHVSR_TCDECODER_CHANNELS_LAST=1 \
FLASHVSR_FUSE_NORM=1 \
FLASHVSR_DIT_ROW_FUSION=1 \
FLASHVSR_ATTN_BACKEND=triton2 \
FLASHVSR_CACHE_MOD=1 \
FLASHVSR_CACHE_MASK_BIAS=1 \
FLASHVSR_CACHE_ROPE_FREQS=0 \
FLASHVSR_FUSE_ROPE=1 \
FLASHVSR_KV_RINGBUF=1 \
FLASHVSR_ATTN_STRIDED_IO=1 \
FLASHVSR_MASKGEN_LEAN=1 \
FLASHVSR_MASKGEN_THRESHOLD_CACHE=1 \
FLASHVSR_LQPROJ_LEAN=1 \
FLASHVSR_FUSED_CSR=1 \
FLASHVSR_ROPE_KERNEL=triton \
FLASHVSR_POOLED_K_CACHE=1 \
FLASHVSR_ATTN_ZEROCOPY=1 \
FLASHVSR_DECODER_OVERLAP=1 \
FLASHVSR_FP8_GEMM=0 \
FLASHVSR_CONV3D_PACKER=triton \
FLASHVSR_TCDECODER_POINTER_STATE=1 \
FLASHVSR_TCDECODER_DIRECT_OUTPUT=1 \
FLASHVSR_TCDECODER_FUSE_POINTWISE=1 \
FLASHVSR_TCDECODER_UPSAMPLE=1 \
FLASHVSR_TCDECODER_CONCAT=1 \
FLASHVSR_TCDECODER_TGROW_UP=1 \
FLASHVSR_TCDECODER_CUDNN_FUSED=1 \
FLASHVSR_TCDECODER_SPLITK_CONV=1 \
python infer_flashvsr_v1.1_tiny.py
```

The same preset is available as:

```bash
./run_flashvsr_v1.1_tiny_gh200.sh
```

> **Notes**
> - The `triton`/`triton2` attention backends, Triton decoder/LQ kernels, and `gemm` Conv3D require a Hopper GPU (`sm_90`); unsupported paths fall back automatically.
> - `TCDECODER_DIRECT_OUTPUT` requires `DECODER_OVERLAP=1`; decoder Triton paths and `TCDECODER_SPLITK_CONV` require `TCDECODER_CHANNELS_LAST=1`; split-K also requires `TCDECODER_CUDNN_FUSED=1`; and `CONV3D_PACKER=triton` requires `CONV3D_BACKEND=gemm`.
> - `channels_last`, `CACHE_MOD`, `CACHE_MASK_BIAS`, and the Phase-2A knobs (`FUSE_ROPE`, `KV_RINGBUF`, `ATTN_STRIDED_IO`, `MASKGEN_LEAN`, `LQPROJ_LEAN`, `CACHE_ROPE_FREQS`) are bit-identical vs the same config without them (`max|diff| = 0`, see `examples/WanVSR/profiling/PHASE_BENCH_LOG.md`). `FUSE_NORM` and the `triton` backend are near-identical (~49-50 dB PSNR vs the default), not bit-exact, due to fp/accumulation order, so they are opt-in.
> - Parity + speed for each path can be checked with the `examples/WanVSR/test_*.py` scripts (`test_phase2a_lossless.py` covers the Phase-2A knobs; `test_phase5_lossless.py` modes `phase6`/`tgrowup`/`cudnnfuse` cover the Phase-6/6b decoder knobs; `test_tcdecoder_tgrow_up.py` is the isolated TGROW_UP kernel test).
> - Phase-2A stack measured on GH200: 38.6 → 41.6 FPS @768x1408 (steady chunk 156 → 138 ms) and 11.0 → 11.5 FPS @1536x2560; `KV_RINGBUF` retains ~+3 GiB @768 (tunable via `FLASHVSR_KV_RINGBUF_SPARE`).
> - Phase-6 production stack measured on GH200: 49.23 → 53.42 FPS @768x1408 and 13.44 → 14.68 FPS @1536x2560. All Phase-6 output changes are bit-identical against the Phase-5 production stack, including the accepted `8n+5` frame-count edge case.
> - Phase-6b: `TCDECODER_TGROW_UP` (bit-identical vs the Phase-6 stack, 53.40 → 54.61 FPS @768 3-run median; single-run spot-check 14.36 → 14.69 FPS @1536x2560 F=41, +2.30%) and `TCDECODER_CUDNN_FUSED` (quality-gated, not bit-exact: 54.58 → 55.91 FPS @768 3-run median; 14.69 → 15.07 FPS @1536 with TGROW_UP already on, +2.59%; E2E 55.4–55.8 dB PSNR vs the lossless stack, gate 49 dB). Disable `TCDECODER_CUDNN_FUSED` for a strictly `max|diff|=0` pipeline; `TCDECODER_TGROW_UP` stays on in that case.
> - Phase-7: row-wise DiT fusion, steady-threshold reuse, and MemBlock split-K move the quality-gated production stack **55.91 → 57.26 FPS** @768x1408 F=81 (3-run medians, +2.41%). Combined E2E PSNR is 49.81 dB at F=29 and 49.59 dB at F=89 vs the preceding production stack (gate 49 dB). The @1536x2560 spot reached 15.42 FPS at F=41 with 49.95 dB at F=29.
> - Use `FLASHVSR_REQUIRE_FASTPATHS=1` with `profiling/run_pipe_target.py` to fail the benchmark if any requested backend silently falls back.

---

### 🛠️ Method

The overview of **FlashVSR**. This framework features:

* **Three-Stage Distillation Pipeline** for streaming VSR training.
* **Locality-Constrained Sparse Attention** to cut redundant computation and bridge the train–test resolution gap.
* **Tiny Conditional Decoder** for efficient, high-quality reconstruction.
* **VSR-120K Dataset** consisting of **120k videos** and **180k images**, supports joint training on both images and videos.

<img src="./examples/WanVSR/assets/flowchart.jpg" width="1000" />

---

### 🤗 Feedback & Support

We welcome feedback and issues. Thank you for trying **FlashVSR**!

---

### 📄 Acknowledgments

We gratefully acknowledge the following open-source projects:

* **DiffSynth Studio** — [https://github.com/modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
* **Block-Sparse-Attention** — [https://github.com/mit-han-lab/Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention)
* **taehv** — [https://github.com/madebyollin/taehv](https://github.com/madebyollin/taehv)

---

### 📞 Contact

* **Junhao Zhuang**
  Email: [zhuangjh23@tsinghua.org.cn](mailto:zhuangjh23@tsinghua.org.cn)

---

### 📜 Citation

```bibtex
@article{zhuang2025flashvsr,
  title={FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution},
  author={Zhuang, Junhao and Guo, Shi and Cai, Xin and Li, Xiaohui and Liu, Yihao and Yuan, Chun and Xue, Tianfan},
  journal={arXiv preprint arXiv:2510.12747},
  year={2025}
}
```
