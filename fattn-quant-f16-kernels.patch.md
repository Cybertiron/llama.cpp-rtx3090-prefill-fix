# Fix #27109: slow prefill with quantized KV cache + flash attention on Ampere

**VERIFIED on 2x RTX 3090 (Ampere, cc 8.6), Qwen3.8-27B, ctx 65536, ubatch 2048.**
Patch to `ggml/src/ggml-cuda/convert.cu`.

## Symptom
With `--flash-attn on` and quantized KV cache, prefill collapses on Ampere:
q4_0 KV → ~74 t/s (GPU idle 0-6%), while q8_0 KV → ~1000 t/s. Generation unaffected.

## Root cause (found by source reading + tested)
1. On Ampere, prefill selects the **MMA-f16** fattn kernel (`fattn.cu`). It needs f16
   K/V, so the **whole KV cache is dequantized to f16 on every attention call**
   (`fattn-common.cuh`, `ggml_get_to_fp16_cuda(...)`), which is O(n^2) over a prefill.
2. `ggml_get_to_fp16_cuda` (convert.cu) gives **q8_0** a dedicated half2 kernel
   `dequantize_block_q8_0_f16_cuda` (guarded by `fp16_available`), but **q4_0 fell back
   to the generic element-wise dequant** → the hot per-call KV->f16 conversion is far
   slower → prefill collapses.

## Fix
Dedicated half2-vectorized `q4_0 -> f16` kernel (same math as `dequantize_block_q4_0`,
il/ir tiling, half2 writes) + dispatch it under `fp16_available`, mirroring q8_0.
q4_1/q5_0/q5_1 kernels included too (same approach).

## Verified results (ALL quantized KV types, Qwen3.8-27B, 2x RTX 3090)
| KV type | BEFORE | AFTER (this patch) | correctness |
|---------|--------|--------------------|-------------|
| q4_0    | 74 t/s | **1063 t/s** | 17x23=391 |
| q4_1    | ~157   | **825 t/s**  | 391 |
| q5_0    | slow   | **830 t/s**  | 391 |
| q5_1    | 125    | **824 t/s**  | 391 |
| q8_0    | (already fast) | 1002 | - |

GPU util went from 0-6% (idle, stalled) to ~100%. q4_0/q8_0 are FA-enabled in a
default build; q4_1/q5_0/q5_1 additionally require `-DGGML_CUDA_FA_ALL_QUANTS=ON`
(they were `return false` in `ggml_cuda_fattn_kv_type_supported` otherwise). With that
flag + these dequant kernels, every quantized KV type prefills at full speed.

## Important note on q4_1/q5_0/q5_1
`ggml_cuda_fattn_kv_type_supported` (fattn.cu ~343) returns **false** for
q4_1/q5_0/q5_1 unless `GGML_CUDA_FA_ALL_QUANTS` is defined. So in a default build only
**q4_0, q8_0, f16, bf16** are FA-enabled KV types; q4_1/q5_0/q5_1 fall back to non-FA
(slow) regardless of the dequant kernel. The q4_1/q5_0/q5_1 kernels in this patch are
correct (12x12=144 verified) and take effect **only** when built with
`-DGGML_CUDA_FA_ALL_QUANTS=ON`. For the common case q4_0 is the smallest FA-enabled
quantized KV type, so this fix already covers the practically useful path: q4_0 KV now
matches q8_0 speed at half the KV VRAM (=> ~2x context for the same budget).

## Build (Windows, verified)
CUDA 13.3 + MSVC 2022 BuildTools + Ninja, `-DGGML_CUDA=ON
-DCMAKE_CUDA_ARCHITECTURES=86`. See `_configure.bat` / `_build.bat`. Add
`-DGGML_CUDA_FA_ALL_QUANTS=ON` to also enable/verify q4_1/q5_0/q5_1.
