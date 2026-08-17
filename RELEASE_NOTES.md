# v0.1 — Ampere KV cache: fast prefill + `q3_K` KV type

A fork of llama.cpp with two Ampere / RTX 3090 KV-cache changes on top of upstream.

## Highlights

### 1. Prefill fix — quantized KV cache + Flash Attention (issue #27109)
On Ampere GPUs, a quantized KV cache (`q4_0`/`q4_1`/`q5_0`/`q5_1`) combined with `--flash-attn on`
collapsed prompt processing to ~74 tok/s instead of ~1000+ tok/s. This restores full prefill speed
for quantized KV + Flash Attention. Submitted upstream as PR #27140.

### 2. New KV cache type: `q3_K`
A 3.4375-bit KV cache option: `--cache-type-k q3_K --cache-type-v q3_K`.

- **~24 % less KV-cache VRAM** than `q4_0` (3.44 bit vs 4.5 bit).
- **Lossless on large models** — Qwen 27B perplexity: `q3_K/q3_K` = 1.0124 ≈ `q4_0/q4_0` = 1.0142
  ≈ `f16` = 1.0142 (within ±0.003).
- **No prefill penalty** — ~940 tok/s (pp4096, RTX 3090), same as `f16`/`q4_0`.
- Reuses the existing `q3_K` K-quant (per-sub-block 6-bit scales); no new ggml type. Smaller and
  higher quality than a hypothetical legacy `q3_0` would be.

> KV-quant sensitivity is mostly a small-model artifact — small models (e.g. 1.5B) can degrade badly
> under `q3_K`/`q4_0` KV, while 27B tolerates it losslessly. Validate on your own model.

## Usage

```bash
# Fast quantized KV + FA on Ampere:
llama-server -m model.gguf --flash-attn on --cache-type-k q4_0 --cache-type-v q4_0

# New q3_K KV type (smallest, lossless on large models):
llama-server -m model.gguf --flash-attn on --cache-type-k q3_K --cache-type-v q3_K
```

## Notes
- CUDA backend, tested on RTX 3090 (Ampere), Flash Attention required.
- `q3_K` KV needs `n_embd_k_gqa % 256 == 0` (K-quant super-block size).
- `llama-bench` also accepts `-ctk q3_K` / `-ctv q3_K`.

Built on [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). See the branch
`fix-27109-quant-kv-fp16` for the changes.
