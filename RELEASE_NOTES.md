# q3_K KV cache type (experimental)

Adds **`q3_K` as a KV cache type** for the CUDA backend — a 3.4375-bit KV cache option:

```bash
llama-server -m model.gguf --flash-attn on --cache-type-k q3_K --cache-type-v q3_K
```

This is the standalone experimental container for the q3_K work. It is intentionally kept **separate
from the `#27109` quantized-KV prefill fix** — that fix is its own change and lives upstream in PR
[ggml-org/llama.cpp#27140](https://github.com/ggml-org/llama.cpp/pull/27140). This release builds on
that fix and adds q3_K on top; the fix itself is not part of what this release contributes.

## What q3_K gives you

- **~24 % less KV-cache VRAM** than `q4_0` (3.44 bit vs 4.5 bit), ~78 % less than `f16` — smaller
  cards fit more context for the same VRAM.
- **Same prefill speed as `q4_0`** — identical at every context length (938 / 854 / 685 tok/s at
  4K / 16K / 64K on an RTX 3090; the KV-quant type does not affect prefill).
- **Perfect long-context recall** — 10/10 needles found at 256K context on Qwen 27B, matching `q4_0`.
- **Near-lossless quality** — Qwen 27B perplexity penalty ≈ 0.13 % vs `f16` (`q4_0` is lossless), and
  the penalty does not grow with context. Reuses the existing `q3_K` K-quant — no new ggml type.

> KV-quant sensitivity is mostly a small-model artifact — small models (e.g. 1.5B) can degrade badly
> under `q3_K`/`q4_0` KV, while a 27B tolerates it. Validate on your own model.

## Notes

- CUDA backend, tested on RTX 3090 (Ampere); Flash Attention required.
- `q3_K` KV needs `n_embd_k_gqa % 256 == 0` (K-quant super-block size).
- `llama-bench` also accepts `-ctk q3_K` / `-ctv q3_K`.
- Branch: `q3k-kv-experimental`. Built on top of the `#27109` prefill fix
  ([ggml-org/llama.cpp#27140](https://github.com/ggml-org/llama.cpp/pull/27140)), which is the
  separate upstream contribution — this container is the q3_K experiment only.
