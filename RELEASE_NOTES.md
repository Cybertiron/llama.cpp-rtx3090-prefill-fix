# q3_K KV cache type (experimental)

> **Update 2026-08-20:** fixed a flash-attention kernel-routing bug that aborted
> `q3_K`/`q2_K` on a plain single-token (batch=1) decode on Ampere — they had no
> VEC-kernel instance and hit `GGML_ABORT` at `fattn.cu`. They now fall through to
> the TILE/MMA f16 path for every batch size. Re-download the binary below if you
> grabbed an earlier build. (Prefill and speculative/MTP decode were unaffected.)

Adds `q3_K` (3.4375 bits/elem) as a KV cache type on the CUDA backend, for
squeezing more context out of limited VRAM:

    llama-server --flash-attn on --cache-type-k q3_K --cache-type-v q3_K

- ~24% smaller KV cache than `q4_0`, ~78% smaller than `f16` — smaller cards fit
  more context for the same VRAM.
- Same prefill speed as `q4_0` (pp4096 on RTX 3090: q3_K 967 t/s vs q4_0 966 t/s).
- Reuses the existing `q3_K` K-quant machinery — no new ggml type.
- Requires `n_embd_k_gqa % 256 == 0` (K-quant super-block size).

## Download (if you don't want to compile)

`llama-q3k-win-cuda13-ampere.zip` below is a standalone Windows / CUDA 13 build for
RTX 3090 (Ampere, arch 86). Unzip and run `llama-server.exe` with the flags above —
no compiling, CUDA runtime DLLs are included.

## KV cache type comparison

Bit-widths are exact (KV storage per element); VRAM is the KV footprint vs `f16`. Quality was
measured directly on Qwen3 27B for the **bold** rows; legacy rows follow from bit-width.

| KV type | Bits/elem | KV VRAM vs `f16` | Quality (Qwen3 27B) | Status |
| ------- | --------: | ---------------: | ------------------- | ------ |
| `f16`   | 16.0      | 100 %            | reference           | baseline |
| `q8_0`  | 8.5       | 53 %             | lossless            | safe on any model |
| `q5_1`  | 6.0       | 38 %             | lossless (large)    | |
| `q5_0`  | 5.5       | 34 %             | lossless (large)    | |
| `q4_1`  | 5.0       | 31 %             | lossless (large)    | |
| `q4_0`  | 4.5       | 28 %             | **lossless**        | common default |
| **`q3_K`** | 3.4375 | 21.5 %           | **≈ lossless (+~0.1 %)** | **recommended floor** |
| **`q2_K`** | 2.625  | 16.4 %           | **+2.4 % ppl**      | experimental, extreme VRAM |
| KVarN-3 † | ~3.0    | ~19 %            | ~2.5× better KLD/bit than `q3_K` | external fork, not included |

> † **KVarN** (variance-aware KV quant from the separate
> [Anbeeld/beellama.cpp](https://github.com/Anbeeld/beellama.cpp) fork, **not** in this build) landed
> ~2.5× closer to `f16` per bit than `q3_K` in a KL-divergence test — the more interesting
> quality-per-bit direction. Its catch on the Qwen3-Next hybrid: the large recurrent-state cache
> forces two GPUs, defeating the single-24 GB-card goal, so this fork ships `q3_K`/`q2_K` instead.

## Benchmarks (Qwen3 27B, RTX 3090)

**Prefill parity with `q4_0`** — the KV-quant type has no measurable effect on prefill
(llama-bench, same build, only the KV type differs). The slowdown with context is the usual
O(n²) attention, identical for every KV type:

| ctx | q3_K | q4_0 |
| --- | ---: | ---: |
| 4096 | 938 t/s | 937 t/s |
| 16384 | 854 | 855 |
| 65536 | 685 | 686 |

**Long-context retrieval** — needle-in-haystack with 10 secrets placed at depths 5%–95% over a
259 K-token context: **10/10 recalled, identical to `q4_0`**.

**Perplexity** (wikitext, 16K ctx): `f16`, `q4_0`, and `q3_K` all land within the measurement
noise (±0.037) on the 27B — no measurable quality loss from the q3_K KV cache. On a small model
(e.g. 1.5B) a `q3_K`/`q4_0` KV cache degrades badly, so validate on your own model; large models
are robust.

## Implementation

cache-type parsing (arg.cpp), SET_ROWS quantized write (cpy-utils / set-rows /
supports_op), flash-attn support gate + f16 conversion (fattn.cu), and a
non-contiguous q3_K->f16 dequant for the strided KV view (convert.cu).
`llama-bench` accepts `-ctk`/`-ctv q3_K`.

## Acknowledgements

The `q3_K` quantization format itself is [@ikawrakow](https://github.com/ikawrakow)'s
k-quant work. This release only wires that existing quant into the KV-cache path — it
does not add a new quantization scheme.

## Also: q2_K (experimental)

The same build also wires `q2_K` (~2.6 bpw) in as a KV cache type
(`--cache-type-k q2_K --cache-type-v q2_K`), for extreme VRAM squeezing — ~24 % smaller
than `q3_K`. It needs a super-block-scale clamp to stay finite (q2_K's `d = max_scale/15`
overflows f16 on large KV values otherwise). It generates correctly, but carries a real,
measurable quality cost — Qwen3 27B 16K perplexity is ~+2.4 % vs `f16` (about 4σ, unlike
`q3_K` which is within noise). Treat `q3_K` as the recommended lower bound for quality;
reach for `q2_K` only when you need the last bit of KV VRAM and can accept the degradation.

## Notes

- CUDA backend, tested on RTX 3090 (Ampere); Flash Attention required.
- Submitted upstream as a standalone PR: [ggml-org/llama.cpp#27362](https://github.com/ggml-org/llama.cpp/pull/27362).
- The prebuilt binary above is also based on the `#27109` quantized-KV prefill fix
  ([ggml-org/llama.cpp#27140](https://github.com/ggml-org/llama.cpp/pull/27140)) —
  that fix is a separate upstream contribution; q3_K itself does not depend on it.
