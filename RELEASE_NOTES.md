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

All numbers here were measured on **Qwen3.6-27B** (Unsloth `Qwen3.6-27B-UD-Q4_K_XL.gguf`, a Qwen3-Next
hybrid, `context_length` 262144) on a single RTX 3090.

Bit-widths are exact (KV storage per element); VRAM is the KV footprint vs `f16`. Quality was
measured directly on that model for the **bold** rows; legacy rows follow from bit-width.

| KV type | Bits/elem | KV VRAM vs `f16` | Quality (Qwen3 27B) | Status |
| ------- | --------: | ---------------: | ------------------- | ------ |
| `f16`   | 16.0      | 100 %            | reference           | baseline |
| `q8_0`  | 8.5       | 53 %             | lossless            | safe on any model |
| `q5_1`  | 6.0       | 38 %             | lossless (large)    | |
| `q5_0`  | 5.5       | 34 %             | lossless (large)    | |
| `q4_1`  | 5.0       | 31 %             | lossless (large)    | |
| `q4_0`  | 4.5       | 28 %             | **lossless**        | common default |
| **`q3_K`** | 3.4375 | 21.5 %           | **≈ lossless (+~0.1 %)** | **recommended floor** |
| KVarN-3 † | 3.375   | 21.1 %           | ~2.5× lower KL-div than `q3_K` (beats `q4_0`) | external fork, not included |
| **`q2_K`** | 2.625  | 16.4 %           | **+2.4 % ppl**      | experimental, extreme VRAM |
| KVarN-2 † | 2.375   | 14.8 %           | behind `q3_K` (2-bit tier) | external fork, not included |

> † **KVarN** (variance-normalized KV quant from the separate
> [Anbeeld/beellama.cpp](https://github.com/Anbeeld/beellama.cpp) fork, **not** in this build). On the
> 27B, perplexity can't separate KV quants (all within noise), so they were ranked by KL-divergence vs
> `f16`: `kvarn3` 0.0016 < `q4_0` 0.0020 < `q3_K` 0.0039. The 3.375-bit `kvarn3` lands ~2.5× closer to
> `f16` than the 3.44-bit `q3_K` and even beats `q4_0`; the 2.375-bit `kvarn2` falls behind `q3_K` (bit
> gap too wide). Catch on the Qwen3-Next hybrid: KVarN's large recurrent-state cache forces two GPUs,
> defeating the single-24 GB-card goal, so this fork ships `q3_K`/`q2_K` instead.

## Tested across models

So the KV types are not tuned to a single model, they were exercised on more than one architecture
(one RTX 3090, Flash Attention on, plain batch = 1 decode — no speculative/MTP):

| Model | KV type | Context | Result |
| ----- | ------- | ------- | ------ |
| Qwen3.6-27B (`UD-Q4_K_XL`) | `q3_K`/`q3_K` | 262144 (256K, max) | needle **10/10**, depths 5–95 %, no crash |
| Qwen3.8-27B (`UD-Q4_K_XL`) | `q2_K`/`q2_K` | 262144 (256K, max) | needle **10/10**, depths 5–95 %, no crash |
| Gemma 4 E4B (`Q8_0`, head_dim 512) | `q3_K`/`q3_K` | 262144 (256K, YaRN 2×) | needle **10/10**, depths 5–95 % |
| Gemma 4 E4B (`Q8_0`, head_dim 512) | `q2_K`/`q2_K` | 262144 (256K, YaRN 2×) | needle **10/10**, depths 5–95 % |
| Qwen3.6-27B (`UD-Q4_K_XL`) | `kvarn3` ‡ | 262144 (256K, max) | needle **10/10** — reference, external fork |
| Qwen3.6-27B (`UD-Q4_K_XL`) | `kvarn2` ‡ | 262144 (256K, max) | needle **10/10** — reference, external fork |

Both K-quant KV types hold full 256K-context retrieval on two 27B models and on the much smaller
Gemma 4 E4B (sliding-window attention, stretched to 256K with YaRN) — even the aggressive `q2_K`. The
paths work across different head dimensions (Qwen key_length 256, Gemma 512).

‡ The `kvarn*` rows are **not** this build — run for reference on the external
[Anbeeld/beellama.cpp](https://github.com/Anbeeld/beellama.cpp) release (v0.4.3), the only build with
KVarN. On the Qwen3-Next hybrid KVarN needs **two RTX 3090s** (its recurrent-state cache does not fit
one 24 GB card) and keeps an intrinsic 128-token exact suffix; retrieval stays intact even at 2.375-bit
`kvarn2`, but at the cost of a second GPU — which is why this fork ships single-card `q3_K`/`q2_K`.

## Benchmarks (Qwen3.6-27B `UD-Q4_K_XL`, RTX 3090)

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
