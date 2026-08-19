# q3_K KV cache type (experimental)

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

## Notes

- CUDA backend, tested on RTX 3090 (Ampere); Flash Attention required.
- Submitted upstream as a standalone PR: [ggml-org/llama.cpp#27362](https://github.com/ggml-org/llama.cpp/pull/27362).
- The prebuilt binary above is also based on the `#27109` quantized-KV prefill fix
  ([ggml-org/llama.cpp#27140](https://github.com/ggml-org/llama.cpp/pull/27140)) —
  that fix is a separate upstream contribution; q3_K itself does not depend on it.
