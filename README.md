# llama.cpp — RTX 3090 / Ampere KV-cache fork

A fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) focused on making the
**quantized KV cache fast and small on Ampere GPUs** (RTX 3090 / 3090 Ti and the rest of the
`sm_80`/`sm_86` family).

It carries two independent changes on top of upstream:

1. **Prefill fix for quantized KV cache + Flash Attention** (upstream issue
   [#27109](https://github.com/ggml-org/llama.cpp/issues/27109)) — restores full prompt-processing
   speed when a quantized KV cache is combined with Flash Attention.
2. **New K-quant KV cache types** (new):
   - **`q3_K`** (3.44-bit) — saves ~24 % VRAM versus `q4_0`, essentially lossless on large models,
     at full prefill speed. **Recommended** as the practical lower bound.
   - **`q2_K`** (2.6-bit, *experimental*) — squeezes another ~24 % below `q3_K` for the tightest KV
     budgets, but at a real, measurable quality cost. For extreme-VRAM cases only.

Everything else is stock upstream llama.cpp. The original project README is preserved as
[`README.upstream.md`](README.upstream.md) — use it for general build and usage instructions.

> **2026-08-20 update.** Fixed a Flash-Attention kernel-routing bug that aborted `q3_K`/`q2_K` on a
> plain single-token (batch = 1) decode on Ampere — they had no VEC-kernel instance and hit a fatal
> abort. They now fall through to the TILE/MMA f16 path for every batch size. Prefill and
> speculative/MTP decode were unaffected. Re-download the binary if you grabbed an earlier build.

---

## 1. Prefill fix: quantized KV cache + Flash Attention on Ampere

**The problem.** On Ampere GPUs, running a quantized KV cache (`q4_0`, `q4_1`, `q5_0`, `q5_1`)
together with Flash Attention (`--flash-attn on`) collapsed prompt processing (prefill) to a tiny
fraction of the expected throughput — on the order of **~74 tok/s instead of ~1000+ tok/s** — because
the quantized→f16 conversion feeding the Flash-Attention kernels took a slow path. The usual
workaround was to keep the KV cache at `q8_0`, which is fast but uses more VRAM.

**The fix.** Dedicated dequantization handling so a quantized KV cache runs the fast Flash-Attention
path. Quantized KV + FA now prefills at full GPU speed on Ampere.

Submitted upstream as PR
[#27140](https://github.com/ggml-org/llama.cpp/pull/27140).

```bash
# Now fast on RTX 3090 instead of ~74 tok/s:
llama-server -m model.gguf --flash-attn on \
             --cache-type-k q4_0 --cache-type-v q4_0
```

---

## 2. New KV cache types: `q3_K` and `q2_K`

Adds `q3_K` (a K-quant super-block format, **3.4375 bits/weight**) as a selectable KV cache type:

```bash
llama-server -m model.gguf --flash-attn on \
             --cache-type-k q3_K --cache-type-v q3_K
```

### Why `q3_K` and not a legacy `q3_0`/`q3_1`

There is no legacy 3-bit block type in llama.cpp (legacy KV types stop at `q4_0`). `q3_K` already
exists as a weight-quantization format with a full CPU + CUDA implementation, so this change only had
to wire it into the KV-cache paths rather than define a brand-new ggml type. As a bonus, `q3_K`
(3.4375 bit) is *smaller* than a hypothetical 32-block `q3_0` (3.5 bit) and higher quality, thanks to
its per-sub-block 6-bit scales. (The "K" in `q3_K` means *K-quant*, not *Keys* — the collision with
the K cache is a coincidence.)

### Quality — lossless on large models

> **Test model.** All quality, speed, VRAM and needle numbers in this document were measured on
> **Qwen3.6-27B** (Unsloth `Qwen3.6-27B-UD-Q4_K_XL.gguf`, a Qwen3-Next hybrid, `context_length` 262144)
> on a single RTX 3090. The `q2_K` perplexity figures below use the same model. Small models are far
> more KV-quant sensitive — see the note under the table.

Perplexity on Qwen3.6-27B (`--flash-attn on`, V = `f16` unless noted). Differences are within the
measurement noise (±0.003):

| KV cache (K / V) | Bits (K) | Perplexity |
| ---------------- | -------- | ---------- |
| `f16` / `f16`    | 16       | 1.0142     |
| `q4_0` / `q4_0`  | 4.5      | 1.0142     |
| `q3_K` / `f16`   | 3.44     | 1.0115     |
| `q3_K` / `q3_K`  | 3.44     | 1.0124     |

> **Note.** KV-quant sensitivity is largely a *small-model* artifact. On Qwen2.5-1.5B a `q4_0` K
> cache blows perplexity up catastrophically (only `q8_0` is safe), but a 27B model tolerates
> aggressive KV quantization with no measurable quality loss. Validate on your own model before
> using `q3_K` on small models.

### Speed — no prefill penalty

Prefill throughput (`llama-bench -p 4096 -n 0`, Qwen 27B, RTX 3090, `--flash-attn on`). `q3_K` matches
`f16`/`q4_0` — the conversion path adds no measurable overhead:

| KV cache (K / V) | Prefill (tok/s) |
| ---------------- | --------------- |
| `f16` / `f16`    | 949.9           |
| `q8_0` / `f16`   | 944.9           |
| `q4_0` / `f16`   | 936.7           |
| `q3_K` / `f16`   | 945.4           |
| `q3_K` / `q3_K`  | 937.6           |

### VRAM

`q3_K` stores the KV cache at 3.44 bit vs `q4_0` at 4.5 bit — about **24 % less KV-cache VRAM**, which
matters most at long context lengths.

### `q2_K` — experimental, extreme VRAM only

`q2_K` (**2.625 bits/weight**) is wired in through the same machinery, for when you need the last bit
of KV VRAM:

```bash
llama-server -m model.gguf --flash-attn on \
             --cache-type-k q2_K --cache-type-v q2_K
```

It is **more experimental than `q3_K`** and carries a *real* quality cost — unlike `q3_K`, the drop is
measurable, not within noise. On Qwen 27B (16 K perplexity, wikitext): `f16` 5.799, `q3_K` 5.791
(≈ `f16`), `q2_K` 5.937 (**+2.4 %**, ~4σ). It needs an internal f16-range clamp on the super-block
scale to stay finite (`q2_K`'s `d = max_scale/15` has less headroom than `q3_K`'s `/32`, so large KV
values would otherwise overflow the f16 store and produce NaNs). Reach for `q2_K` only when `q3_K` will
not fit and you can accept the degradation; otherwise treat **`q3_K` as the recommended floor**.

### KV cache type comparison

All bit-widths are exact (KV-cache storage per element). VRAM is the KV-cache footprint relative to
`f16`. Quality was measured directly on Qwen3 27B for the **bold** rows; the legacy rows follow from
bit-width (higher bits than `q4_0`, which measured lossless on this model).

| KV type | Bits/elem | KV VRAM vs `f16` | Quality (Qwen3 27B) | Status |
| ------- | --------: | ---------------: | ------------------- | ------ |
| `f16`   | 16.0      | 100 %            | reference           | baseline |
| `q8_0`  | 8.5       | 53 %             | lossless            | safe on any model, incl. small |
| `q5_1`  | 6.0       | 38 %             | lossless (large)    | |
| `q5_0`  | 5.5       | 34 %             | lossless (large)    | |
| `q4_1`  | 5.0       | 31 %             | lossless (large)    | |
| `q4_0`  | 4.5       | 28 %             | **lossless**        | common default |
| **`q3_K`** | 3.4375 | 21.5 %           | **≈ lossless (+~0.1 %)** | **recommended floor** |
| KVarN-3 †  | 3.375  | 21.1 %           | ~2.5× lower KL-div than `q3_K` (beats `q4_0`) | external fork; not included |
| **`q2_K`** | 2.625  | 16.4 %           | **+2.4 % ppl**      | experimental, extreme VRAM |
| KVarN-2 †  | 2.375  | 14.8 %           | behind `q3_K` (2-bit tier) | external fork; not included |

> † **KVarN** is a variance-normalized KV quantization from the separate
> [Anbeeld/beellama.cpp](https://github.com/Anbeeld/beellama.cpp) fork — **not** part of this build.
> On the 27B, perplexity cannot separate KV quants at all (everything is within noise), so these were
> ranked by **KL-divergence** against the `f16` model: `kvarn3` 0.0016 < `q4_0` 0.0020 < `q3_K` 0.0039.
> So the 3.375-bit `kvarn3` lands ~2.5× closer to `f16` than the 3.44-bit `q3_K`, and even beats `q4_0` —
> the more interesting quality-per-bit direction. The 2.375-bit `kvarn2`, however, falls behind `q3_K`
> (the bit gap is too wide to make up), so `q3_K` still wins that tier. The catch for this fork's use
> case: on the Qwen3-Next hybrid KVarN's large recurrent-state cache forces the model across **two GPUs**,
> which defeats the "fit a big context on one 24 GB card" goal — hence `q3_K`/`q2_K` here instead.

### Tested across models

To confirm the KV types are not tuned to a single model, they were exercised on more than one
architecture (all on one RTX 3090, Flash Attention on, plain batch = 1 decode — no speculative/MTP):

| Model | KV type | Context | Result |
| ----- | ------- | ------- | ------ |
| **Qwen3.6-27B** (`UD-Q4_K_XL`, Qwen3-Next) | `q3_K`/`q3_K` | 262144 (256K, max) | needle-in-haystack **10/10** across depths 5–95 %, no crash |
| **Qwen3.8-27B** (`UD-Q4_K_XL`, Qwen3-Next) | `q2_K`/`q2_K` | 262144 (256K, max) | needle-in-haystack **10/10** across depths 5–95 %, no crash |
| **Gemma-3n E4B** (`Q8_0`, head_dim 512) | `q3_K`/`q3_K` | 262144 (256K, YaRN 2× over 128K native) | needle-in-haystack **10/10** across depths 5–95 % |
| **Gemma-3n E4B** (`Q8_0`, head_dim 512) | `q2_K`/`q2_K` | 262144 (256K, YaRN 2×) | needle-in-haystack **10/10** across depths 5–95 % |

So both K-quant KV types hold full 256K-context retrieval on two different 27B models *and* on the
much smaller Gemma-3n E4B (sliding-window attention, head_dim 512, stretched to 256K with YaRN) — even
the aggressive `q2_K`. The code paths work across different head dimensions (Qwen key_length 256,
Gemma 512). Small models are more KV-quant sensitive in general free-form generation (Gemma's `q2_K`
output reads a touch looser than `f16`), but that did not cost any needle retrieval here.

### Requirements & limitations

- CUDA build with Flash Attention. Tested on RTX 3090 (Ampere).
- K-quant uses a 256-element super-block, so it requires `n_embd_k_gqa % 256 == 0`
  (satisfied by e.g. Qwen 27B: 4 KV heads × 256 = 1024).
- Implemented for the CUDA backend.

---

## Building

Standard llama.cpp CUDA build — see [`README.upstream.md`](README.upstream.md). In short:

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

The `q3_K` / `q2_K` KV types live on the experimental `q3k-kv-experimental` branch (this release); the
standalone `#27109` prefill fix lives on `fix-27109-quant-kv-fp16` and is submitted upstream as
PR [#27140](https://github.com/ggml-org/llama.cpp/pull/27140). Each K-quant KV type touches the CUDA
KV-cache paths (cache-type parsing, the SET_ROWS quantized write, the Flash-Attention support gate and
f16 conversion, and the non-contiguous K-quant→f16 dequant for the KV view), plus `llama-bench` so it
accepts `-ctk q3_K`/`q2_K` and `-ctv q3_K`/`q2_K`. `q3_K` alone is also submitted upstream as a
standalone PR [#27362](https://github.com/ggml-org/llama.cpp/pull/27362).

## License

MIT, same as upstream llama.cpp.

## Credits

Built on [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) by Georgi Gerganov and
contributors. Fork changes by [@Cybertiron](https://github.com/Cybertiron).
