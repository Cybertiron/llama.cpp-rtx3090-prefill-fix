# llama.cpp — RTX 3090 / Ampere KV-cache fork

A fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) focused on making the
**quantized KV cache fast and small on Ampere GPUs** (RTX 3090 / 3090 Ti and the rest of the
`sm_80`/`sm_86` family).

It carries two independent changes on top of upstream:

1. **Prefill fix for quantized KV cache + Flash Attention** (upstream issue
   [#27109](https://github.com/ggml-org/llama.cpp/issues/27109)) — restores full prompt-processing
   speed when a quantized KV cache is combined with Flash Attention.
2. **`q3_K` as a KV cache type** (new) — a 3.44-bit KV cache option that saves ~24 % VRAM versus
   `q4_0` while staying lossless on large models, at full prefill speed.

Everything else is stock upstream llama.cpp. The original project README is preserved as
[`README.upstream.md`](README.upstream.md) — use it for general build and usage instructions.

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

## 2. New KV cache type: `q3_K`

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

Perplexity on Qwen 27B (`--flash-attn on`, V = `f16` unless noted). Differences are within the
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

The `q3_K` KV type lives on the experimental `q3k-kv-experimental` branch (this release); the
standalone `#27109` prefill fix lives on `fix-27109-quant-kv-fp16` and is submitted upstream as
PR [#27140](https://github.com/ggml-org/llama.cpp/pull/27140). The `q3_K` KV type touches the CUDA
KV-cache paths (cache-type parsing, the SET_ROWS quantized write, the Flash-Attention support gate and
f16 conversion, and the non-contiguous q3_K→f16 dequant for the KV view), plus `llama-bench` so it
accepts `-ctk q3_K` / `-ctv q3_K`.

## License

MIT, same as upstream llama.cpp.

## Credits

Built on [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) by Georgi Gerganov and
contributors. Fork changes by [@Cybertiron](https://github.com/Cybertiron).
