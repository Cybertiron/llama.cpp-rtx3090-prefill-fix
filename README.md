# llama.cpp — RTX 3090 / Ampere KV-cache + speculative-decoding fork

A fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) focused on fitting **large
contexts on a single RTX 3090** (and the rest of the `sm_80`/`sm_86` Ampere family) by making the
quantized KV cache both **fast** and **small**, and on running **DFlash2 speculative decoding** on top.

It carries four changes on top of upstream:

1. **Prefill fix for quantized KV cache + Flash Attention** (upstream issue
   [#27109](https://github.com/ggml-org/llama.cpp/issues/27109)) — restores full prompt-processing
   speed when a quantized KV cache is combined with Flash Attention on Ampere.
2. **K-quant KV cache types** — `q3_K` (3.44-bit) and `q2_K` (2.6-bit) as selectable KV types.
3. **`kvarn2` / `kvarn3`** — 2-bit / 3-bit KV types with a built-in Walsh–Hadamard rotation
   (variance-normalization), single-GPU, and compatible with speculative decoding.
4. **DFlash2 speculative decoding** — the upstream DFlash2 block-denoising drafter, cherry-picked in.

The original project README is preserved as [`README.upstream.md`](README.upstream.md) for general
build and usage instructions.

> **Downloads.** Prebuilt Windows / CUDA 13 / Ampere binaries are on the
> [Releases](../../releases) page. The current **[KVarN + DFlash2 release](../../releases/latest)**
> bundles all four features in one `llama-server`.

---

## 1. Prefill fix: quantized KV cache + Flash Attention on Ampere

**The problem.** On Ampere GPUs, a quantized KV cache (`q4_0`, `q4_1`, `q5_0`, `q5_1`) together with
Flash Attention (`--flash-attn on`) collapsed prefill to a fraction of expected throughput — on the
order of **~74 tok/s instead of ~1000+ tok/s** — because the quantized→f16 conversion feeding the
Flash-Attention kernels took a slow path. The usual workaround was to keep the KV cache at `q8_0`.

**The fix.** Dedicated dequantization handling so a quantized KV cache runs the fast Flash-Attention
path. Quantized KV + FA now prefills at full GPU speed on Ampere. Submitted upstream as PR
[#27140](https://github.com/ggml-org/llama.cpp/pull/27140).

```bash
# Now fast on RTX 3090 instead of ~74 tok/s:
llama-server -m model.gguf --flash-attn on --cache-type-k q4_0 --cache-type-v q4_0
```

---

## 2. K-quant KV cache types: `q3_K` and `q2_K`

Adds `q3_K` (a K-quant super-block format, **3.4375 bits/weight**) and `q2_K` (**2.625 bits/weight**)
as selectable KV cache types:

```bash
llama-server -m model.gguf --flash-attn on --cache-type-k q3_K --cache-type-v q3_K
```

`q3_K` saves about **24 % KV-cache VRAM** versus `q4_0` and is essentially lossless on large models,
at full prefill speed. `q2_K` squeezes another ~24 % below `q3_K` for the tightest KV budgets, at a
real, measurable quality cost — use it only when `q3_K` will not fit.

### Quality — lossless on large models

> **Test models.** Numbers here were measured on **Qwen3.6-27B** / **Qwen3.8-27B** (Unsloth
> `UD-Q4_K_XL`, Qwen3-Next hybrids, `context_length` 262144) and **Gemma 4 E4B** on a single RTX 3090.

Perplexity on Qwen3 27B (`--flash-attn on`, V = `f16` unless noted); differences within noise (±0.003):

| KV cache (K / V) | Bits (K) | Perplexity |
| ---------------- | -------- | ---------- |
| `f16` / `f16`    | 16       | 1.0142     |
| `q4_0` / `q4_0`  | 4.5      | 1.0142     |
| `q3_K` / `f16`   | 3.44     | 1.0115     |
| `q3_K` / `q3_K`  | 3.44     | 1.0124     |

`q2_K` carries a real cost: on Qwen 27B (16 K wikitext perplexity) `f16` 5.799, `q3_K` 5.791 (≈ `f16`),
`q2_K` 5.937 (**+2.4 %**). Treat **`q3_K` as the recommended floor**; `q2_K` for extreme-VRAM only.

> **Note.** KV-quant sensitivity is largely a *small-model* artifact. A 27B model tolerates aggressive
> KV quantization with no measurable loss; a 1.5B model does not. Validate on your own small models.

### Speed — no prefill penalty

Prefill throughput (`llama-bench -p 4096 -n 0`, Qwen 27B, RTX 3090, `--flash-attn on`):

| KV cache (K / V) | Prefill (tok/s) |
| ---------------- | --------------- |
| `f16` / `f16`    | 949.9           |
| `q4_0` / `f16`   | 936.7           |
| `q3_K` / `f16`   | 945.4           |
| `q3_K` / `q3_K`  | 937.6           |

---

## 3. `kvarn2` / `kvarn3` — low-bit KV, single GPU

`kvarn2` and `kvarn3` are 2-bit (`Q2_0S`) and 3-bit (`Q3_0`) KV cache types:

```bash
llama-server -m model.gguf --flash-attn on --cache-type-k kvarn3 --cache-type-v kvarn3
```

Like **every** quantized KV type in this fork, they automatically get a **Walsh–Hadamard rotation**
of K and V (`attn_rot_k`/`attn_rot_v`, auto-enabled for quantized KV when `head_dim % 64 == 0`). The
rotation spreads quantization error evenly across the head dimension — the variance-normalization idea
behind KVarN — and it is applied inside the attention graph (rotate q + k, store rotated, un-rotate
the output), so the KV cache **stays quantized in VRAM** the whole time.

This is a *pragmatic* KVarN: `kvarn2`/`kvarn3` are the low-bit storage formats plus that built-in
rotation. They do **not** implement beellama's full KVarN (per-tile Sinkhorn scales + 128-token exact
tail). In exchange they run on a **single 24 GB card** — no recurrent-state cache multiplication — and
they work with DFlash2 speculative decoding (see §4).

### VRAM (Qwen3.8-27B, `-c 32768 --parallel 1 --flash-attn on`, single RTX 3090)

| KV type | VRAM | vs `f16` |
| ------- | ---- | -------- |
| `f16`   | 18756 MiB | — |
| `q4_0`  | 17370 MiB | −1386 |
| `q2_K`  | 17130 MiB | −1626 |
| **`kvarn2`** | **17114 MiB** | **−1642** |
| **`kvarn3`** | 17242 MiB | −1514 |

Within ~86 MiB of the [beellama.cpp](https://github.com/Anbeeld/beellama.cpp) build on the same model
(a fixed base-build offset, not KV-cache size), with identical greedy output.

---

## 4. DFlash2 speculative decoding

The upstream **DFlash2** draft-model implementation (block-denoising drafter with a top-k selector)
is cherry-picked in, so this build can run DFlash2 drafters:

```bash
llama-server -m target.gguf --flash-attn on -ngl 99 \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  --model-draft dflash2-drafter.gguf --gpu-layers-draft 999
```

(DFlash2 `block_size = 8` → `--spec-draft-n-max 7`.) Draft acceptance is **identical** to the upstream
reference build — 0.390 (57/146) on Qwen3.8-27B with an `f16` KV cache.

### DFlash2 works together with the low-bit KV types

Because the Hadamard rotation is shared between normal decode and DFlash's KV injection, the low-bit KV
types keep speculative decoding working — you get small KV **and** drafting at once:

| KV type + DFlash2 (Qwen3.8-27B) | draft acceptance |
| ------------------------------- | ---------------- |
| `q8_0`  | 0.390 |
| `q2_K`  | 0.319 |
| **`kvarn2`** | **0.331** |
| **`kvarn3`** | **0.350** |
| beellama `kvarn2` (reference) | 0.315 |

### Choosing a KV type: context vs VRAM vs speed

The whole point of a low-bit KV cache is the three-way trade-off between **max context**, **min VRAM**,
and **max throughput** on a single 24 GB card. Higher-bit types (`q8_0`) decode fastest but run out of
VRAM soonest; lower-bit types (`kvarn2`, `q2_K`) reach far larger contexts for a small speed cost.

All measurements below: single RTX 3090, `--flash-attn on`, `--split-mode none`, `--parallel 1`, with
the DFlash drafter loaded. **VRAM** is at a filled 16 K-token context; **decode t/s** is on a code prompt
(spec-decode shines on structured output — see the task note below); **max context** is the largest
`--ctx-size` that still loads on 24 GB.

**Qwen3.6-27B + DFlash** (`--spec-draft-n-max 15`):

| KV cache | Bits | VRAM @16K | Decode t/s | Max context (24 GB) |
| -------- | ---: | --------: | ---------: | ------------------- |
| `q8_0`       | 8.5    | 22.6 GB | **124.9** | ~34K |
| `q4_0`       | 4.5    | 22.3 GB | 121.7 | ~60K |
| `q3_K`       | 3.44   | 22.2 GB | 117.2 | ~68K |
| **`kvarn3`** | 3.5    | 22.2 GB | 117.0 | ~68K |
| `q2_K`       | 2.625  | 22.2 GB | 109.8 | ~85K |
| **`kvarn2`** | 2.5    | **22.2 GB** | 114.9 | **~95K** |

**Qwen3.8-27B + DFlash2** (`--spec-draft-n-max 7`; same hybrid arch, but noticeably lighter than 3.6 —
it fits much larger contexts):

| KV cache | Bits | VRAM @16K | Decode t/s | Max context (24 GB) |
| -------- | ---: | --------: | ---------: | ------------------- |
| `q8_0`       | 8.5    | 21.1 GB | 53.4 | ~100K |
| `q4_0`       | 4.5    | 20.8 GB | 52.3 | ~150K |
| `q3_K`       | 3.44   | 20.7 GB | 50.4 | ~200K |
| **`kvarn3`** | 3.5    | 20.7 GB | 50.2 | ~200K |
| `q2_K`       | 2.625  | 20.6 GB | 52.2 | ~220K |
| **`kvarn2`** | 2.5    | **20.6 GB** | **55.1** | **~240K** (near the 262K model limit) |

**How to read it.** Throughput barely moves across KV types (all within ~10 %) — decode speed is
dominated by draft acceptance, not the KV quant. What changes a lot is **max context**: the low-bit
types reach **~2.5–3× the context of `q8_0`** before OOM, because their KV cache is ~3× smaller. So pick
the *highest*-bit type whose max context covers your workload — `q8_0` for shorter runs, down to
`kvarn2`/`q2_K` when you want to push toward the model's full 262K; `kvarn3`/`q3_K` are the balanced
middle. Absolute numbers are very model-dependent: Qwen3.6 caps `q8_0` at ~34K while the lighter Qwen3.8
reaches ~100K — but within each model the *ordering* by KV bits is the same. (On these Qwen3-Next hybrids
the VRAM spread at 16 K is small — most layers are linear-attention, so only a few carry a quantizable KV
cache — but the gap compounds with context and decides the OOM point.)

> **Throughput is task-dependent.** The code-prompt numbers above are a *best case*: structured output
> is highly predictable, so the DFlash drafter is accepted often (mean ~8–9 tokens per step). On
> free-form prose the *same* setup drops to **~25–35 tok/s** for every KV type — acceptance, not the KV
> quant, dominates.

---

## Tested across models

KV types exercised on more than one architecture (single RTX 3090, Flash Attention on):

| Model | KV type | Context | Result |
| ----- | ------- | ------- | ------ |
| **Qwen3.6-27B** (`UD-Q4_K_XL`, Qwen3-Next) | `q3_K` | 262144 (256K, max) | needle **10/10**, depths 5–95 % |
| **Qwen3.8-27B** (`UD-Q4_K_XL`, Qwen3-Next) | `q2_K` | 262144 (256K, max) | needle **10/10**, depths 5–95 % |
| **Gemma 4 E4B** (`Q8_0`, head_dim 512) | `q3_K` | 262144 (256K, YaRN 2×) | needle **10/10**, depths 5–95 % |
| **Qwen3.8-27B** | `kvarn2` / `kvarn3` | 32768 | coherent generation; DFlash2 acceptance 0.33 / 0.35 |
| **Gemma 4 E4B** (head_dim 512) | `kvarn2` | — | coherent generation |

Head dims 128 / 256 / 512 are covered (Qwen key_length 256, Gemma 512).

---

## Building

Standard llama.cpp CUDA build — see [`README.upstream.md`](README.upstream.md):

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

### Branches

| Branch | Contents |
| ------ | -------- |
| `q3k-kv-experimental` (default) | prefill fix + `q3_K` / `q2_K` KV types |
| `kvarn-cuda` | the above + `kvarn2` / `kvarn3` |
| `kvarn-dflash2` | the above + DFlash2 (this is what the latest release binary is built from) |
| `fix-27109-quant-kv-fp16` | the standalone `#27109` prefill fix (upstream PR [#27140](https://github.com/ggml-org/llama.cpp/pull/27140)) |

`q3_K` alone is also upstreamed as a standalone PR
[#27362](https://github.com/ggml-org/llama.cpp/pull/27362).

Requirements: CUDA build with Flash Attention, RTX 3090 / Ampere. K-quant uses a 256-element
super-block (`n_embd_k_gqa % 256 == 0`).

---

## License

MIT, same as upstream llama.cpp.

## Credits

- Built on [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) by Georgi Gerganov and contributors.
- **KVarN** variance-normalized KV quantization concept & name: [@Anbeeld](https://github.com/Anbeeld/beellama.cpp) (beellama.cpp), based on Huawei research.
- **Walsh–Hadamard KV rotation** (`attn_rot`): upstream llama.cpp PR [#21038](https://github.com/ggml-org/llama.cpp/pull/21038).
- **DFlash2** speculative decoding: @SubSir (upstream DFlash2 PR), cherry-picked unchanged.
- Fork changes by [@Cybertiron](https://github.com/Cybertiron).
