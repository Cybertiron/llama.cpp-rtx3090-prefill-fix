# KVarN CUDA port → llama.cpp-src (branch `kvarn-cuda`)

Goal: incorporate **CUDA-only** KVarN from Anbeeld/beellama.cpp into this fork.
Vulkan / HIP / tests deferred. Source reference: `C:\AI\beellama.cpp-src` (v0.4.3).
**Credit required** (per user rule): KVarN = @Anbeeld (beellama.cpp), variance-normalization
based on Huawei research. Must be attributed in code + any release.

## Architecture (mapped 2026-08-20)
KVarN is NOT a ggml_type. It is a KV-cache **mode** (`cparams.kvarn`, `LLAMA_KVARN_TYPE_*`):
- storage falls back to legacy quant types: `kvarn2`→`Q2_0S`, `kvarn3`→`Q3_0`, `kvarn4`→`Q4_0`,
  `kvarn5`→`Q5_0`, `kvarn6`→`Q6_0`, `kvarn8`→`Q8_0` (only Q2_0S/Q3_0/Q6_0/Q2_1/Q3_1/Q6_1 are new;
  Q4_0/Q5_0/Q8_0 already exist). 6 new GGML_TYPE enums.
- new ggml ops: `GGML_OP_KVARN_WHT` (Walsh-Hadamard rotation = variance normalization),
  `GGML_OP_KVARN_STORE`.
- FA integration via `GGML_FLASH_ATTN_EXT_OP_PARAM_KVARN_DOMAIN` (rotated / original / mixed).
- separate KV-cache class: `src/llama-kv-cache-kvarn.{cpp,h}` + config `src/llama-kvarn.{cpp,h}`.
- per-tile asymmetric quant (scales + zero-points, K and V), 128-token exact suffix
  (`KVAR_N_GROUP = 128`), ISWA policy for sliding-window models.
- CUDA: `kvarn.cu`, `kvarn-wht.cu`, `fattn-kvarn-*.cuh`, `fattn-mma-kvarn-*.cuh` +
  53 template instances (`fattn-mma-kvarn-decode-instance-k{2..8}-v{2..8}.cu`).

Scope: ~19.4k lines core (excl. template instances), ~130 files touched.

## Fork base divergence
This fork base = upstream `27df919`; beellama base is different (49 GGML types vs our 43).
Shared files (fattn.cu, ggml-cuda.cu, ggml.c, ggml-common.h, arg.cpp, common.cpp,
llama-kv-cache.cpp, ggml-cpu/ops.cpp) must be re-merged by hand, not cherry-picked.
Our fork already carries q3_K/q2_K KV + #27109 fix on these files — must be preserved.

## Phased roadmap (each phase must build before moving on)
- [x] **P1 — fallback quant types. DONE 2026-08-20.** Added GGML_TYPE_Q3_0 (43) + Q2_0S (44),
      COUNT=45. Files: ggml.h enum, ggml-common.h (QK/QR + block structs), ggml.c traits,
      ggml-quants.c/.h ref quant/dequant, ggml-cuda dequantize.cuh (device dequant), convert.cu
      (6 dispatchers), cpy-utils.cuh (quantize block), set-rows.cu (dispatch), ggml-cuda.cu
      (SET_ROWS gate), fattn.cu (kv_type_supported + need_f16 + kv_has_vec_instance exclusion —
      reused the batch=1 VEC-abort fix), arg.cpp (kv_cache_types). Full build OK (CPU+CUDA).
      VALIDATED: `-ctk q3_0` and `-ctk q2_0s` load + generate coherently on Gemma 4 E4B
      ("Paris."), no crash. These are plain linear quants (q3_0: d=max/-4; q2_0s: d=max/-2).
- [ ] **P2 — KVarN ggml ops.** GGML_OP_KVARN_WHT + KVARN_STORE: enum, ggml.c op wiring,
      CPU impl (ops.cpp), CUDA impl (kvarn.cu + kvarn-wht.cu). Test: op unit via cli.
- [ ] **P3 — llama-kvarn config + KV-cache class.** Port src/llama-kvarn.{cpp,h} +
      llama-kv-cache-kvarn.{cpp,h}; wire cparams.kvarn, llama.h enums, arg.cpp parsing.
- [ ] **P4 — FA KVarN CUDA kernels.** fattn-kvarn-dispatch/vec + fattn-mma-kvarn-* + the
      needed template instances (start k3-v3, k2-v2). Wire fattn.cu domain routing.
- [ ] **P5 — build + validate.** Full CUDA build; needle @256K kvarn3/kvarn2 vs beellama.

## Status log
- 2026-08-20: branch `kvarn-cuda` created off `23ff897`. Architecture mapped. Plan written.
- 2026-08-20: **P1 DONE** — Q3_0/Q2_0S fallback KV types wired (11 files) + built + validated
  (`-ctk q3_0`/`q2_0s` generate on Gemma 4 E4B). Next: **P2** — KVarN ggml ops (GGML_OP_KVARN_WHT
  = Walsh-Hadamard + GGML_OP_KVARN_STORE): enum in ggml.h, op wiring in ggml.c, CPU impl in
  ggml-cpu/ops.cpp, CUDA impl (port kvarn-wht.cu + kvarn.cu). Reference: beellama ggml.h
  GGML_OP_KVARN_WHT/STORE + ggml/src/ggml-cuda/kvarn-wht.{cu,cuh}, kvarn.{cu,cuh}.
