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
- [ ] **P1 — fallback quant types.** Add GGML_TYPE_Q3_0 + Q2_0S (min for kvarn3/kvarn2):
      ggml.h enum, ggml-common.h block structs, ggml.c type_traits, ggml-quants.c ref
      quant/dequant, ggml-cpu quants, ggml-cuda dequant + convert + set-rows + fattn gate.
      Reuse the q3_K/q2_K KV-wiring pattern already in this fork. Test: `-ctk q3_0` loads.
- [ ] **P2 — KVarN ggml ops.** GGML_OP_KVARN_WHT + KVARN_STORE: enum, ggml.c op wiring,
      CPU impl (ops.cpp), CUDA impl (kvarn.cu + kvarn-wht.cu). Test: op unit via cli.
- [ ] **P3 — llama-kvarn config + KV-cache class.** Port src/llama-kvarn.{cpp,h} +
      llama-kv-cache-kvarn.{cpp,h}; wire cparams.kvarn, llama.h enums, arg.cpp parsing.
- [ ] **P4 — FA KVarN CUDA kernels.** fattn-kvarn-dispatch/vec + fattn-mma-kvarn-* + the
      needed template instances (start k3-v3, k2-v2). Wire fattn.cu domain routing.
- [ ] **P5 — build + validate.** Full CUDA build; needle @256K kvarn3/kvarn2 vs beellama.

## Status log
- 2026-08-20: branch `kvarn-cuda` created off `23ff897`. Architecture mapped. Plan written.
  Next: P1 — bring in Q3_0/Q2_0S block structs + traits.
