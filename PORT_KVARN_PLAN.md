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
- [x] **P2 — KVarN ggml ops. DONE 2026-08-20.** 4 ops (WHT/STORE/VIEW/MATERIALIZE), not 2.
      P2a (03a2baf): enum GGML_OP_COUNT 101->105, name/symbol arrays, op_params enum, 4 graph
      constructors (ggml.c/ggml.h). P2b: copied kvarn-wht.cu (233L) + kvarn.cu (2288L) + .cuh
      verbatim from beellama (self-contained, only include common.cuh; WHT_TYPE/LAUNCH macros
      defined in-file); added `add_compile_definitions(GGML_CUDA_KVARN)` to ggml-cuda CMake
      (GLOB auto-includes); dispatch cases + supports_op + includes in ggml-cuda.cu; ported
      ggml_backend_cuda_kvarn_ops (device_capabilities STUBBED to cc>=Turing until P4 FA).
      Full build clean (CUDA+CPU+server); f16 sanity gen OK. Not runtime-tested yet (needs P3
      to wire ops into a graph). CPU compute falls through to default (CUDA-only, fine).
- [~] **P3 — llama-kvarn config + KV-cache integration.** ADAPTED-MINIMAL path (user choice):
      keep the fork's existing KV-cache core (llama_memory_i unchanged, zero regression risk to
      dsv4/iswa/hybrid), integrate KVarN on top via the materialize->standard-FA path. NOT porting
      beellama's 2951-line llama-kv-cache-kvarn.cpp + its ~20 extra llama_memory_i virtuals +
      placement/tail-request.
      - [x] **P3a DONE (3805149):** config layer — llama.h enum llama_kvarn_type + llama_kvarn_params
            + API, ggml.h enum ggml_flash_attn_ext_kvarn_domain, llama-kvarn.cpp/h (descriptors,
            tile layout, attention planning) verbatim. llama.dll builds clean.
      - [!] **P3b FINDING (2026-08-20):** beellama's llama_kv_cache_kvarn is deeply coupled to
            beellama's *extended* KV-cache interfaces. Its context overrides **56** methods absent
            from our llama_kv_cache_context (all tail-*: get_tail_*, cpy_*_tail, build_input_tail_*,
            get_tail_route/storage_kind + new types llama_kv_tail_route/storage_kind/layer_route),
            plus ~20 on llama_memory_i. So "interface extension" (user chose A) is really ~76 methods
            + several new types = effectively the faithful port. Our base DOES have the core the
            minimal path needs (get_k/get_v/cpy_k/cpy_v/get_n_kv/type_k/type_v/build_input_k_idxs/
            v_idxs). Real minimal = STRIP the ported cache's tail overrides to fit our existing
            interface (get_k->materialize, cpy_k->store), dropping the 128-token exact-tail
            optimization (quality nicety, not correctness). That is delicate surgery on 2951 lines.
            Cache files stashed in _kvarn_p3b_pending/. AWAITING user decision: strip-surgery
            (large, uncertain) vs pause and bank P1/P2/P3a.
      - [ ] **P3b (was): allocate KVarN cache tensors (records i8 + stage f16)
            and wire the graph — on write call ggml_kvarn_store; on read call ggml_kvarn_materialize
            -> f16 K/V -> standard ggml_flash_attn_ext. arg.cpp: parse `kvarn2`/`kvarn3` -> set the
            llama_kvarn_type config. This is the riskiest part (new integration glue, not a port).
            Design decision needed: extend existing llama_kv_cache with a kvarn storage mode, or a
            thin wrapper cache. Reference beellama llama-kv-cache-kvarn.cpp for the store/materialize
            call shapes (tensor dims, stage_groups, indices), but write minimal glue for our core.
- [ ] **P4 — FA KVarN CUDA kernels.** fattn-kvarn-dispatch/vec + fattn-mma-kvarn-* + the
      needed template instances (start k3-v3, k2-v2). Wire fattn.cu domain routing.
- [ ] **P5 — build + validate.** Full CUDA build; needle @256K kvarn3/kvarn2 vs beellama.

## Status log
- 2026-08-20: branch `kvarn-cuda` created off `23ff897`. Architecture mapped. Plan written.
- 2026-08-20: **P1 DONE** — Q3_0/Q2_0S fallback KV types wired (11 files) + built + validated.
- 2026-08-20: **P2 DONE** — 4 KVarN ggml ops (WHT/STORE/VIEW/MATERIALIZE) scaffolded + CUDA
  impls (kvarn.cu/wht.cu) copied verbatim, compile+link clean. Next: **P3** — port
  src/llama-kvarn.{cpp,h} + src/llama-kv-cache-kvarn.{cpp,h}; wire cparams.kvarn, llama.h enums
  (llama_kvarn_type/config), arg.cpp `kvarn2/kvarn3` parsing, and the KV-cache graph build that
  calls ggml_kvarn_store/view/materialize. This makes the ops runtime-reachable and testable
  end-to-end (materialize path -> standard FA; native FA kvarn kernels are P4). Watch: base
  divergence in llama-kv-cache.cpp/llama-graph.cpp (our fork already carries q3_K/q2_K there).
