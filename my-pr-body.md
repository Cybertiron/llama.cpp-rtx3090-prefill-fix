## Overview

I found an issue with quants, that they become very slow on prefill. Later I googled the exact issue. Found out that q_8 is working perfectly fine, but it did not compress enough memory for single gpu to use newest qwen 3.8 model. Tested each quant individually to get results, which were unsatisfactory:

All "after" numbers measured at the same ~9000 token prompt for a fair comparison. Before the fix the small quants collapse and time out at long prompts, so those are at shorter prompts:

| KV | before | after (~9000 tok prompt) |
|----|--------|--------------------------|
| q4_0 | 74 t/s | 1182 t/s |
| q4_1 | ~157 | 1175 |
| q5_0 | slow / timeout | 1167 |
| q5_1 | 125 | 1164 |
| q8_0 | 1159 (already fast) | 1159 |

At equal prompt length all quant KV types now run within ~2% of each other, so the fix makes them all as fast as q8_0.

Than tried reverse engineer to find the problem with opus 4.8. Because it is working on q_8 i decided that the fix will be easy, as AI can find patterns and apply similar patterns for smaller quants. q4_0 was successfull from the start, later I ran into issues with q4_1, q5_0, q5_1. It took several hours to recompile them (they also need GGML_CUDA_FA_ALL_QUANTS=ON to be enabled), but the issue was fixed. After re-testing at the same prompt length all quants run at about the same speed. Still though I think there is room for improvement, but for maximum memory squeeze and performance q4_0 looks like the sweet spot for now. For a little bit better quality use q8_0, or q5_1 which sits between q4_0 and q8_0 in quality.

## Additional information

Fixes #27109. Tested on 2x RTX 3090 with Qwen 3.8 27B, outputs are numerically correct.

llama.cpp prebuild for rtx 3090 lower Quants fast prefill fix (if you don't want to compile): https://github.com/Cybertiron/llama.cpp/releases/tag/b27109-quant-kv-fix

## Requirements

- [x] I have read and agree with the contributing guidelines
- AI usage disclosure: YES - the CUDA kernel code was written with AI (Opus 4.8). I found the problem, directed the debugging and tested/verified everything myself on my own 2x RTX 3090.
