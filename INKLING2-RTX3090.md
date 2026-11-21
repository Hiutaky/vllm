# Inkling-2-Mini on a single RTX 3090 (Ampere, SM86)

This fork is [vLLM PR #69028](https://github.com/vllm-project/vllm/pull/69028) (Inkling-2 support) plus
a set of commits (`ampere-port`) that make it run on Ampere GPUs. Hopper and Blackwell code paths are
unchanged: every new branch is guarded by a device-capability check (below SM90 or SM89), and the only
change that applies to all GPUs (`wrap_modules` on `embed_tokens`) does nothing unless
`--cpu-offload-*` is passed.

## What changed

| Commit | Why |
|---|---|
| PDL only on SM90+ | `griddepcontrol` does not assemble before SM90 |
| Minimum capability 80, linear without TMA | TMA descriptors need SM90+ |
| `conv_mix` fallback | native kernels need Hopper/SM10x; Ampere uses Triton `sconv` + `attn_res` |
| MoE tests on Ampere | Humming experts run on SM8x |
| Attention/vision on SM86 | FA2 for the vision tower |
| Emulated FP8 (E4M3) KV cache | no hardware E4M3 conversions before SM89; encode/decode with bit operations |
| DSpark draft KV in bf16 | the draft's FA2 cannot read E4M3 |
| CPU offload of `embed_tokens` | frees ~0.86 GiB of VRAM for the KV cache |

## Tested configuration

- 1x RTX 3090 (24 GB), TP=1, checkpoint `thinkingmachines/Inkling-2-Mini` (NVFP4 experts, MXFP8 dense).
- Image `vllm/vllm-openai:nightly` (vLLM 0.34.0rc1, torch 2.14 cu130, Triton 3.8, Python 3.14) with the
  Python package replaced by this fork's `vllm/` directory (no C++ rebuild; the Ampere paths are Triton/torch).
  Copy the image's `vllm` package, overlay the `*.py`/`*.json`/`*.yaml` files of this repo on top and mount it
  at `/usr/local/lib/python3.14/dist-packages/vllm`. An editable install with precompiled wheels
  (`VLLM_USE_PRECOMPILED=1`) should also work but has **not** been tested.

```bash
vllm serve /model --served-model-name inkling-2-mini --dtype bfloat16 \
  --max-model-len 131072 --max-num-seqs 8 --max-num-batched-tokens 256 \
  --gpu-memory-utilization 0.98 --kv-cache-memory=5153960755 \
  --kv-cache-dtype fp8 --limit-mm-per-prompt.image=0 \
  --cpu-offload-gb 1 --cpu-offload-params embed_tokens \
  --compilation-config.cudagraph_mode=FULL_DECODE_ONLY \
  --speculative-config.method=dspark --speculative-config.num_speculative_tokens=7 \
  --enable-auto-tool-choice --tool-call-parser inkling --reasoning-parser inkling
```

Notes: DSpark needs exactly 7 speculative tokens (block size 8); `--limit-mm-per-prompt.image=0`
removes the vision tower to save VRAM; `--kv-cache-memory` must leave about 0.3 GiB free for lazy kernel loading.

## Measured numbers (single 3090)

- Decode with DSpark: 112-147 tok/s (91 tok/s without it).
- Prefill: about 2,500-2,700 tok/s up to 33k tokens, 1,870 tok/s at 126k.
- KV pool: 207k tokens with 8 concurrent sequences (224k with 2); context 131,072.
- CPU offload of the embedding: decode about -1%, prefill -1% to -3%.

## Known limits

- Quality benchmarks (HumanEval+, IFEval) with FP8 KV + DSpark have not been re-run; FP8 KV is a precision change.
- The FP8 KV tests below SM89 are still skipped even though the emulation now exists.
- Not tested on other GPUs: Hopper/Blackwell behaviour is unchanged by construction, not by measurement.
- The memory budget is tight (about 23.8 of 24 GB used); 8 concurrent sequences needed a smaller KV pool than 2.
