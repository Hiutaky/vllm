# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Micro-benchmark of one Inkling2 MoE layer (Humming experts, Mini sizes)."""

import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, "/tests")
sys.path.insert(0, "/tests/models/inkling2")
from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.models.inkling2.nvidia.moe import Inkling2MoE  # noqa: E402

cfg = SimpleNamespace(
    hidden_size=2304, intermediate_size=1152, n_routed_experts=128,
    n_shared_experts=0, num_experts_per_tok=8, route_scale=8.0,
    mlp_gate_softcap=4.0, mlp_up_softcap=25.0, use_gate_bias=True,
)


def rnd_nvfp4(*shape):
    v = torch.randint(0, 256, (*shape[:-1], shape[-1] // 2), device="cuda").to(torch.uint8)
    s = (torch.rand(*shape[:-1], shape[-1] // 16, device="cuda") + 0.5).to(torch.float8_e4m3fn)
    return v, s


def rnd_mxfp8(*shape):
    v = torch.randn(shape, device="cuda").to(torch.float8_e4m3fn)
    s = torch.randint(120, 128, (*shape[:-1], shape[-1] // 32), device="cuda").to(torch.uint8)
    return v, s


from vllm.distributed import init_distributed_environment, initialize_model_parallel  # noqa

import os  # noqa
os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29533")
_ctx = set_current_vllm_config(VllmConfig())
_ctx.__enter__()
init_distributed_environment(1, 0, "tcp://127.0.0.1:29533", 0, "nccl")
initialize_model_parallel(1, 1)

for fmt in ("nvfp4", "mxfp8"):
    q = rnd_nvfp4 if fmt == "nvfp4" else rnd_mxfp8
    w = {"gate.weight": (torch.randn(128, 2304, device="cuda") * 0.05).bfloat16(),
         "gate.bias": torch.randn(128, device="cuda") * 0.01}
    for n, (name, nn_, kk) in enumerate((("w13", 2304, 2304), ("w2", 2304, 1152))):
        p = f"experts.{name}_weight"
        w[p], w[p + "_scale"] = q(128, nn_, kk)
        if fmt == "nvfp4":
            w[p + "_amax"] = torch.rand(128, device="cuda") * 100 + 100
            w[p + "_outer_scale_denominator"] = torch.tensor([2688.0], device="cuda")
    with set_current_vllm_config(VllmConfig()), torch.device("cuda"):
        moe = Inkling2MoE(cfg, fmt)
    for name, param in moe.named_parameters():
        param.weight_loader(param, w[name])
    moe.experts.process_weights_after_loading()
    for t in (1, 8, 64):
        x = torch.randn(t, 2304, device="cuda").bfloat16()
        for _ in range(5):
            moe(x)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            moe(x)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(50):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        ms = e0.elapsed_time(e1) / 50
        print(f"{fmt} tokens={t:3d}: {ms:.3f} ms/layer  ({t / ms * 1000:.0f} tok/s/layer)", flush=True)
    del moe, w
    torch.cuda.empty_cache()
