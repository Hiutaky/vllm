# Decode-time microbenchmark of the Triton rel_attention kernel (Ampere path), bf16 KV.
import torch
from vllm.models.inkling2.nvidia.ops.attention_sm90 import MODE_OUT_LSE, MODE_PARTIAL, rel_attention

H, KVH, D, PAGE = 24, 8, 128, 16  # per-GPU heads of Inkling2 (GQA 3)
def bench(ctx, bs, window, splits):
    nblk = (ctx + PAGE - 1) // PAGE
    kv = torch.randn(bs * nblk, PAGE, KVH, 2 * D, device="cuda").bfloat16()
    k, v = kv.split(D, -1)
    table = torch.arange(bs * nblk, device="cuda", dtype=torch.int32).view(bs, nblk)
    lens = torch.full((bs,), ctx, device="cuda", dtype=torch.int32)
    cu = torch.arange(bs + 1, device="cuda", dtype=torch.int32)
    q = torch.randn(bs, H, D, device="cuda").bfloat16()
    rel = torch.randn(bs + 128, H, 1024 + 256, device="cuda").bfloat16()
    if splits == 1:
        out = torch.empty(bs, H, D, device="cuda"); lse = torch.empty(H, bs, device="cuda"); mode = MODE_OUT_LSE
    else:
        out = torch.empty(splits, bs, H, D, device="cuda"); lse = torch.empty(splits, bs, H, device="cuda"); mode = MODE_PARTIAL
    f = lambda: rel_attention(q, k, v, out, lse, rel, block_table=table, cache_seqlens=lens, cu_seqlens_q=cu,
        max_query_len=1, softmax_scale=D**-0.5, window=window, num_splits=splits, mode=mode)
    for _ in range(5): f()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(50): f()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / 50 * 1000
for ctx in (4096, 32768):
    for bs in (1, 8):
        for name, w in (("global", 0), ("local1024", 1024)):
            r = {sp: bench(ctx, bs, w, sp) for sp in (1, 4, 16)}
            print(f"ctx={ctx} bs={bs} {name}: " + " ".join(f"splits={k}: {v:.1f}us" for k, v in r.items()))
