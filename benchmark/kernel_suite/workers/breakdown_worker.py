# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-kernel profiler breakdown worker.

Profiles one suite case (one component, one shape, one pass) with
torch.profiler and prints one ``KERNEL,`` line per CUDA kernel observed:

    KERNEL,<pass>,<kernel_name>,<calls>,<total_us>

Used by ``breakdown.py`` (one subprocess per case) to report where the time
goes inside the fused ops — including the auxiliary kernels (index convert /
padding in DSA, l2norm / chunk scan / state passing in GDN/KDA) that the
op-level timing rolls up.

Builders:
- ``dsa``: reuses ``benchmark.dsa.benchmark_single_dsa`` setup_fwd/setup_bwd
  (same inputs as the timed cases).
- ``dsa_aux``: reuses the builders in ``workers/dsa_aux_worker.py``.
- ``csa``: reuses the builders in ``workers/csa_worker.py`` (the compressor
  is 1 kernel fwd / 1 kernel + dAPE zero-fill bwd; the breakdown confirms it).
- ``la_aux``: reuses the builders in ``workers/la_aux_worker.py`` (standalone
  GDN/KDA per-span summary ops).
- ``gdn`` / ``kda``: builds THD inputs directly, mirroring the linear
  attention harness conventions (k L2-normalized, zero log-gate = alpha 1,
  in-kernel Q/K L2 norm on; KDA adds its safe-gate plumbing).
"""

import argparse
import math
import sys
from pathlib import Path
from types import SimpleNamespace

# This file is launched as a plain script by breakdown.py, so sys.path[0] is
# the workers/ directory. Put the repo root on the path for `benchmark.*` and
# `benchmark.kernel_suite.workers.*` imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from torch.profiler import ProfilerActivity, profile


def profile_run(fn, profile_pass, warmup=2, iters=3):
    """Warm up (JIT compile outside the profile), then profile `iters` calls."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    rows = []
    for item in prof.key_averages():
        total_us = getattr(item, "device_time_total", 0.0)
        if total_us and total_us > 0:
            name = item.key.replace(",", ";")
            rows.append((name, item.count, total_us))
    rows.sort(key=lambda r: -r[2])
    for name, count, total_us in rows:
        print(f"KERNEL,{profile_pass},{name},{count},{total_us / iters:.1f}")
    print("BREAKDOWN_OK")


def build_dsa(args):
    """Main sparse attention fwd/bwd via the DSA benchmark harness builders."""
    from benchmark.dsa import benchmark_single_dsa as dsa_worker

    device = torch.device("cuda")
    s_q, s_kv, heads = args.q_seqlen, args.kv_seqlen, args.num_q_heads
    d_qk, d_v = args.head_dim_qk, args.head_dim_vo
    softmax_scale = 1.0 / math.sqrt(d_qk)
    q = torch.randn(s_q, heads, d_qk, dtype=torch.bfloat16, device=device).mul_(0.1)
    kv = torch.randn(s_kv, d_qk, dtype=torch.bfloat16, device=device).mul_(0.1)
    topk_idxs = dsa_worker.make_topk_idxs(s_q, s_kv, args.topk, device)
    attn_sink = torch.linspace(-2.0, 2.0, heads, dtype=torch.float32, device=device) if args.has_sink else None
    topk_length = torch.full((s_q,), args.topk, dtype=torch.int32, device=device)
    worker_args = SimpleNamespace(indexer_topk=args.indexer_topk, head_dim_vo=d_v, topk=args.topk, deterministic=False)
    setup = dsa_worker.setup_fwd if args.profile_pass == "fwd" else dsa_worker.setup_bwd
    fn, detail = setup(worker_args, q, kv, topk_idxs, attn_sink, topk_length, softmax_scale)
    return fn, detail


def build_dsa_aux(args):
    from benchmark.kernel_suite.workers.dsa_aux_worker import BUILDERS

    worker_args = SimpleNamespace(seqlen=args.seqlen, heads=args.heads, top_k=args.top_k, ratio=args.ratio)
    run, _flops, _moved, detail = BUILDERS[args.op](worker_args)
    return run, detail


def build_csa(args):
    from benchmark.kernel_suite.workers.csa_worker import BUILDERS

    worker_args = SimpleNamespace(
        parsed_seqlens=[int(x) for x in args.seqlens.split(",") if x], head_dim=args.head_dim, ratio=args.ratio, coff=args.coff, seed=0
    )
    run, _flops, _moved, detail = BUILDERS[args.profile_pass](worker_args)
    return run, detail


def build_la_aux(args):
    from benchmark.kernel_suite.workers.la_aux_worker import BUILDERS

    worker_args = SimpleNamespace(
        variant=args.variant,
        batch_size=args.batch_size,
        seqlen=args.seqlen,
        num_q_heads=args.num_q_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        seed=0,
    )
    run, _flops, _moved, detail = BUILDERS[args.op](worker_args)
    return run, detail


def build_la(args):
    """THD inputs for gated_delta_net / kimi_delta_attention, mirroring the
    linear attention harness (zero log-gate => alpha = 1, random beta, k
    L2-normalized, in-kernel Q/K L2 norm on)."""
    from cudnn.linear_attention.ops import gated_delta_net, kimi_delta_attention

    device = torch.device("cuda")
    variant, b, t = args.variant, args.batch_size, args.seqlen
    hq, hkv, d = args.num_q_heads, args.num_kv_heads, args.head_dim
    ho = max(hq, hkv)
    total = b * t
    torch.manual_seed(0)
    q = torch.randn(total, hq, d, dtype=torch.bfloat16, device=device)
    k = torch.nn.functional.normalize(torch.randn(total, hq, d, dtype=torch.float32, device=device), dim=-1).to(torch.bfloat16)
    v = torch.randn(total, hkv, d, dtype=torch.bfloat16, device=device)
    if variant == "kda":
        g = torch.zeros(total, ho, d, dtype=torch.float32, device=device)
    else:
        g = torch.zeros(total, ho, dtype=torch.float32, device=device)
    beta = torch.randn(total, ho, dtype=torch.float32, device=device)
    cu_seqlens = torch.arange(0, b + 1, dtype=torch.int32, device=device) * t
    scale = 1.0 / math.sqrt(d)

    op = gated_delta_net if variant == "gdn" else kimi_delta_attention

    def fwd(pass_q, pass_k, pass_v, pass_g, pass_beta):
        if variant == "kda":
            return op(
                pass_q,
                pass_k,
                pass_v,
                pass_g,
                pass_beta,
                cu_seqlens,
                scale=scale,
                use_qk_l2norm_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                safe_gate=True,
                gate_lower_bound=-5.0,
                a_log=torch.zeros(ho, dtype=torch.float32, device=device),
                dt_bias=torch.zeros(ho, d, dtype=torch.float32, device=device),
            )
        return op(pass_q, pass_k, pass_v, pass_g, pass_beta, cu_seqlens, scale=scale, use_qk_l2norm_in_kernel=True)

    if args.profile_pass == "fwd":
        fn = lambda: fwd(q, k, v, g, beta)  # noqa: E731
        return fn, f"{variant} fwd THD B{b} T{t}"

    for tensor in (q, k, v, g, beta):
        tensor.requires_grad_(True)
    out0 = fwd(q, k, v, g, beta)  # shape probe (graph discarded)
    dout = torch.randn_like(out0[0])

    def bwd():
        out = fwd(q, k, v, g, beta)
        torch.autograd.backward(out[0], dout)
        q.grad = k.grad = v.grad = g.grad = beta.grad = None

    return bwd, f"{variant} bwd THD B{b} T{t}"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--component", required=True, choices=["dsa", "dsa_aux", "csa", "gdn", "kda", "la_aux"])
    p.add_argument("--profile_pass", required=True, choices=["fwd", "bwd"])
    # dsa (main sparse attention)
    p.add_argument("--q_seqlen", type=int, default=0)
    p.add_argument("--kv_seqlen", type=int, default=0)
    p.add_argument("--num_q_heads", type=int, default=0)
    p.add_argument("--head_dim_qk", type=int, default=512)
    p.add_argument("--head_dim_vo", type=int, default=512)
    p.add_argument("--topk", type=int, default=0)
    p.add_argument("--indexer_topk", type=int, default=0)
    p.add_argument("--has_sink", action="store_true")
    # dsa_aux
    p.add_argument("--op", default="")
    p.add_argument("--heads", type=int, default=64)
    p.add_argument("--top_k", type=int, default=640)
    p.add_argument("--ratio", type=int, default=4)
    # csa
    p.add_argument("--coff", type=int, default=2)
    p.add_argument("--seqlens", default="", help="CSA: comma-separated THD pack, e.g. '65536,65536'")
    # gdn / kda
    p.add_argument("--variant", default="gdn")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seqlen", type=int, default=0, help="LA: tokens per sequence; dsa_aux: T")
    p.add_argument("--num_kv_heads", type=int, default=0)
    p.add_argument("--head_dim", type=int, default=128)
    p.add_argument("--iters", type=int, default=3, help="profiled iterations (kernel times are averaged over them)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.component == "dsa":
        fn, detail = build_dsa(args)
    elif args.component == "dsa_aux":
        fn, detail = build_dsa_aux(args)
    elif args.component == "csa":
        fn, detail = build_csa(args)
    elif args.component == "la_aux":
        fn, detail = build_la_aux(args)
    else:
        fn, detail = build_la(args)
    print(f"# case: {args.component} {args.profile_pass} {detail}", file=sys.stderr)
    profile_run(fn, args.profile_pass, iters=args.iters)


if __name__ == "__main__":
    main()
