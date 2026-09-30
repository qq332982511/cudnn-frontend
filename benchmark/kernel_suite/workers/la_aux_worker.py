# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-case benchmark worker for the linear-attention summary kernels.

Times one standalone per-span summary op (the context-parallelism state /
transition kernels of GDN/KDA) and prints a parseable ``RESULT,`` line.
Called as a subprocess by ``components/la_aux.py`` so each case gets a clean
CUDA context and kernel compile state.

Ops (``cudnn.linear_attention.ops``, THD packed layout, bf16 io, fp32 gates):

  gdn_summary / kda_summary          state-only forward over each sequence:
                                     final_state (N, HO, V, K) + span
                                     transition (N, HO, K, K), fp32
  gdn_summary_bwd / kda_summary_bwd  reverse state-gradient recurrence:
                                     d_initial_state (N, HO, V, K) +
                                     transition (N, HO, K, K), fp32

These are the same kernels the fused op launches internally (the breakdown
report shows them as gdn_summary / gdn_bprop_summary / kda split_k_scan rows);
here they are timed standalone with TFLOPS + TB/s like every other op.

Input conventions mirror the linear-attention harness (k L2-normalized, zero
log-gate => alpha = 1, in-kernel Q/K L2 norm on; KDA adds its safe-gate
plumbing). Timing: CUDA events around each steady-state call after a 256 MiB
L2 flush, median of ``--num_iterations`` after warmup. Prints
``RESULT,ms,tflops,tb_per_sec,gpu,cudnn_version,detail``; exit code 3 with
``UNSUPPORTED,<reason>`` marks an honestly unserved case.
"""

import argparse
import math
import sys

import torch

UNSUPPORTED_EXIT_CODE = 3

_BF16 = 2
_FP32 = 4


def time_fn(fn, num_warmup, num_iters) -> float:
    """Median per-call GPU time of fn() in ms via CUDA events, L2 flushed before each call."""
    l2_flush = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.int8)
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(num_iters):
        l2_flush.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def make_thd_inputs(args, need_q=False, need_v=True, need_do=False):
    """THD pack for the summary ops, mirroring the harness conventions
    (k L2-normalized, zero log-gate, random beta; KDA per-channel gate)."""
    variant, b, t = args.variant, args.batch_size, args.seqlen
    hq, hkv, d = args.num_q_heads, args.num_kv_heads, args.head_dim
    ho = max(hq, hkv)
    total = b * t
    torch.manual_seed(args.seed)
    device = "cuda"
    k = torch.nn.functional.normalize(torch.randn(total, hq, d, dtype=torch.float32, device=device), dim=-1).to(torch.bfloat16)
    v = torch.randn(total, hkv, d, dtype=torch.bfloat16, device=device) if need_v else None
    q = torch.randn(total, hq, d, dtype=torch.bfloat16, device=device) if need_q else None
    if variant == "kda":
        g = torch.zeros(total, ho, d, dtype=torch.float32, device=device)
    else:
        g = torch.zeros(total, ho, dtype=torch.float32, device=device)
    beta = torch.randn(total, ho, dtype=torch.float32, device=device)
    do = torch.randn(total, ho, d, dtype=torch.bfloat16, device=device) if need_do else None
    cu_seqlens = torch.arange(0, b + 1, dtype=torch.int32, device=device) * t
    return q, k, v, g, beta, do, cu_seqlens, ho, total


def _gate_kwargs(args, ho, d):
    """KDA safe-gate plumbing (matches the harness's KDA call); empty for GDN."""
    if args.variant != "kda":
        return {}
    device = "cuda"
    return dict(
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        gate_lower_bound=-5.0,
        a_log=torch.zeros(ho, dtype=torch.float32, device=device),
        dt_bias=torch.zeros(ho, d, dtype=torch.float32, device=device),
    )


def build_summary(args):
    """Per-span forward summary: final_state + span transition (context parallel)."""
    from cudnn.linear_attention.ops import gated_delta_net_summary, kimi_delta_attention_summary

    _, k, v, g, beta, _, cu_seqlens, ho, total = make_thd_inputs(args, need_v=True)
    d, n = args.head_dim, args.batch_size
    op = gated_delta_net_summary if args.variant == "gdn" else kimi_delta_attention_summary

    def run():
        op(k, v, g, beta, cu_seqlens, use_qk_l2norm_in_kernel=True, **_gate_kwargs(args, ho, d))

    flops = 4 * total * ho * d * d  # per token/head: k-outer-v state update + transition update (approximate)
    g_elems = g.numel()
    moved = k.numel() * _BF16 + v.numel() * _BF16 + g_elems * _FP32 + beta.numel() * _FP32 + 2 * n * ho * d * d * _FP32
    return run, flops, moved, f"{args.variant}_summary state+transition (context-parallel fwd; approximate FLOPs)"


def build_summary_bwd(args):
    """Per-span backward summary: d_initial_state + transition (context parallel)."""
    from cudnn.linear_attention.ops import gated_delta_net_summary_bwd, kimi_delta_attention_summary_bwd

    q, k, _, g, beta, do, cu_seqlens, ho, total = make_thd_inputs(args, need_q=True, need_v=False, need_do=True)
    d, n = args.head_dim, args.batch_size
    scale = 1.0 / math.sqrt(d)
    op = gated_delta_net_summary_bwd if args.variant == "gdn" else kimi_delta_attention_summary_bwd

    def run():
        op(q, k, g, beta, do, cu_seqlens, scale=scale, output_transition=True, use_qk_l2norm_in_kernel=True, **_gate_kwargs(args, ho, d))

    flops = 6 * total * ho * d * d  # G += dO-outer-q + reverse state recurrence + transition (approximate)
    moved = q.numel() * _BF16 + k.numel() * _BF16 + g.numel() * _FP32 + beta.numel() * _FP32 + do.numel() * _BF16 + 2 * n * ho * d * d * _FP32
    return run, flops, moved, f"{args.variant}_summary_bwd dstate+transition (context-parallel bwd; approximate FLOPs)"


BUILDERS = {
    "gdn_summary": build_summary,
    "gdn_summary_bwd": build_summary_bwd,
    "kda_summary": build_summary,
    "kda_summary_bwd": build_summary_bwd,
}


def unsupported(reason: str):
    print(f"UNSUPPORTED,{reason.replace(',', ';')}")
    sys.exit(UNSUPPORTED_EXIT_CODE)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--op", required=True, choices=sorted(BUILDERS))
    p.add_argument("--variant", required=True, choices=["gdn", "kda"])
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--seqlen", type=int, required=True, help="tokens per sequence (all sequences equal length)")
    p.add_argument("--num_q_heads", type=int, required=True)
    p.add_argument("--num_kv_heads", type=int, required=True)
    p.add_argument("--head_dim", type=int, default=128)
    p.add_argument("--num_iterations", type=int, default=20)
    p.add_argument("--num_warmup_iterations", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    if not torch.cuda.is_available():
        unsupported("CUDA not available")
    major, minor = torch.cuda.get_device_capability()
    if major < 9:
        unsupported(f"linear-attention summary kernels require SM90+, found SM{major}{minor}")
    try:
        import cudnn
        from cudnn.linear_attention import ops  # noqa: F401
    except (ImportError, AttributeError) as e:
        unsupported(f"cudnn not available: {e}")

    try:
        run, flops, moved, detail = BUILDERS[args.op](args)
        run()
        torch.cuda.synchronize()
    except (ValueError, RuntimeError) as e:
        unsupported(str(e))

    ms = time_fn(run, args.num_warmup_iterations, args.num_iterations)
    tflops = flops / (ms * 1e-3) / 1e12
    tb_per_sec = moved / (ms * 1e-3) / 1e12
    gpu = torch.cuda.get_device_name().replace(",", ";")
    print(f"RESULT,{ms:.6f},{tflops:.3f},{tb_per_sec:.3f},{gpu},{cudnn.__version__},{detail.replace(',', ';')}")


if __name__ == "__main__":
    main()
