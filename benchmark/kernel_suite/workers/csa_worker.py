# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-case benchmark worker for the CSA fused-compressor kernels.

Times one ``cudnn.csa`` compressor call (the gated-softmax pooling region:
every ``ratio`` tokens are pooled into one compressed token) and prints a
parseable ``RESULT,`` line. Called as a subprocess by ``components/csa.py`` so
each case gets a clean CUDA context and CuTe DSL compile state.

The wrappers route by ratio (``python/cudnn/csa/compressor/api.py``):

  ratio {2, 4}   generic kernels (``compressor_sm100.py``; whole window in
                 registers)
  ratio 128      dedicated kernels (``compressor_sm100_r128.py``; bucketed-
                 schedule chunked-softmax forward, staged-smem backward) —
                 the 128-token compress path

``coff=1`` is the own-block window (window = ratio tokens), ``coff=2`` the
production overlapping window (2 * ratio tokens).

Timing: CUDA events around each steady-state call after a 256 MiB L2 flush,
median of ``--num_iterations`` after warmup (JIT compilation happens in
warmup). The wrapper allocates its output buffers on every call, so
allocation is inside the timed region (same convention as
``benchmark/csa/bench_csa_compressor.py``). Prints
``RESULT,ms,tflops,tb_per_sec,gpu,cudnn_version,detail``; exit code 3 with
``UNSUPPORTED,<reason>`` marks an honestly unserved case.
"""

import argparse
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


def make_inputs(lens, d, ratio, coff, seed):
    """THD pack: kv/score (total, coff*d) bf16, ape (ratio, coff*d) fp32,
    cu/cu_comp int32, grad_out (total_comp, d) bf16. Tail tokens
    (len % ratio) are dropped by the kernel, matching the eager semantics."""
    total = sum(lens)
    w = coff * d
    torch.manual_seed(seed)
    kv = torch.randn(total, w, dtype=torch.bfloat16, device="cuda")
    score = torch.randn(total, w, dtype=torch.float32, device="cuda").mul_(1.5).to(torch.bfloat16)
    ape = torch.randn(ratio, w, dtype=torch.float32, device="cuda").mul_(0.25)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0)], dtype=torch.int32, device="cuda")
    seg_comp = torch.tensor([seg_len // ratio for seg_len in lens])
    cuc = torch.tensor([0, *seg_comp.cumsum(0)], dtype=torch.int32, device="cuda")
    total_comp = int(cuc[-1].item())
    grad_out = torch.randn(total_comp, d, dtype=torch.bfloat16, device="cuda")
    return kv, score, ape, cu, cuc, total_comp, grad_out


# ---------------------------------------------------------------------------
# Per-pass builders. Each returns (run_fn, flops, moved_bytes, detail).
# ---------------------------------------------------------------------------


def build_fwd(args):
    """Compressor forward: out[b, j] = sum_k kv * softmax(score + ape) over the
    per-block window (coff * ratio tokens). 2 FLOPs per (token, column) MAC;
    the softmax exp/adds are not counted (pooling is bandwidth-bound)."""
    import cudnn.csa

    lens, d, ratio, coff = args.parsed_seqlens, args.head_dim, args.ratio, args.coff
    kv, score, ape, cu, cuc, total_comp, _ = make_inputs(lens, d, ratio, coff, args.seed)
    total, w = kv.shape

    def run():
        cudnn.csa.csa_compressor_forward_wrapper(kv, score, ape, cu, cuc, ratio=ratio, head_dim=d, coff=coff, total_comp=total_comp)

    flops = 2 * total * w
    moved = 2 * total * w * _BF16 + ratio * w * _FP32 + total_comp * d * _BF16
    return run, flops, moved, f"csa_compressor_forward_wrapper r{ratio} coff{coff} (softmax FLOPs not counted; output alloc included)"


def build_bwd(args):
    """Compressor backward: dkv = weights*grad_out, dscore = weights*(kv*grad_out
    - rowsum), dAPE by fp32 atomics — approximately 3x the forward MACs."""
    import cudnn.csa

    lens, d, ratio, coff = args.parsed_seqlens, args.head_dim, args.ratio, args.coff
    kv, score, ape, cu, cuc, total_comp, grad_out = make_inputs(lens, d, ratio, coff, args.seed)
    total, w = kv.shape

    def run():
        cudnn.csa.csa_compressor_backward_wrapper(kv, score, ape, cu, cuc, grad_out, ratio=ratio, head_dim=d, coff=coff)

    flops = 6 * total * w  # ~3x forward (dkv / dscore_pre / dscore); approximate
    moved = 4 * total * w * _BF16 + 2 * ratio * w * _FP32 + total_comp * d * _BF16
    return run, flops, moved, f"csa_compressor_backward_wrapper r{ratio} coff{coff} (approx FLOPs; grad allocs + dAPE zero-fill included)"


BUILDERS = {"fwd": build_fwd, "bwd": build_bwd}


def unsupported(reason: str):
    print(f"UNSUPPORTED,{reason.replace(',', ';')}")
    sys.exit(UNSUPPORTED_EXIT_CODE)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile_pass", required=True, choices=["fwd", "bwd"])
    p.add_argument("--ratio", type=int, required=True, help="compression ratio: tokens pooled per output block")
    p.add_argument("--coff", type=int, required=True, help="window form: 1 = own block, 2 = overlapping (2*ratio window)")
    p.add_argument("--head_dim", type=int, required=True, help="output feature dimension d (kv/score width is coff*d)")
    p.add_argument("--seqlens", required=True, help="comma-separated THD pack, e.g. '8192' or '65536,65536'")
    p.add_argument("--num_iterations", type=int, default=20)
    p.add_argument("--num_warmup_iterations", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    args.parsed_seqlens = [int(x) for x in args.seqlens.split(",") if x]
    return args


@torch.no_grad()
def main():
    args = parse_args()
    if not torch.cuda.is_available():
        unsupported("CUDA not available")
    major, minor = torch.cuda.get_device_capability()
    if major < 10:
        unsupported(f"CSA compressor kernels require SM100+, found SM{major}{minor}")
    if args.ratio not in (2, 4, 128):
        unsupported(f"ratio must be in {{2, 4, 128}}, got {args.ratio}")
    if args.coff not in (1, 2):
        unsupported(f"coff must be in {{1, 2}}, got {args.coff}")
    if args.ratio == 128 and args.head_dim not in (128, 512):
        unsupported(f"ratio=128 requires head_dim in {{128, 512}}, got {args.head_dim}")
    total = sum(args.parsed_seqlens)
    if total * args.coff * args.head_dim >= 2**31:
        unsupported(f"total_tokens * coff * head_dim must be < 2^31 (int32 offsets), got {total * args.coff * args.head_dim}")
    try:
        import cudnn
        import cudnn.csa  # noqa: F401
    except (ImportError, AttributeError) as e:
        unsupported(f"cudnn not available: {e}")

    try:
        run, flops, moved, detail = BUILDERS[args.profile_pass](args)
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
