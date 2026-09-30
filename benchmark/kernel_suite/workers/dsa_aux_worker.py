# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-case benchmark worker for the DSA auxiliary kernels.

Times one auxiliary op of the DSA stack (everything around the main sparse
attention kernel) and prints a parseable ``RESULT,`` line. Called as a
subprocess by ``components/dsa_aux.py`` so each case gets a clean CUDA
context and CuTe DSL compile state.

Ops (all via the public ``cudnn.DSA`` namespace, bf16, BSHD with b=1):

  indexer_fwd            dense indexer scores ``sum_h ReLU(Q_h.K_h^T).W_h``
                         (``indexer_forward_wrapper``, ratio-causal mask,
                         out (1,T,T/ratio) fp32)
  indexer_fwd_topk       fused compressed-score + radix top-K + softmax
                         (``indexer_forward_top_k_wrapper``, SM100-only, the
                         production indexer path)
  topk                   standalone radix top-K over an (n_rows, num_cols)
                         score matrix (``indexer_top_k_wrapper``)
  index_score_recompute  training-loss helper re-scoring the selected
                         positions from the indexer (predict scores)
  attn_score_recompute   same from the attention side (target scores)
  indexer_bwd            indexer backward (``indexer_backward_wrapper``),
                         the only backward among the auxiliary ops
  local_to_global        local top-K indices -> global index space
                         (``local_to_global_wrapper``, pure index transform)
  compactify             row-pack valid indices + emit ``topk_length``
                         (``compactify_wrapper``)
  dense_index_score_recompute / dense_attn_score_recompute
                         dense (non-sparse) full-KV counterparts of the two
                         score-recompute helpers; causal, the full
                         ``(T, s_k)`` / ``(T, T)`` fp32 matrix is written
  dense_indexer_bwd      dense full-KV indexer backward
                         (``dense_indexer_backward_wrapper``)

Shape convention follows production: with T tokens, the indexer scores
s_k = T / ratio compressed positions per query, and the attention side
selects top_k = 640 entries (the dsv4_flash gather width).

Timing: CUDA events around each steady-state call after a 256 MiB L2 flush,
median of ``--num_iterations`` after warmup (JIT compilation happens in
warmup). Prints ``RESULT,ms,tflops,tb_per_sec,gpu,cudnn_version,detail``;
exit code 3 with ``UNSUPPORTED,<reason>`` marks an honestly unserved case.
"""

import argparse
import math
import sys

import torch

UNSUPPORTED_EXIT_CODE = 3

_BF16 = 2
_FP32 = 4
_INT32 = 4


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


def make_topk_indices(b, s_q, s_k, topk, device, chunk=1024):
    """Per-row unique random column ids < s_k, (b, s_q, topk) int32, chunked to bound memory."""
    idxs = torch.empty(b, s_q, topk, dtype=torch.int32, device=device)
    for i in range(0, b * s_q, chunk):
        n = min(chunk, b * s_q - i)
        idxs.view(-1, topk)[i : i + n] = torch.rand(n, s_k, device=device).argsort(dim=-1)[:, :topk].to(torch.int32)
    return idxs


# ---------------------------------------------------------------------------
# Per-op builders. Each returns (run_fn, flops, moved_bytes, detail).
# flops is None where a FLOPs model is meaningless (radix top-k).
# ---------------------------------------------------------------------------


def build_indexer_fwd(args):
    """Dense indexer scores, ratio-causal: valid k positions for query q are
    k < (q+1)/ratio, i.e. ~T^2/(2*ratio) scored positions per batch per head."""
    from cudnn import DSA

    t, h, d, ratio = args.seqlen, args.heads, 128, args.ratio
    s_k = t // ratio
    q = torch.randn(1, t, h, d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, s_k, 1, d, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(1, t, h, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(1, t, s_k, dtype=torch.float32, device="cuda")

    def run():
        DSA.indexer_forward_wrapper(q, k, w, ratio=ratio, qhead_per_kv_head=h, out=out)

    flops = h * 128 * t * t // ratio  # 2*h*d * T^2/(2*ratio)
    moved = t * h * d * _BF16 + s_k * d * _BF16 + t * h * _BF16 + t * s_k * _FP32
    return run, flops, moved, f"indexer_forward_wrapper dense s_k=T/{ratio} out=T^2/{ratio} fp32"


def build_indexer_fwd_topk(args):
    """Fused compressed-score + top-K + softmax (production indexer, SM100)."""
    from cudnn import DSA

    t, h, d, ratio, top_k = args.seqlen, args.heads, 128, args.ratio, args.top_k
    s_k = t // ratio
    q = torch.randn(1, t, h, d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, s_k, 1, d, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(1, t, h, dtype=torch.bfloat16, device="cuda").abs_().mul_(0.1)
    cand = torch.empty(DSA.compress_topk_cand_buffer_size(1, t, s_k, ratio), dtype=torch.float32, device="cuda")
    out_indices = torch.empty(1, t, top_k, dtype=torch.int32, device="cuda")
    out_logits = torch.empty(1, t, top_k, dtype=torch.float32, device="cuda")
    out_softmax = torch.empty(1, t, top_k, dtype=torch.float32, device="cuda")

    def run():
        DSA.indexer_forward_top_k_wrapper(
            q, k, w, top_k, ratio=ratio, cand_buffer=cand, out_indices=out_indices, out_logits=out_logits, softmax_out=out_softmax
        )

    flops = h * 128 * t * t // ratio  # score GEMM; selection overhead not modeled
    moved = t * h * d * _BF16 + s_k * d * _BF16 + t * h * _BF16 + 3 * t * top_k * _FP32
    return run, flops, moved, f"indexer_forward_top_k_wrapper fused compress+topk (cand-buffer traffic internal, TB/s is a lower bound)"


def build_topk(args):
    """Standalone radix top-K over an (n_rows, num_cols) fp32 score matrix."""
    from cudnn import DSA

    t, top_k = args.seqlen, args.top_k
    num_cols = t // args.ratio
    input_values = torch.randn(t, num_cols, dtype=torch.float32, device="cuda")
    seq_lens = torch.full((t,), num_cols, dtype=torch.int32, device="cuda")
    out_indices = torch.empty(t, top_k, dtype=torch.int32, device="cuda")
    out_values = torch.empty(t, top_k, dtype=torch.float32, device="cuda")

    def run():
        DSA.indexer_top_k_wrapper(input_values, seq_lens, top_k, next_n=1, return_val=True)

    moved = t * num_cols * _FP32 + t * top_k * (_INT32 + _FP32)
    return run, None, moved, f"indexer_top_k_wrapper radix topk num_cols=T/{args.ratio} (comparison-bound, no FLOPs model)"


def build_index_score_recompute(args):
    """Training-loss predict scores: re-score the selected positions (indexer side)."""
    from cudnn import DSA

    t, h, d, top_k = args.seqlen, args.heads, 128, args.top_k
    s_kv = t
    q = torch.randn(1, t, h, d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, s_kv, d, dtype=torch.bfloat16, device="cuda")
    weights = torch.randn(1, t, h, dtype=torch.bfloat16, device="cuda")
    topk_indices = make_topk_indices(1, t, s_kv, top_k, "cuda")
    topk_length = torch.full((1, t), top_k, dtype=torch.int32, device="cuda")
    out = torch.empty(1, t, top_k, dtype=torch.float32, device="cuda")

    def run():
        DSA.sparse_indexer_score_recompute_wrapper(q, k, weights, topk_indices, qhead_per_kv_head=h, topk_length=topk_length, out=out)

    flops = 2 * t * h * top_k * d
    moved = t * h * d * _BF16 + t * top_k * d * _BF16 + t * h * _BF16 + t * top_k * _INT32 + t * top_k * _FP32
    return run, flops, moved, "sparse_indexer_score_recompute_wrapper (predict)"


def build_attn_score_recompute(args):
    """Training-loss target scores: re-score the selected positions (attention side, d=512)."""
    from cudnn import DSA

    t, h, d, top_k = args.seqlen, args.heads, 512, args.top_k
    s_kv = t
    q = torch.randn(1, t, h, d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, s_kv, d, dtype=torch.bfloat16, device="cuda")
    lse = torch.randn(1, t, h, dtype=torch.float32, device="cuda")
    topk_indices = make_topk_indices(1, t, s_kv, top_k, "cuda")
    topk_length = torch.full((1, t), top_k, dtype=torch.int32, device="cuda")
    out = torch.empty(1, t, top_k, dtype=torch.float32, device="cuda")

    def run():
        DSA.sparse_attn_score_recompute_wrapper(q, k, lse, topk_indices, 1.0 / math.sqrt(d), qhead_per_kv_head=h, topk_length=topk_length, out=out)

    flops = 2 * t * h * top_k * d
    moved = t * h * d * _BF16 + t * top_k * d * _BF16 + t * h * _FP32 + t * top_k * _INT32 + t * top_k * _FP32
    return run, flops, moved, "sparse_attn_score_recompute_wrapper (target, d=512)"


def build_indexer_bwd(args):
    """Indexer backward. attn_score is consumed in place, so two score buffers
    alternate across iterations (values are irrelevant to the timed work)."""
    from cudnn import DSA

    t, h, d, ratio, top_k = args.seqlen, args.heads, 128, args.ratio, args.top_k
    s_k = t // ratio
    index_q = torch.randn(1, t, h, d, dtype=torch.bfloat16, device="cuda")
    weights = torch.randn(1, t, h, dtype=torch.bfloat16, device="cuda")
    index_k = torch.randn(1, s_k, d, dtype=torch.bfloat16, device="cuda")
    topk_indices = make_topk_indices(1, t, s_k, top_k, "cuda")
    index_score = torch.softmax(torch.randn(1, t, top_k, device="cuda"), dim=-1).contiguous()
    attn_scores = [torch.softmax(torch.randn(1, t, top_k, device="cuda"), dim=-1).contiguous() for _ in range(2)]
    grad_loss = torch.ones((), dtype=torch.float32, device="cuda")
    loss_coeff = float(t)
    state = {"i": 0}

    def run():
        buf = attn_scores[state["i"] % 2]
        state["i"] += 1
        DSA.indexer_backward_wrapper(index_q, weights, index_k, buf, index_score, topk_indices, grad_loss, sm_scale=1.0, loss_coeff=loss_coeff, block_I=128)

    flops = 4 * t * h * top_k * d  # d_index_q + d_index_k GEMMs (approximate)
    moved = (
        2 * t * h * d * _BF16  # read index_q, write d_index_q
        + t * top_k * d * _BF16  # gathered index_k rows
        + 2 * t * h * _BF16  # read weights, write d_weights
        + 3 * t * top_k * _FP32  # attn_score + index_score + topk_indices-as-int... (indices int32)
        + s_k * d * _BF16  # d_index_k
    )
    return run, flops, moved, "indexer_backward_wrapper (attn_score scratch alternates 2 buffers)"


def build_local_to_global(args):
    """Local top-K indices -> global index space (pure index-layout transform)."""
    from cudnn import DSA

    t, top_k = args.seqlen, args.top_k
    seqlen_k = t // args.ratio
    local_indices = torch.randint(0, seqlen_k, (1, t, top_k), dtype=torch.int32, device="cuda")

    def run():
        DSA.local_to_global_wrapper(local_indices, seqlen_k)

    moved = 2 * t * top_k * _INT32  # read local, write global
    return run, None, moved, "local_to_global_wrapper BSHD int32 (data-movement only, no FLOPs model)"


def build_compactify(args):
    """Pack valid indices row-wise + emit topk_length (10% of entries invalid/-1)."""
    from cudnn import DSA

    t, top_k = args.seqlen, args.top_k
    seqlen_k = t // args.ratio
    indices = torch.randint(0, seqlen_k, (1, t, top_k), dtype=torch.int32, device="cuda")
    indices[torch.rand(1, t, top_k, device="cuda") < 0.1] = -1

    def run():
        DSA.compactify_wrapper(indices)

    moved = 2 * t * top_k * _INT32 + t * _INT32  # read idxs, write idxs + topk_length
    return run, None, moved, "compactify_wrapper row-pack + topk_length (data-movement only, no FLOPs model)"


def build_dense_index_score_recompute(args):
    """Dense indexer score recompute over the full (compressed) KV — the non-sparse
    counterpart of index_score_recompute. Causal: ~half the dense GEMM positions are
    valid; invalid entries are filled, so the full (T, s_k) fp32 matrix is written."""
    from cudnn import DSA

    t, h, d, ratio = args.seqlen, args.heads, 128, args.ratio
    s_k = t // ratio
    q = torch.randn(1, t, h, d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, s_k, 1, d, dtype=torch.bfloat16, device="cuda")
    weights = torch.randn(1, t, h, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(1, t, s_k, dtype=torch.float32, device="cuda")
    denom_out = torch.empty(1, t, dtype=torch.float32, device="cuda")

    def run():
        DSA.dense_indexer_score_recompute_wrapper(q, k, weights, qhead_per_kv_head=h, out=out, denom_out=denom_out, ratio=ratio)

    flops = h * d * t * s_k  # 2*h*d MACs over the ~t*s_k/2 causal-valid positions
    moved = t * h * d * _BF16 + s_k * d * _BF16 + t * h * _BF16 + t * s_k * _FP32 + t * _FP32
    return run, flops, moved, f"dense_indexer_score_recompute_wrapper full-KV causal s_k=T/{ratio} (dense counterpart, fills invalid entries)"


def build_dense_attn_score_recompute(args):
    """Dense attention score recompute over the full uncompressed KV (d=512) — the
    non-sparse counterpart of attn_score_recompute. Causal, full (T, T) fp32 out."""
    from cudnn import DSA

    t, h, d = args.seqlen, args.heads, 512
    q = torch.randn(1, t, h, d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, t, 1, d, dtype=torch.bfloat16, device="cuda")
    lse = torch.randn(1, t, h, dtype=torch.float32, device="cuda")
    out = torch.empty(1, t, t, dtype=torch.float32, device="cuda")
    denom_out = torch.empty(1, t, dtype=torch.float32, device="cuda")

    def run():
        DSA.dense_attn_score_recompute_wrapper(q, k, lse, 1.0 / math.sqrt(d), qhead_per_kv_head=h, out=out, denom_out=denom_out)

    flops = h * d * t * t  # 2*h*d MACs over the ~t^2/2 causal-valid positions
    moved = t * h * d * _BF16 + t * d * _BF16 + t * h * _FP32 + t * t * _FP32 + t * _FP32
    return run, flops, moved, "dense_attn_score_recompute_wrapper full-KV causal (dense counterpart, d=512, T^2 fp32 out)"


def build_dense_indexer_bwd(args):
    """Dense full-KV indexer backward. attn_score/index_score are consumed in place,
    so two buffer pairs alternate across iterations (values irrelevant to timing)."""
    from cudnn import DSA

    t, h, d, ratio = args.seqlen, args.heads, 128, args.ratio
    s_k = t // ratio
    index_q = torch.randn(1, t, h, d, dtype=torch.bfloat16, device="cuda")
    weights = torch.randn(1, t, h, dtype=torch.bfloat16, device="cuda")
    index_k = torch.randn(1, s_k, d, dtype=torch.bfloat16, device="cuda")
    attn_scores = [torch.randn(1, t, s_k, dtype=torch.float32, device="cuda") for _ in range(2)]
    index_scores = [torch.randn(1, t, s_k, dtype=torch.float32, device="cuda") for _ in range(2)]
    attn_l1norm = torch.randn(1, t, dtype=torch.float32, device="cuda")
    index_lse = torch.randn(1, t, dtype=torch.float32, device="cuda")
    grad_loss = torch.ones((), dtype=torch.float32, device="cuda")
    state = {"i": 0}

    def run():
        i = state["i"] % 2
        state["i"] += 1
        DSA.dense_indexer_backward_wrapper(
            index_q,
            weights,
            index_k,
            attn_scores[i],
            attn_l1norm,
            index_scores[i],
            index_lse,
            grad_loss,
            sm_scale=1.0,
            loss_coeff=float(t),
            block_I=128,
            ratio=ratio,
        )

    flops = 2 * h * d * t * s_k  # dq + dk GEMMs over the causal-valid positions (approximate)
    moved = (
        2 * t * h * d * _BF16  # read index_q, write d_index_q
        + 2 * s_k * d * _BF16  # read index_k, write d_index_k
        + 2 * t * h * _BF16  # read weights, write d_weights
        + 4 * t * s_k * _FP32  # attn_score + index_score, read + written in place
        + 2 * t * _FP32  # attn_l1norm + index_lse
    )
    return run, flops, moved, "dense_indexer_backward_wrapper full-KV (score buffers alternate 2 pairs, consumed in place)"


BUILDERS = {
    "indexer_fwd": build_indexer_fwd,
    "indexer_fwd_topk": build_indexer_fwd_topk,
    "topk": build_topk,
    "index_score_recompute": build_index_score_recompute,
    "attn_score_recompute": build_attn_score_recompute,
    "indexer_bwd": build_indexer_bwd,
    "local_to_global": build_local_to_global,
    "compactify": build_compactify,
    "dense_index_score_recompute": build_dense_index_score_recompute,
    "dense_attn_score_recompute": build_dense_attn_score_recompute,
    "dense_indexer_bwd": build_dense_indexer_bwd,
}


def unsupported(reason: str):
    print(f"UNSUPPORTED,{reason.replace(',', ';')}")
    sys.exit(UNSUPPORTED_EXIT_CODE)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--op", required=True, choices=sorted(BUILDERS))
    p.add_argument("--seqlen", type=int, required=True, help="T: number of query tokens (b=1)")
    p.add_argument("--heads", type=int, default=64, help="query heads per the single shared KV head (MQA)")
    p.add_argument("--top_k", type=int, default=640, help="selected entries per query (dsv4_flash width)")
    p.add_argument("--ratio", type=int, default=4, help="indexer compression ratio: s_k = T / ratio")
    p.add_argument("--num_iterations", type=int, default=20)
    p.add_argument("--num_warmup_iterations", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    if not torch.cuda.is_available():
        unsupported("CUDA not available")
    if args.seqlen % args.ratio:
        unsupported(f"seqlen ({args.seqlen}) must be a multiple of ratio ({args.ratio})")
    major, minor = torch.cuda.get_device_capability()
    if args.op == "indexer_fwd_topk" and major != 10:
        unsupported(f"compressed indexer forward top-k requires SM100, found SM{major}{minor}")
    if major not in (9, 10):
        unsupported(f"DSA auxiliary kernels require SM90/SM100, found SM{major}{minor}")
    try:
        import cudnn
        from cudnn import DSA  # noqa: F401
    except (ImportError, AttributeError) as e:
        unsupported(f"cudnn not available: {e}")

    torch.manual_seed(args.seed)
    try:
        run, flops, moved, detail = BUILDERS[args.op](args)
        run()
        torch.cuda.synchronize()
    except (ValueError, RuntimeError) as e:
        unsupported(str(e))

    ms = time_fn(run, args.num_warmup_iterations, args.num_iterations)
    tflops = flops / (ms * 1e-3) / 1e12 if flops else 0.0
    tb_per_sec = moved / (ms * 1e-3) / 1e12
    gpu = torch.cuda.get_device_name().replace(",", ";")
    print(f"RESULT,{ms:.6f},{tflops:.3f},{tb_per_sec:.3f},{gpu},{cudnn.__version__},{detail.replace(',', ';')}")


if __name__ == "__main__":
    main()
