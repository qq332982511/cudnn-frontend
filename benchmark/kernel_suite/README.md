# Kernel Benchmark Suite (DSA / CSA / GDN / KDA / QSA)

One-click benchmark of the cuDNN Frontend component kernels on a B200
(SM100) GPU, over **regular** and **large** shapes, forward **and** backward.
It reuses the existing per-component harnesses — every case runs in its own
subprocess (clean CUDA context and CuTe DSL compile state, independent
failures) — and records everything into a CSV plus a Markdown report.

| Component | Kernel(s) under test | Harness reused | Adapter |
|---|---|---|---|
| DSA | `cudnn.DSA.SparseAttentionForward` / `SparseAttentionBackward` (CuTe DSL, fwd SM100-only, bwd SM90/SM100) | `benchmark/dsa/` | `components/dsa.py` |
| DSA aux | indexer (`indexer_forward_wrapper`, fused `indexer_forward_top_k_wrapper`), radix top-K (`indexer_top_k_wrapper`), sparse score recompute (`sparse_indexer_/sparse_attn_score_recompute_wrapper`), indexer backward (`indexer_backward_wrapper`), dense recompute family (`dense_index_score_recompute` / `dense_attn_score_recompute` / `dense_indexer_bwd`), index-layout helpers (`local_to_global`, `compactify`) | `workers/dsa_aux_worker.py` | `components/dsa_aux.py` |
| CSA | `cudnn.csa.csa_compressor_forward/backward_wrapper` (gated-softmax pooling; generic kernels for ratio {2, 4}, dedicated r128 family for the 128-token compress) | `benchmark/csa/` conventions | `components/csa.py` |
| GDN | `cudnn.linear_attention.ops.gated_delta_net` (FROST engine on SM100) | `benchmark/linear_attention/` | `components/linear_attention.py` |
| KDA | `cudnn.linear_attention.ops.kimi_delta_attention` (FROST engine on SM100) | `benchmark/linear_attention/` | `components/linear_attention.py` |
| GDN2 / GDP | `gated_delta_net` two-chunk-state variant / `gated_delta_product` attention variants (suite timing only, no breakdown) | `benchmark/linear_attention/` | `components/linear_attention.py` |
| LA aux | standalone summary ops: `gdn_summary` / `gdn_summary_bwd` / `kda_summary` / `kda_summary_bwd` (the ops the fused GDN/KDA graphs embed) | `workers/la_aux_worker.py` | `components/la_aux.py` |
| QSA | Qwen sparse attention — **external repository**, adapter stub | — | `components/qsa.py` |

For per-kernel granularity *inside* the fused ops (the auxiliary kernels the
op-level timing rolls up — l2norm, chunk scan, state passing, index
padding/convert, ...), see [Per-kernel breakdown](#per-kernel-breakdown).

## Requirements

- NVIDIA B200 (SM100). DSA forward is SM100-family only; GDN/KDA run on the
  FROST engines there. (DSA backward also works on SM90; GDN/KDA fall back to
  the cuTile engines on non-SM100 archs, which additionally need CUDA 13.3+
  and `cuda-tile`.)
- PyTorch with CUDA.
- This repository's Python package with the CuTe DSL dependencies:
  `pip install -e .` (needs `nvidia-cutlass-dsl >= 4.7.0` for the FROST
  kernels).
- `pynvml` (optional) for the dense-MMA peak used by the %SOL column.

## Quick start

```bash
# from the repository root
python -m benchmark.kernel_suite.runner                 # all components (incl. dsa_aux), all shapes, fwd+bwd
python -m benchmark.kernel_suite.runner --dry-run       # list the cases without a GPU
python -m benchmark.kernel_suite.runner --components dsa,kda --shape-set regular
python -m benchmark.kernel_suite.runner --components dsa_aux   # DSA auxiliary kernels only
python -m benchmark.kernel_suite.runner --components csa       # CSA compressor sweep (ratio 2/4/128)
python -m benchmark.kernel_suite.runner --passes fwd --num-iterations 50
python -m benchmark.kernel_suite.runner --qsa-repo /path/to/qwen-sparse-attention
python -m benchmark.kernel_suite.breakdown              # per-kernel profiler breakdown of every case
```

Output lands in `results/<gpu>/` (`--output-dir` to override,
`--gpu-label` to override the auto-detected `<gpu>` slug):

```
results/b200/
    suite_<timestamp>.csv   # one row per case, all fields
    suite_<timestamp>.md    # environment header + per-component tables
```

Exit code is 1 if any case failed. Skipped cases (an unserved shape/arch,
or QSA before its repo is integrated) do not fail the run.

## Shapes

All shapes are defined in `shapes.py` (plain data, edit to taste).

### DSA (flat MQA: `q (s_q, H, d_qk)`, shared `kv (s_kv, d_qk)`, `topk_idxs (s_q, K)`)

Model presets are imported from `benchmark/dsa/configs/`; `s_q == s_kv` and
every query gathers the full top-k (per-token upper bound). bf16, per-head
sink + full-length `topk_length` where the model has them, non-deterministic
backward.

| model | H | d_qk | d_vo | topk | indexer_topk | sink |
|---|---:|---:|---:|---:|---:|---|
| dsv4_flash | 64 | 512 | 512 | 640 | 512 | yes |
| dsv4_pro | 128 | 512 | 512 | 1152 | 1024 | yes |
| glm53 | 64 | 576 | 512 | 2048 | 0 | no |

| shape set | (s_q, s_kv) |
|---|---|
| regular | 4096², 8192² |
| large | 65536², 131072² |

3 models × 4 shapes × fwd+bwd = 24 cases. Note: large-shape setup (random
top-k index generation, chunked reference forward for the backward inputs)
takes minutes per case before timing starts.

### GDN (`--variant gdn --la_backend cudnn`)

Qwen3-Next-style geometry: Hq=16, Hkv=32, d=128, bf16, in-kernel Q/K L2
normalization on (harness default), reference validation skipped (needs the
external FLA package; correctness is covered by `test/python/linear_attention`).

### KDA (`--variant kda --la_backend cudnn`)

Hq=Hkv=96, d=128, bf16 (the harness's per-variant default geometry), safe-gate
defaults of the harness.

| shape set | (batch, seqlen) — GDN and KDA |
|---|---|
| regular | (4, 4096), (4, 8192) |
| large | (1, 65536), (1, 131072) |

4 shapes × fwd+bwd = 8 cases per variant.

Two more attention variants are timed on the same shape grid (suite timing
only — they share the GDN/KDA fused-graph structure, so their per-kernel
makeup is covered by the GDN/KDA breakdowns): **GDN2** (`--variant gdn2`,
the two-chunk-state gated_delta_net variant; harness defaults Hq=16, Hkv=8)
and **GDP** (`--variant gdp`, gated delta product attention; harness
defaults Hq=Hkv=40).

### DSA auxiliary kernels (component `dsa_aux`)

Every auxiliary op of the DSA stack, benchmarked individually
(`workers/dsa_aux_worker.py`, CUDA events + L2 flush, median). Production
shape convention with T tokens: the indexer scores `s_k = T / ratio`
compressed positions per query (ratio = 4), the attention side gathers
`top_k = 640` (dsv4_flash width); b = 1, bf16.

| op | what it is | pass | shapes (regular / large) |
|---|---|---|---|
| `indexer_fwd` | dense indexer scores `Σ_h ReLU(Q_h·K_hᵀ)·W_h`, ratio-causal mask, dense fp32 out | fwd | 4096, 8192 / 16384, 32768 (dense out caps large at 32k) |
| `indexer_fwd_topk` | fused compressed-score + radix top-K + softmax (production indexer, SM100-only) | fwd | 4096, 8192 / 65536, 131072 |
| `topk` | standalone radix top-K over an (T, T/ratio) fp32 score matrix | fwd | 4096, 8192 / 65536, 131072 |
| `index_score_recompute` | training-loss predict scores over the selected positions (indexer side, d=128) | fwd | 4096, 8192 / 65536, 131072 |
| `attn_score_recompute` | training-loss target scores over the selected positions (attention side, d=512) | fwd | 4096, 8192 / 65536, 131072 |
| `indexer_bwd` | indexer backward (d_index_q / d_weights / d_index_k) | bwd | 4096, 8192 / 65536, 131072 |
| `local_to_global` | per-segment local top-k indices → global indices (pure index transform) | fwd | 4096, 8192 / 65536, 131072 |
| `compactify` | remove padding from the top-k index list (pure index transform) | fwd | 4096, 8192 / 65536, 131072 |
| `dense_index_score_recompute` | dense (non-sparse) predict scores, full `(T, s_k)` fp32 matrix out | fwd | 4096, 8192 / 16384, 32768 (dense out caps large at 32k) |
| `dense_attn_score_recompute` | dense (non-sparse) target scores, full `(T, T)` fp32 matrix out — heaviest aux op | fwd | 4096, 8192 / 16384, 32768 |
| `dense_indexer_bwd` | dense indexer backward (no top-k sparsity) | bwd | 4096, 8192 / 16384, 32768 |

11 ops × 4 shapes = 44 cases. Metric models (`workers/dsa_aux_worker.py` has
the exact formulas): indexer FLOPs count the ratio-causal valid positions
(`H·128·T²/ratio`); score-recompute FLOPs are `2·T·H·topk·d`; `topk` is
comparison-bound and reports no TFLOPS; `local_to_global` / `compactify` are
pure data movement and report TB/s only; the dense family uses the same
dense-score FLOPs model as `indexer_fwd`. TB/s counts each tensor the op
touches once (the compressed path's internal cand-buffer traffic is not
modeled — TB/s is a lower bound).

### CSA fused compressor (component `csa`)

The `cudnn.csa` gated-softmax pooling kernels — every `ratio` tokens pooled
into one compressed token — swept over the full validated envelope
(`workers/csa_worker.py`, CUDA events + L2 flush, median):

| axis | values |
|---|---|
| ratio (tokens per compressed block) | 2, 4 (generic `compressor_sm100.py`), **128** (dedicated `compressor_sm100_r128.py`) |
| coff (window form) | 1 (own block, window = ratio), 2 (overlapping, window = 2·ratio — production) |
| head_dim d | 128, 512 (kv/score width = coff·d) |
| shape set | regular: 8192 tokens (1 segment); large: 131072 tokens (2×65536) |

3 ratios × 2 coffs × 2 head_dims × 2 passes × 2 shape sets = 48 cases.
FLOPs count the weighted-sum MACs (`2·T·coff·d` fwd, ~3× that bwd — softmax
exp/adds uncounted; pooling is bandwidth-bound, so read the TB/s column
first). TB/s counts kv + score + ape + out (fwd) / + grad_out + dkv + dscore
+ dAPE (bwd) exactly once. The wrappers allocate their output buffers per
call, so allocation is inside the timed region (same convention as
`benchmark/csa/bench_csa_compressor.py`).

### Linear-attention summary ops (component `la_aux`)

The standalone summary ops that the fused GDN/KDA graphs embed internally —
timed on their own so the summary cost is visible without the attention
kernels around it (`workers/la_aux_worker.py`, CUDA events + L2 flush,
median):

| op | what it is |
|---|---|
| `gdn_summary` | GDN per-chunk state summary (fwd: decayed chunk states + l2norm + tinv inside one op) |
| `gdn_summary_bwd` | GDN summary backward (state gradients + recompute) |
| `kda_summary` | KDA per-chunk summary (fwd) |
| `kda_summary_bwd` | KDA summary backward (bprop summary + recompute) |

2 variants × 2 ops × 4 shapes (same (batch, seqlen) grid as GDN/KDA) = 16
cases, same head geometry as the corresponding fused variant. FLOPs use an
approximate recurrence MAC model (noted in the `detail` column), so read the
TB/s column first.

## Metrics

- **time (ms)**: median over `--num-iterations` (default 20) after
  `--num-warmup-iterations` (default 5; kernel compilation happens in
  warmup). **The timing convention differs per component**: DSA, DSA-aux,
  CSA and LA-aux rows are CUDA-event wall time around each steady-state call
  after a 256 MiB L2 flush (includes the lightweight execution adapter and,
  for backward, the gradient zeroing the kernel performs); GDN/KDA (and the
  GDN2/GDP variants) rows are
  torch.profiler CUDA device time summed over the pass's kernels, median
  after dropping the first samples, L2 flushed between iterations. Compare
  components by TFLOPS, not raw ms.
- **TFLOPS**: DSA counts only the gathered top-k rows
  (`2 * s_q * H * topk * (d_qk + d_v)` fwd; `2 * s_q * H * topk * (3*d_qk + 2*d_v)`
  bwd), so numbers are independent of `s_kv`. GDN/KDA (and GDN2/GDP) use the
  chunked linear-attention FLOPs model of the harness (chunk 64 for GDN, 16
  for KDA). LA-aux uses an approximate recurrence MAC model (the summary ops
  are bandwidth-heavy, so their TB/s is the primary signal).
- **TB/s**: idealized DRAM bandwidth — bytes moved / measured time. GDN/KDA
  use the harness's own traffic model. DSA derives it from a model that
  counts every tensor the pass touches exactly once (the gathered KV rows
  once per query — DSA is MQA, so all heads share them): fwd reads
  q + gathered KV + topk_idxs and writes out + max_logits + lse
  (+ lse_indexer); bwd reads q + gathered KV + topk_idxs + out + dout + lse
  and writes dq + dkv + d_sink. Backward inter-kernel recompute traffic is
  not modeled, so DSA bwd TB/s is a lower bound.
- **%SOL**: achieved TFLOPS over the dense-MMA peak at the SM clock sampled
  via NVML during the case (DSA only; empty without `pynvml`).
- **status**: `ok` / `skipped` (the kernel's own support check declined —
  honest gap) / `failed`.

## Per-kernel breakdown

`breakdown.py` profiles every suite case with torch.profiler and reports each
CUDA kernel inside the op separately — this is where the auxiliary kernels
*within* the fused ops become visible (GDN/KDA: l2norm, tinv, summary,
prefill, state chain/prologue; DSA backward: preprocess / backward / convert;
etc.).

```bash
python -m benchmark.kernel_suite.breakdown                 # all in-repo components, all shapes
python -m benchmark.kernel_suite.breakdown --components gdn,kda --shape-set regular
python -m benchmark.kernel_suite.breakdown --dry-run
```

Output: `results/<gpu>/breakdown_<timestamp>.csv` (one row per
case × kernel: calls, µs/call, share %) and `.md` (per-case tables, top 15
kernels + an "other" roll-up). Each case is profiled in its own subprocess
(`workers/breakdown_worker.py`); kernel times are averaged over `--iters`
(default 3) profiled calls after warmup. Inputs match the timed cases
(DSA reuses the `benchmark/dsa` builders; GDN/KDA build THD inputs with the
harness's conventions; `dsa_aux` reuses the aux worker's builders).

## QSA integration (external repo)

QSA (Qwen sparse attention) is not part of this repository. The adapter
`components/qsa.py` is a stub: without a repo it records every QSA case as
`skipped`; with one it is the single integration point. To integrate:

1. Point the suite at the repo: `--qsa-repo /path/to/repo` or
   `export QSA_REPO_PATH=/path/to/repo`.
2. Implement `QsaAdapter.run`: import (or subprocess into) the repo, build
   inputs from `case.params` (`seqlen`, `shape_set`, `profile_pass`), and
   return `CaseResult(status="ok", time_ms=..., tflops=..., tb_per_sec=..., detail=...)` —
   every kernel in this suite reports both compute (TFLOPS) and DRAM
   bandwidth (TB/s); derive TB/s from a byte-traffic model like the DSA
   adapter's.
3. Use the DSA timing convention (CUDA events around each steady-state call
   after a 256 MiB L2 flush, median over iterations) so QSA numbers are
   comparable with the DSA rows; note any deviation in `detail`.
4. Adjust `QSA_SHAPES` in `shapes.py` to the geometry QSA actually serves.

## Results

### B200

Measured on an NVIDIA B200 (SM100, 183 GB), 2026-09-30: torch 2.14.0+cu130,
CUDA 13.0, cudnn_frontend 1.31.0, cudnn_backend 9.24, nvidia-cutlass-dsl
4.8.0, cuda-python 13.4.1. 64 ok / 4 skipped (QSA) / 0 failed out of 68
cases. Full artifacts: `results/b200/suite_20260930_041659.{csv,md}`.
The CSA sweep (48 cases, all ok) was a follow-up run on a second B200
instance with the same software stack:
`results/b200/suite_20260930_103206.{csv,md}` (timing) and
`results/b200/breakdown_20260930_103523.{csv,md}` (per-kernel).
The extended auxiliary coverage (dsa_aux 44 + la_aux 16 + gdn2 8 + gdp 8 =
76 cases, all ok) was a third run on a third instance, same stack:
`results/b200/suite_20260930_111730.{csv,md}` (timing) and
`results/b200/breakdown_20260930_112424.{csv,md}` (per-kernel, dsa_aux +
la_aux, 60 cases). Across all runs the suite is 168 cases: 164 measurable,
4 QSA skipped.

#### DSA (main sparse attention)

| case | pass | time (ms) | TFLOPS | TB/s | %SOL |
|---|---|---:|---:|---:|---:|
| dsv4_flash seq=4096² | fwd | 0.443 | 774.8 | 7.29 | 32.5 |
| dsv4_flash seq=4096² | bwd | 1.569 | 547.6 | 2.41 | 23.0 |
| dsv4_flash seq=8192² | fwd | 0.853 | 805.6 | 7.58 | 33.8 |
| dsv4_flash seq=8192² | bwd | 3.039 | 565.3 | 2.48 | 23.7 |
| dsv4_pro seq=4096² | fwd | 1.000 | 1236.4 | 5.93 | 51.9 |
| dsv4_pro seq=4096² | bwd | 4.228 | 731.4 | 1.66 | 30.7 |
| dsv4_pro seq=8192² | fwd | 1.979 | 1249.8 | 5.99 | 52.5 |
| dsv4_pro seq=8192² | bwd | 8.473 | 730.0 | 1.65 | 31.2 |
| glm53 seq=4096² | fwd | 1.242 | 940.7 | 8.27 | 39.5 |
| glm53 seq=4096² | bwd | 4.911 | 601.7 | 2.21 | 25.3 |
| glm53 seq=8192² | fwd | 2.475 | 944.0 | 8.30 | 39.6 |
| glm53 seq=8192² | bwd | 9.560 | 618.2 | 2.27 | 25.9 |
| dsv4_flash seq=65536² | fwd | 7.496 | 733.4 | 6.90 | 33.6 |
| dsv4_flash seq=65536² | bwd | 29.755 | 461.9 | 2.03 | 21.9 |
| dsv4_flash seq=131072² | fwd | 16.276 | 675.5 | 6.36 | 32.6 |
| dsv4_flash seq=131072² | bwd | 67.450 | 407.5 | 1.79 | 19.1 |
| dsv4_pro seq=65536² | fwd | 18.865 | 1049.1 | 5.03 | 52.9 |
| dsv4_pro seq=65536² | bwd | 80.896 | 611.6 | 1.39 | 29.5 |
| dsv4_pro seq=131072² | fwd | 40.829 | 969.5 | 4.65 | 53.1 |
| dsv4_pro seq=131072² | bwd | 168.771 | 586.3 | 1.33 | 28.7 |
| glm53 seq=65536² | fwd | 22.282 | 838.9 | 7.37 | 42.1 |
| glm53 seq=65536² | bwd | 100.167 | 472.0 | 1.73 | 22.4 |
| glm53 seq=131072² | fwd | 49.997 | 747.7 | 6.57 | 40.7 |
| glm53 seq=131072² | bwd | 229.104 | 412.7 | 1.51 | 19.2 |

#### DSA auxiliary ops (indexer / compress / topk / recompute)

Every compress path is timed standalone (this answers "does each DSA compress
configuration get its own perf numbers": yes — `indexer_fwd` is the dense
compress, `indexer_fwd_topk` the fused compress+topk, `topk` the standalone
radix top-k, the two `*_score_recompute` ops the predict/target recompute
kernels, and `indexer_bwd` the indexer gradient op; the second block covers
the index-layout helpers and the dense non-sparse family). Shapes: T tokens,
KV side s_k = T/4 (compress ratio 4), H=64 indexer heads, K=640 selected
indices where applicable. Full 44-case sweep re-measured in one run:
`results/b200/suite_20260930_111730.{csv,md}`.

| op | T | pass | time (ms) | TFLOPS | TB/s |
|---|---:|---|---:|---:|---:|
| indexer_fwd | 4096 | fwd | 0.060 | 568.7 | 1.40 |
| indexer_fwd | 8192 | fwd | 0.150 | 913.2 | 1.35 |
| indexer_fwd | 16384 | fwd | 0.474 | 1159.5 | 1.14 |
| indexer_fwd | 32768 | fwd | 1.774 | 1239.9 | 0.91 |
| indexer_fwd_topk | 4096 | fwd | 0.113 | 305.0 | 0.88 |
| indexer_fwd_topk | 8192 | fwd | 0.275 | 499.0 | 0.72 |
| indexer_fwd_topk | 65536 | fwd | 10.420 | 844.1 | 0.15 |
| indexer_fwd_topk | 131072 | fwd | 40.922 | 859.8 | 0.08 |
| topk | 4096 | fwd | 0.060 | — | 0.62 |
| topk | 8192 | fwd | 0.126 | — | 0.87 |
| topk | 65536 | fwd | 2.681 | — | 1.73 |
| topk | 131072 | fwd | 8.688 | — | 2.06 |
| index_score_recompute | 4096 | fwd | 0.081 | 531.6 | 9.40 |
| index_score_recompute | 8192 | fwd | 0.146 | 586.5 | 10.37 |
| index_score_recompute | 65536 | fwd | 1.056 | 650.9 | 11.51 |
| index_score_recompute | 131072 | fwd | 2.104 | 653.1 | 11.55 |
| attn_score_recompute | 4096 | fwd | 0.273 | 628.3 | 10.88 |
| attn_score_recompute | 8192 | fwd | 0.525 | 654.0 | 11.32 |
| attn_score_recompute | 65536 | fwd | 4.982 | 551.8 | 9.55 |
| attn_score_recompute | 131072 | fwd | 12.084 | 454.9 | 7.88 |
| indexer_bwd | 4096 | bwd | 0.455 | 188.9 | 1.84 |
| indexer_bwd | 8192 | bwd | 0.886 | 194.0 | 1.89 |
| indexer_bwd | 65536 | bwd | 6.876 | 199.9 | 1.95 |
| indexer_bwd | 131072 | bwd | 13.730 | 200.2 | 1.95 |
| local_to_global | 4096 | fwd | 0.015 | — | 1.36 |
| local_to_global | 8192 | fwd | 0.026 | — | 1.64 |
| local_to_global | 65536 | fwd | 0.155 | — | 2.17 |
| local_to_global | 131072 | fwd | 0.302 | — | 2.22 |
| compactify | 4096 | fwd | 0.024 | — | 0.89 |
| compactify | 8192 | fwd | 0.034 | — | 1.24 |
| compactify | 65536 | fwd | 0.191 | — | 1.75 |
| compactify | 131072 | fwd | 0.374 | — | 1.80 |
| dense_index_score_recompute | 4096 | fwd | 0.063 | 541.2 | 1.33 |
| dense_index_score_recompute | 8192 | fwd | 0.159 | 865.7 | 1.28 |
| dense_index_score_recompute | 16384 | fwd | 0.505 | 1089.0 | 1.07 |
| dense_index_score_recompute | 32768 | fwd | 1.873 | 1174.1 | 0.86 |
| dense_attn_score_recompute | 4096 | fwd | 0.368 | 1495.3 | 0.93 |
| dense_attn_score_recompute | 8192 | fwd | 1.422 | 1546.1 | 0.57 |
| dense_attn_score_recompute | 16384 | fwd | 6.318 | 1392.2 | 0.34 |
| dense_attn_score_recompute | 32768 | fwd | 27.594 | 1275.1 | 0.23 |
| dense_indexer_bwd | 4096 | bwd | 0.338 | 203.4 | 0.60 |
| dense_indexer_bwd | 8192 | bwd | 1.069 | 257.1 | 0.51 |
| dense_indexer_bwd | 16384 | bwd | 3.766 | 291.9 | 0.43 |
| dense_indexer_bwd | 32768 | bwd | 14.244 | 308.8 | 0.38 |

Notes: `topk` is comparison-bound (no FLOPs model). `indexer_fwd_topk` keeps
its candidate-buffer traffic internal, so its TB/s is a lower bound.
`indexer_bwd` alternates two attn_score scratch buffers; ~200 TFLOPS is its
steady state at every scale. `local_to_global` / `compactify` are pure
index-layout transforms (single kernel each, 100% in the breakdown — report
TB/s only). The dense recompute family writes the full `(T, s_k)` / `(T, T)`
fp32 matrix (causal-invalid entries filled), so its large set is capped at
32k tokens; `dense_attn_score_recompute` is the heaviest auxiliary op in the
stack (27.6 ms at T=32768, 1275 TFLOPS — the T² score matrix dominates).
`dense_indexer_bwd` is 95.8% GEMM + 3.9% score-grad in the breakdown.

#### CSA fused compressor (ratio tokens → 1 compressed token)

The compressor is pure bandwidth-bound pooling — read the TB/s columns first
(TFLOPS stay at 1–3 because softmax FLOPs are not counted). Per-kernel
breakdown confirms the fused design: **forward is exactly 1 kernel (100%)**;
backward is 1 compressor kernel (98.9–99.6%) + the dAPE zero-fill
(0.4–1.1%).

regular = 8192 tokens (1 segment):

| config | fwd ms | fwd TFLOPS | fwd TB/s | bwd ms | bwd TFLOPS | bwd TB/s |
|---|---:|---:|---:|---:|---:|---:|
| r2 coff1 d128 | 0.009 | 0.2 | 0.57 | 0.019 | 0.3 | 0.48 |
| r2 coff1 d512 | 0.015 | 0.5 | 1.36 | 0.024 | 1.1 | 1.60 |
| r2 coff2 d128 | 0.011 | 0.4 | 0.84 | 0.022 | 0.6 | 0.83 |
| r2 coff2 d512 | 0.019 | 0.9 | 1.94 | 0.036 | 1.4 | 1.99 |
| r4 coff1 d128 | 0.009 | 0.2 | 0.51 | 0.019 | 0.3 | 0.46 |
| r4 coff1 d512 | 0.013 | 0.6 | 1.42 | 0.021 | 1.2 | 1.66 |
| r4 coff2 d128 | 0.011 | 0.4 | 0.79 | 0.022 | 0.6 | 0.81 |
| r4 coff2 d512 | 0.019 | 0.9 | 1.83 | 0.032 | 1.6 | 2.18 |
| r128 coff1 d128 | 0.017 | 0.1 | 0.25 | 0.019 | 0.3 | 0.44 |
| r128 coff1 d512 | 0.024 | 0.4 | 0.73 | 0.028 | 0.9 | 1.23 |
| r128 coff2 d128 | 0.019 | 0.2 | 0.44 | 0.026 | 0.5 | 0.67 |
| r128 coff2 d512 | 0.028 | 0.6 | 1.24 | 0.043 | 1.2 | 1.59 |

large = 131072 tokens (2×65536 segments):

| config | fwd ms | fwd TFLOPS | fwd TB/s | bwd ms | bwd TFLOPS | bwd TB/s |
|---|---:|---:|---:|---:|---:|---:|
| r2 coff1 d128 | 0.042 | 0.8 | 2.00 | 0.097 | 1.0 | 1.55 |
| r2 coff1 d512 | 0.144 | 0.9 | 2.32 | 0.191 | 2.1 | 3.15 |
| r2 coff2 d128 | 0.054 | 1.2 | 2.78 | 0.110 | 1.8 | 2.60 |
| r2 coff2 d512 | 0.185 | 1.4 | 3.26 | 0.329 | 2.5 | 3.47 |
| r4 coff1 d128 | 0.030 | 1.1 | 2.54 | 0.067 | 1.5 | 2.14 |
| r4 coff1 d512 | 0.091 | 1.5 | 3.31 | 0.155 | 2.6 | 3.69 |
| r4 coff2 d128 | 0.050 | 1.3 | 2.84 | 0.093 | 2.2 | 2.97 |
| r4 coff2 d512 | 0.171 | 1.6 | 3.34 | 0.300 | 2.7 | 3.69 |
| r128 coff1 d128 | 0.034 | 1.0 | 1.99 | 0.073 | 1.4 | 1.85 |
| r128 coff1 d512 | 0.089 | 1.5 | 3.03 | 0.204 | 2.0 | 2.64 |
| r128 coff2 d128 | 0.056 | 1.2 | 2.39 | 0.112 | 1.8 | 2.41 |
| r128 coff2 d512 | 0.140 | 1.9 | 3.84 | 0.465 | 1.7 | 2.31 |

Observations: at 8k tokens every config is launch/allocation-bound
(<0.05 ms, ≤2 TB/s); the kernels only reach their bandwidth regime at 131k
tokens (2.0–3.8 TB/s of B200's ~8 TB/s HBM). The overlapping window (coff=2)
costs ~30–60% over own-block, roughly tracking the 2× window traffic. The
dedicated r128 family is on par with the generic kernels in bandwidth, with
one outlier: **r128 coff2 d512 backward is the slowest case** (0.465 ms,
2.31 TB/s — the staged-smem backward with fused per-chunk reductions), ~40%
slower than the equivalent r4/r2 backward.

#### GDN

| case | pass | time (ms) | TFLOPS | TB/s |
|---|---|---:|---:|---:|
| B4 T4096 | fwd | 0.183 | 423.0 | 2.23 |
| B4 T4096 | bwd | 0.811 | 286.0 | 1.75 |
| B4 T8192 | fwd | 0.341 | 453.0 | 2.38 |
| B4 T8192 | bwd | 1.539 | 301.0 | 1.85 |
| B1 T65536 | fwd | 1.162 | 266.0 | 1.40 |
| B1 T65536 | bwd | 3.875 | 239.0 | 1.47 |
| B1 T131072 | fwd | 2.220 | 279.0 | 1.47 |
| B1 T131072 | bwd | 7.551 | 246.0 | 1.51 |

#### KDA

| case | pass | time (ms) | TFLOPS | TB/s |
|---|---|---:|---:|---:|
| B4 T4096 | fwd | 0.732 | 185.0 | 2.76 |
| B4 T4096 | bwd | 5.178 | 78.0 | 2.18 |
| B4 T8192 | fwd | 1.421 | 190.0 | 2.84 |
| B4 T8192 | bwd | 10.241 | 79.0 | 2.20 |
| B1 T65536 | fwd | 3.875 | 140.0 | 2.08 |
| B1 T65536 | bwd | 27.646 | 59.0 | 1.63 |
| B1 T131072 | fwd | 7.644 | 142.0 | 2.11 |
| B1 T131072 | bwd | 55.136 | 59.0 | 1.64 |

#### LA aux (standalone summary ops)

From `results/b200/suite_20260930_111730.{csv,md}`. The summary ops scale
linearly in tokens and are bandwidth-lean — they sit at 0.8–2.0 TB/s while
the FLOPs (approximate recurrence model) climb with seqlen. KDA's summary is
~4–5× slower than GDN's at the same token count (H=96 vs H=16 heads, chunk
16 vs 64), and its backward is the slowest la_aux op (17.1 ms at B1
T131072).

| op | case | pass | time (ms) | TFLOPS | TB/s |
|---|---|---|---:|---:|---:|
| gdn_summary | B4 T4096 | fwd | 0.227 | 151.2 | 0.98 |
| gdn_summary | B4 T8192 | fwd | 0.385 | 178.6 | 1.11 |
| gdn_summary | B1 T65536 | fwd | 0.720 | 190.9 | 1.15 |
| gdn_summary | B1 T131072 | fwd | 1.322 | 207.9 | 1.25 |
| gdn_summary_bwd | B4 T4096 | bwd | 0.379 | 136.0 | 0.76 |
| gdn_summary_bwd | B4 T8192 | bwd | 0.633 | 162.9 | 0.89 |
| gdn_summary_bwd | B1 T65536 | bwd | 1.155 | 178.5 | 0.95 |
| gdn_summary_bwd | B1 T131072 | bwd | 2.150 | 191.7 | 1.02 |
| kda_summary | B4 T4096 | fwd | 1.006 | 102.5 | 1.66 |
| kda_summary | B4 T8192 | fwd | 1.880 | 109.7 | 1.75 |
| kda_summary | B1 T65536 | fwd | 3.393 | 121.5 | 1.91 |
| kda_summary | B1 T131072 | fwd | 6.608 | 124.8 | 1.96 |
| kda_summary_bwd | B4 T4096 | bwd | 2.513 | 61.5 | 0.82 |
| kda_summary_bwd | B4 T8192 | bwd | 4.805 | 64.4 | 0.85 |
| kda_summary_bwd | B1 T65536 | bwd | 8.577 | 72.1 | 0.94 |
| kda_summary_bwd | B1 T131072 | bwd | 17.051 | 72.5 | 0.95 |

#### GDN2 (two-chunk-state gated delta net)

From `results/b200/suite_20260930_111730.{csv,md}`. ~2× the GDN forward
FLOPs-model time per token at lower TFLOPS (smaller head count, Hkv=8); the
backward holds ~1.5–1.9 TB/s like GDN.

| case | pass | time (ms) | TFLOPS | TB/s |
|---|---|---:|---:|---:|
| B4 T4096 | fwd | 0.253 | 89.0 | 1.99 |
| B4 T4096 | bwd | 1.559 | 43.0 | 1.53 |
| B4 T8192 | fwd | 0.480 | 94.0 | 2.10 |
| B4 T8192 | bwd | 2.934 | 46.0 | 1.62 |
| B1 T65536 | fwd | 1.084 | 83.0 | 1.86 |
| B1 T65536 | bwd | 5.176 | 52.0 | 1.84 |
| B1 T131072 | fwd | 2.038 | 89.0 | 1.98 |
| B1 T131072 | bwd | 10.129 | 53.0 | 1.88 |

#### GDP (gated delta product attention)

From `results/b200/suite_20260930_111730.{csv,md}`. The most
compute-dense linear-attention variant here (Hq=Hkv=40): 265–294 TFLOPS
forward — above KDA (140–190) and below GDN's regular-shape peak (423–453)
— at a steady 1.2–1.6 TB/s.

| case | pass | time (ms) | TFLOPS | TB/s |
|---|---|---:|---:|---:|
| B4 T4096 | fwd | 1.012 | 287.0 | 1.34 |
| B4 T4096 | bwd | 3.832 | 227.0 | 1.45 |
| B4 T8192 | fwd | 1.975 | 294.0 | 1.37 |
| B4 T8192 | bwd | 7.481 | 233.0 | 1.49 |
| B1 T65536 | fwd | 4.377 | 265.0 | 1.24 |
| B1 T65536 | bwd | 14.352 | 242.0 | 1.55 |
| B1 T131072 | fwd | 8.603 | 270.0 | 1.26 |
| B1 T131072 | bwd | 28.534 | 244.0 | 1.56 |

#### QSA

| case | pass | time (ms) | TFLOPS | TB/s |
|---|---|---:|---:|---:|
| _(pending external repo — all cases recorded as skipped)_ | | | | |

#### Per-kernel breakdown highlights

Full artifacts: `results/b200/breakdown_20260930_043810.{csv,md}` (DSA /
dsa_aux / GDN / KDA, 64 cases), `results/b200/breakdown_20260930_103523.{csv,md}`
(CSA, 48 cases) and `results/b200/breakdown_20260930_112424.{csv,md}`
(extended dsa_aux + la_aux, 60 cases) — one table per case (top 15 kernels + "other" roll-up).
Shares below are % of
the case's profiler device time per call (averaged over 3 profiled iterations;
absolute ms differ slightly from the timed suite because profiling adds
overhead and the bwd pass includes its forward recompute).

- **CSA fwd**: exactly 1 kernel, 100% (`_compressor_fwd[_r128]_kernel`).
- **CSA bwd**: 1 compressor kernel (98.9–99.6%) + the fp32 dAPE buffer
  zero-fill (0.4–1.1%) — the fused 1+1 design, up from ~39 fwd / ~51 bwd
  kernel launches in the eager region it replaces.

- **DSA fwd**: one fused kernel — 100% `SparseAttentionForwardSm100Head64`
  (dsv4_flash / glm53) or `SparseAttentionForwardSm100Head128SmallTopk`
  (dsv4_pro). No auxiliary kernels on the forward path.
- **DSA bwd**: 98–99.8% in the backward kernel
  (`FlashAttentionDSABackwardSm100[H128TwoCta]`); the aux tail is `sum_OdO`
  (≤1.5%, grows with seqlen), `sum_dSink`, `convert_dkv`, and
  `clear_dKV_workspace` / `zero_init` (≤0.1% each).
- **DSA indexer_bwd** (aux op): 97.6% `IndexerBackwardSm100` GEMM + 2.4%
  `ScoreGradSm100`.
- **DSA indexer_fwd_topk** (aux op, T=131072): internally 76.4%
  `IndexerScoreUnifiedSm100` (compress GEMM) + 22.5% `CompressTopkStage2`
  + 0.9% `LocalToGlobalTopK` — about a quarter of the fused compress+topk op
  is top-k selection.
- **DSA dense_indexer_bwd** (aux op): 95.8% dense GEMM + 3.9%
  `ScoreGradSm100` — same shape as sparse `indexer_bwd`, minus the top-k.
- **DSA dense_attn_score_recompute** (aux op): 97.3% in one
  `DenseScoreRecomputeSm100` kernel.
- **DSA local_to_global / compactify** (aux ops): 100% a single kernel each
  — pure index transforms, no hidden work.
- **LA aux gdn_summary** (standalone op): 60.6% `GdnSummary` + 21.8% tinv +
  10.9% l2norm — matches its share inside the fused GDN graph.
- **LA aux kda_summary_bwd** (standalone op): 65.7% `KdaBpropSummary` +
  31.9% `KdaRecompute`.

| GDN fwd | B4 T8192 | B1 T131072 |
|---|---:|---:|
| gdn_prefill | 72.9% | 34.9% |
| gdn_summary | — | 36.3% |
| l2norm_qk | 22.5% | 13.6% |
| gdn_tinv | — | 13.5% |
| state chain / prologue / split-k | ~4.6% | ~1.7% |

| GDN bwd | B4 T8192 | B1 T131072 |
|---|---:|---:|
| gdn_bprop | 50.0% | 37.9% |
| gdn_summary | — | 15.5% |
| head_reduce | 13.8% | 10.6% |
| gdn_prefill (bwd recompute) | 12.3% | 7.5% |
| gdn_recompute | 10.0% | 7.4% |
| gdn_bprop_summary | — | 8.6% |
| l2norm_qk / tinv / misc | ~14% | ~12% |

| KDA fwd | B4 T8192 | B1 T131072 |
|---|---:|---:|
| kda_prefill | 98.8% | 94.0% |
| split_k_scan_channel (gate scan) | 0.4% | 5.4% |
| prologue / walk / plan | ~0.8% | ~0.6% |

| KDA bwd | B4 T8192 | B1 T131072 |
|---|---:|---:|
| kda_bprop | 62.8% | 62.6% |
| kda_recompute | 17.9% | 17.5% |
| kda_prefill (bwd recompute) | 12.1% | 11.5% |
| channel_gate_bwd (partial+finish) | 6.8% | 6.9% |
| gate scan / prologue | ~0.4% | ~1.4% |

Takeaways: DSA is essentially a single-kernel op both directions (aux kernels
are noise except `sum_OdO` at 128k). GDN backward spends **over half** its
device time in auxiliary kernels (summary / head_reduce / recompute / l2norm),
and GDN forward pays 13–23% for the in-kernel Q/K L2 norm alone. KDA is the
leanest: ~95–99% in the prefill kernel forward, and backward auxiliaries
(recompute + channel-gate) are a fixed ~37%.
