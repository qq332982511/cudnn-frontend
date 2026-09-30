# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shape presets for the kernel benchmark suite.

Two shape sets per component:

- ``regular`` — shapes in the range the existing per-component sweeps already
  cover (a few thousand tokens of sequence, small batch).
- ``large`` — long-context shapes (64k/128k tokens), the regime where kernel
  behavior at scale (occupancy, workspace, compile variants) matters.

DSA model presets are imported from ``benchmark/dsa/configs/`` so the suite
measures exactly the geometries the DSA sweep documents. GDN/KDA shapes
follow the model geometries those variants serve (Qwen3-Next-style GDN,
Kimi-style KDA) and the conventions of
``benchmark/linear_attention/benchmark_single_linear_attention.py``.

Everything in this module is plain data — no torch import — so ``--dry-run``
works on a GPU-less host.
"""

from benchmark.dsa.configs.deepseek_v4 import DSV4_FLASH, DSV4_PRO
from benchmark.dsa.configs.glm53 import GLM53

SHAPE_SETS = ("regular", "large")

# ---------------------------------------------------------------------------
# DSA (DeepSeek Sparse Attention): flat MQA, q (s_q, H, d_qk), one shared
# K=V record kv (s_kv, d_qk), topk_idxs (s_q, topk). s_q == s_kv, every query
# gathers the full top-k (per-token upper bound).
# ---------------------------------------------------------------------------

DSA_MODELS = [DSV4_FLASH, DSV4_PRO, GLM53]

DSA_SEQLENS = {
    "regular": [(4096, 4096), (8192, 8192)],
    "large": [(65536, 65536), (131072, 131072)],
}

# bf16, per-head sink + full-length topk_length where the model has them,
# non-deterministic backward — the production configuration.
DSA_DATA_TYPE = "bfloat16"

# ---------------------------------------------------------------------------
# GDN / KDA (linear attention): dense varlen packing, batch_size sequences of
# equal seqlen. Qwen3-Next-style GDN: 16 q/k heads, 32 v/gate heads, d=128.
# KDA follows the benchmark script's default geometry: 96 heads, d=128.
# ---------------------------------------------------------------------------

LINEAR_ATTENTION_SHAPES = {
    "regular": [
        {"batch_size": 4, "seqlen": 4096},
        {"batch_size": 4, "seqlen": 8192},
    ],
    "large": [
        {"batch_size": 1, "seqlen": 65536},
        {"batch_size": 1, "seqlen": 131072},
    ],
}

LINEAR_ATTENTION_VARIANTS = {
    "gdn": {"num_q_heads": 16, "num_kv_heads": 32, "head_dim": 128},
    "kda": {"num_q_heads": 96, "num_kv_heads": 96, "head_dim": 128},
    # gdn2 (channel-wise decay/erase/write gates) and gdp (Householder
    # updates) follow the harness's per-variant default head counts.
    "gdn2": {"num_q_heads": 16, "num_kv_heads": 8, "head_dim": 128},
    "gdp": {"num_q_heads": 40, "num_kv_heads": 40, "head_dim": 128},
}

LINEAR_ATTENTION_DATA_TYPE = "bfloat16"

# ---------------------------------------------------------------------------
# DSA auxiliary kernels (indexer / top-k / score recompute / indexer bwd).
# Shape convention follows production: with T tokens, the indexer scores
# s_k = T / ratio compressed positions per query and the attention side
# gathers top_k = 640 entries (dsv4_flash width). b = 1, bf16.
# indexer_fwd materializes a dense (T, T/ratio) fp32 score matrix, so its
# "large" set is capped at 32k tokens (4.3 GB of scores).
# ---------------------------------------------------------------------------

DSA_AUX_OPS = {
    # op -> (profile_pass, heads, default top_k)
    "indexer_fwd": ("fwd", 64, 0),
    "indexer_fwd_topk": ("fwd", 64, 640),
    "topk": ("fwd", 0, 640),
    "index_score_recompute": ("fwd", 64, 640),
    "attn_score_recompute": ("fwd", 64, 640),
    "indexer_bwd": ("bwd", 64, 640),
    "local_to_global": ("fwd", 0, 640),
    "compactify": ("fwd", 0, 640),
    "dense_index_score_recompute": ("fwd", 64, 0),
    "dense_attn_score_recompute": ("fwd", 64, 0),
    "dense_indexer_bwd": ("bwd", 64, 0),
}

DSA_AUX_SHAPES = {
    "indexer_fwd": {"regular": [4096, 8192], "large": [16384, 32768]},
    "indexer_fwd_topk": {"regular": [4096, 8192], "large": [65536, 131072]},
    "topk": {"regular": [4096, 8192], "large": [65536, 131072]},
    "index_score_recompute": {"regular": [4096, 8192], "large": [65536, 131072]},
    "attn_score_recompute": {"regular": [4096, 8192], "large": [65536, 131072]},
    "indexer_bwd": {"regular": [4096, 8192], "large": [65536, 131072]},
    "local_to_global": {"regular": [4096, 8192], "large": [65536, 131072]},
    "compactify": {"regular": [4096, 8192], "large": [65536, 131072]},
    # The dense recompute ops materialize the full (T, s_k) / (T, T) fp32 score
    # matrix, so their "large" set is capped at 32k tokens like indexer_fwd.
    "dense_index_score_recompute": {"regular": [4096, 8192], "large": [16384, 32768]},
    "dense_attn_score_recompute": {"regular": [4096, 8192], "large": [16384, 32768]},
    "dense_indexer_bwd": {"regular": [4096, 8192], "large": [16384, 32768]},
}

DSA_AUX_RATIO = 4

# ---------------------------------------------------------------------------
# Linear-attention auxiliary kernels (component ``la_aux``): the per-span
# summary ops used for context parallelism, standalone. Same THD input
# conventions and per-variant geometry as the main ops.
# ---------------------------------------------------------------------------

LA_AUX_OPS = {
    # op -> (variant, profile_pass)
    "gdn_summary": ("gdn", "fwd"),
    "gdn_summary_bwd": ("gdn", "bwd"),
    "kda_summary": ("kda", "fwd"),
    "kda_summary_bwd": ("kda", "bwd"),
}

# ---------------------------------------------------------------------------
# CSA fused compressor (gated-softmax pooling): every ``ratio`` tokens are
# pooled into one compressed token. ratio {2, 4} run on the generic kernels,
# 128 on the dedicated r128 family (the 128-token compress path); coff 1/2
# select the own-block / overlapping window forms. Token packs: regular = a
# single 8k segment (the Megatron-LM production bench shape), large = 128k
# tokens over two segments (long-context + multi-segment packing).
# ---------------------------------------------------------------------------

CSA_RATIOS = (2, 4, 128)
CSA_COFFS = (1, 2)
CSA_HEAD_DIMS = (128, 512)
CSA_PACKS = {
    "regular": [8192],
    "large": [65536, 65536],
}

# ---------------------------------------------------------------------------
# QSA (Qwen sparse attention): external repository, adapter stub. Placeholder
# shapes so the report shows the intended coverage until the adapter is
# implemented (see components/qsa.py and the README's QSA section).
# ---------------------------------------------------------------------------

QSA_SHAPES = {
    "regular": [{"seqlen": 8192}],
    "large": [{"seqlen": 65536}],
}

# Default measurement effort per case (component adapters may be overridden
# from the runner CLI).
NUM_ITERATIONS = 20
NUM_WARMUP_ITERATIONS = 5
