# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DSA (DeepSeek Sparse Attention) adapter.

Reuses the existing DSA benchmark harness end to end: cases are expressed as
``benchmark.dsa.config_types`` presets and executed through
``benchmark.dsa.runner.DsaBenchmarkRunner.run_single``, which subprocesses
``benchmark/dsa/benchmark_single_dsa.py`` (clean CUDA context and CuTe DSL
compile state per case, UNSUPPORTED exit code mapped to a skipped row).

Timing (measured by the worker): CUDA events around each steady-state
``SparseAttentionForward.execute`` / ``SparseAttentionBackward.execute``
call after a 256 MiB L2 flush, median over iterations. TFLOPS count only the
gathered top-k rows (see benchmark/dsa/README.md for the FLOPs model).

The worker does not report bandwidth, so this adapter derives TB/s from the
measured time and an idealized DRAM traffic model that counts every tensor
the pass touches exactly once (the gathered KV rows once per query — DSA is
MQA, so all heads share them):

- fwd: read q, gathered KV, topk_idxs; write out, max_logits, lse (+
  lse_indexer when indexer_topk > 0).
- bwd: read q, gathered KV, topk_idxs, out, dout, lse; write dq, dkv, d_sink.

Internal recompute/accumulate traffic between the backward's preprocess,
backward, and convert kernels is not modeled, so bwd TB/s is a lower bound.
"""

from typing import List, Sequence

from benchmark.dsa.config_types import DsaBenchmarkConfig
from benchmark.dsa.runner import DsaBenchmarkRunner

from ..shapes import DSA_DATA_TYPE, DSA_MODELS, DSA_SEQLENS
from ..suite_types import STATUS_FAILED, STATUS_OK, STATUS_SKIPPED, CaseResult, SuiteCase
from .base import ComponentAdapter

_ELT_BYTES = {"bfloat16": 2, "float16": 2}
_INT32 = 4
_FP32 = 4


def _bytes_fwd(p) -> int:
    elt = _ELT_BYTES[p["data_type"]]
    s_q, heads, topk = p["q_seqlen"], p["num_q_heads"], p["topk"]
    d_qk, d_v = p["head_dim_qk"], p["head_dim_vo"]
    total = s_q * heads * d_qk * elt  # q
    total += s_q * topk * d_qk * elt  # gathered KV rows (shared across heads)
    total += s_q * topk * _INT32  # topk_idxs
    total += s_q * heads * d_v * elt  # out
    total += s_q * heads * _FP32 * 2  # max_logits + lse
    if p["indexer_topk"]:
        total += s_q * heads * _FP32  # lse_indexer
    return total


def _bytes_bwd(p) -> int:
    elt = _ELT_BYTES[p["data_type"]]
    s_q, s_kv, heads, topk = p["q_seqlen"], p["kv_seqlen"], p["num_q_heads"], p["topk"]
    d_qk, d_v = p["head_dim_qk"], p["head_dim_vo"]
    total = s_q * heads * d_qk * elt * 2  # read q, write dq
    total += s_q * topk * d_qk * elt  # gathered KV rows (shared across heads)
    total += s_q * topk * _INT32  # topk_idxs
    total += s_q * heads * d_v * elt * 2  # out + dout
    total += s_q * heads * _FP32  # lse
    total += s_kv * d_qk * elt  # dkv (zeroed, then accumulated)
    total += heads * _FP32  # d_sink
    return total


class DsaAdapter(ComponentAdapter):
    name = "dsa"

    def cases(self, shape_sets: Sequence[str], passes: Sequence[str]) -> List[SuiteCase]:
        cases = []
        for shape_set in shape_sets:
            for model in DSA_MODELS:
                for s_q, s_kv in DSA_SEQLENS[shape_set]:
                    for profile_pass in passes:
                        label = (
                            f"{model.name} | seq={s_q}x{s_kv} | H{model.num_q_heads} "
                            f"d={model.head_dim_qk}/{model.head_dim_vo} K={model.topk} idx={model.indexer_topk} | {profile_pass}"
                        )
                        cases.append(
                            SuiteCase(
                                component=self.name,
                                case_tag=f"dsa_{model.name}_{shape_set}_s{s_q}_{profile_pass}",
                                shape_set=shape_set,
                                profile_pass=profile_pass,
                                params={
                                    "model": model.name,
                                    "q_seqlen": s_q,
                                    "kv_seqlen": s_kv,
                                    "num_q_heads": model.num_q_heads,
                                    "head_dim_qk": model.head_dim_qk,
                                    "head_dim_vo": model.head_dim_vo,
                                    "topk": model.topk,
                                    "indexer_topk": model.indexer_topk,
                                    "has_sink": model.has_sink,
                                    "data_type": DSA_DATA_TYPE,
                                },
                                label=label,
                            )
                        )
        return cases

    def run(self, case: SuiteCase, num_iterations: int, num_warmup_iterations: int) -> CaseResult:
        model = next(m for m in DSA_MODELS if m.name == case.params["model"])
        config = DsaBenchmarkConfig(
            name="kernel_suite",
            models=[model],
            seqlens=[(case.params["q_seqlen"], case.params["kv_seqlen"])],
            profile_pass=case.profile_pass,
            num_iterations=num_iterations,
            num_warmup_iterations=num_warmup_iterations,
        )
        case_dict = {
            "config": config,
            "model": model,
            "q_seqlen": case.params["q_seqlen"],
            "kv_seqlen": case.params["kv_seqlen"],
            "backend": "cudnn",
            "data_type": DSA_DATA_TYPE,
            "profile_pass": case.profile_pass,
            "deterministic_bwd": False,
        }
        r = DsaBenchmarkRunner(verbose=False).run_single(case_dict)
        if r.success:
            moved = (_bytes_fwd if case.profile_pass == "fwd" else _bytes_bwd)(case.params)
            tb_per_sec = moved / (r.time_ms * 1e-3) / 1e12
            return CaseResult(
                case=case,
                status=STATUS_OK,
                time_ms=r.time_ms,
                tflops=r.tflops,
                tb_per_sec=tb_per_sec,
                peak_mma_tflops=r.peak_mma_tflops,
                gpu_name=r.gpu_name,
                cudnn_version=r.cudnn_version,
                detail=r.backend_detail,
            )
        status = STATUS_SKIPPED if r.skipped else STATUS_FAILED
        return CaseResult(case=case, status=status, error_message=r.error_message)
