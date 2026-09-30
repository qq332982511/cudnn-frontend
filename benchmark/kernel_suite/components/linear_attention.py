# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GDN / KDA (linear attention) adapter.

Reuses ``benchmark/linear_attention/benchmark_single_linear_attention.py``
through its library entry point ``run_benchmark``, which subprocesses the
script (clean CUDA context and kernel compile state per case) with
``--la_backend cudnn --skip_ref --format_output`` and parses the CSV line.
On B200 (SM100) the cudnn backend routes to the FROST CuTe DSL engines.

Timing (measured by the worker): torch.profiler CUDA device time summed over
the pass's kernels, median over iterations after dropping the first samples,
with a 256 MiB L2 flush between iterations. This differs from the DSA
adapter's CUDA-event wall time — see the README's metrics section.

Reference validation is skipped (``skip_ref=True``): it needs the external
FLA package, and correctness is covered by ``test/python/linear_attention``.
"""

import importlib.util
from pathlib import Path
from typing import List, Sequence

from ..shapes import LINEAR_ATTENTION_DATA_TYPE, LINEAR_ATTENTION_SHAPES, LINEAR_ATTENTION_VARIANTS
from ..suite_types import STATUS_FAILED, STATUS_OK, CaseResult, SuiteCase
from .base import ComponentAdapter

_LA_SCRIPT = Path(__file__).resolve().parents[2] / "linear_attention" / "benchmark_single_linear_attention.py"


def _load_la_module():
    """Import the benchmark script as a module (its directory is not a package)."""
    spec = importlib.util.spec_from_file_location("benchmark_single_linear_attention", _LA_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # imports torch; called lazily from run()
    return module


class LinearAttentionAdapter(ComponentAdapter):
    """One adapter instance per linear-attention variant (``gdn`` / ``kda``)."""

    def __init__(self, variant: str, **kwargs):
        super().__init__(**kwargs)
        assert variant in LINEAR_ATTENTION_VARIANTS, variant
        self.name = variant
        self.variant = variant

    def cases(self, shape_sets: Sequence[str], passes: Sequence[str]) -> List[SuiteCase]:
        geom = LINEAR_ATTENTION_VARIANTS[self.variant]
        cases = []
        for shape_set in shape_sets:
            for shape in LINEAR_ATTENTION_SHAPES[shape_set]:
                b, t = shape["batch_size"], shape["seqlen"]
                for profile_pass in passes:
                    label = f"{self.variant} | B{b} T{t} | Hq{geom['num_q_heads']} Hkv{geom['num_kv_heads']} " f"d={geom['head_dim']} | cudnn | {profile_pass}"
                    cases.append(
                        SuiteCase(
                            component=self.name,
                            case_tag=f"{self.variant}_{shape_set}_b{b}_t{t}_{profile_pass}",
                            shape_set=shape_set,
                            profile_pass=profile_pass,
                            params={
                                "variant": self.variant,
                                "batch_size": b,
                                "seqlen": t,
                                "num_q_heads": geom["num_q_heads"],
                                "num_kv_heads": geom["num_kv_heads"],
                                "head_dim": geom["head_dim"],
                                "data_type": LINEAR_ATTENTION_DATA_TYPE,
                                "backend": "cudnn",
                            },
                            label=label,
                        )
                    )
        return cases

    def run(self, case: SuiteCase, num_iterations: int, num_warmup_iterations: int) -> CaseResult:
        p = case.params
        try:
            r = _load_la_module().run_benchmark(
                batch_size=p["batch_size"],
                seqlen=p["seqlen"],
                num_q_heads=p["num_q_heads"],
                num_kv_heads=p["num_kv_heads"],
                head_dim=p["head_dim"],
                data_type=p["data_type"],
                backend="cudnn",
                variant=self.variant,
                profile_pass=case.profile_pass,
                num_iterations=num_iterations,
                num_warmup_iterations=num_warmup_iterations,
                skip_ref=True,
            )
        except Exception as e:
            return CaseResult(case=case, status=STATUS_FAILED, error_message=str(e)[-1000:])
        return CaseResult(
            case=case,
            status=STATUS_OK,
            time_ms=r["time_ms"],
            tflops=r["tflops"],
            tb_per_sec=r["bw_tb_per_sec"],
            gpu_name=r["gpu_name"],
            cudnn_version=r["cudnn_version"],
            detail=f"benchmark_single_linear_attention --variant {self.variant} --la_backend cudnn (torch.profiler device time)",
        )
