# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""QSA (Qwen sparse attention) adapter — stub.

QSA lives in an external repository (to be provided with the benchmark
environment), so this adapter is a pluggable integration point rather than
an implementation:

- Without a repo path (``--qsa-repo`` CLI flag or ``QSA_REPO_PATH``
  environment variable), every case is reported as ``skipped`` so the suite
  report shows the intended coverage.
- With a repo path, :meth:`QsaAdapter.run` is the single place to fill in:
  import the external implementation, build the inputs for ``case.params``,
  and time it. Follow the DSA worker's timing convention (CUDA events around
  each steady-state call after a 256 MiB L2 flush, median over iterations)
  so QSA numbers are comparable with the rest of the report. See the README's
  QSA section for the contract.
"""

import os
from typing import List, Optional, Sequence

from ..shapes import QSA_SHAPES
from ..suite_types import STATUS_SKIPPED, CaseResult, SuiteCase
from .base import ComponentAdapter


class QsaAdapter(ComponentAdapter):
    name = "qsa"

    def __init__(self, repo_path: Optional[str] = None, **kwargs):
        super().__init__(repo_path=repo_path or os.environ.get("QSA_REPO_PATH"), **kwargs)

    def cases(self, shape_sets: Sequence[str], passes: Sequence[str]) -> List[SuiteCase]:
        cases = []
        for shape_set in shape_sets:
            for shape in QSA_SHAPES[shape_set]:
                for profile_pass in passes:
                    cases.append(
                        SuiteCase(
                            component=self.name,
                            case_tag=f"qsa_{shape_set}_s{shape['seqlen']}_{profile_pass}",
                            shape_set=shape_set,
                            profile_pass=profile_pass,
                            params={"seqlen": shape["seqlen"]},
                            label=f"qsa | T{shape['seqlen']} | external repo | {profile_pass}",
                        )
                    )
        return cases

    def run(self, case: SuiteCase, num_iterations: int, num_warmup_iterations: int) -> CaseResult:
        if not self.repo_path:
            return CaseResult(
                case=case,
                status=STATUS_SKIPPED,
                error_message="external Qwen sparse-attention repo not provided (set QSA_REPO_PATH or pass --qsa-repo)",
            )
        # TODO(qsa-integration): once the external Qwen sparse-attention repo
        # is available, implement the measurement here:
        #   1. sys.path.insert(0, self.repo_path) and import the QSA kernels
        #      (or subprocess a worker script in that repo, mirroring
        #      benchmark/dsa/benchmark_single_dsa.py's RESULT-line contract).
        #   2. Build inputs from case.params (seqlen, shape_set, profile_pass).
        #   3. Time with CUDA events after a 256 MiB L2 flush, median of
        #      num_iterations after num_warmup_iterations.
        #   4. Return CaseResult(case=case, status=STATUS_OK, time_ms=..., tflops=...,
        #      tb_per_sec=..., detail=...) — every kernel in this suite reports
        #      both compute (TFLOPS) and DRAM bandwidth (TB/s); derive TB/s
        #      from a byte-traffic model like the DSA adapter's.
        return CaseResult(
            case=case,
            status=STATUS_SKIPPED,
            error_message=f"QSA adapter not implemented yet (repo at {self.repo_path}); see benchmark/kernel_suite/README.md",
        )
