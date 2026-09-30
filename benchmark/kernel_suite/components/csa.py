# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CSA fused-compressor adapter.

Benchmarks the ``cudnn.csa`` gated-softmax pooling kernels — every ``ratio``
tokens compressed into one — over ratio {2, 4, 128} x coff {1, 2} x head_dim
{128, 512}, forward and backward, by subprocessing ``workers/csa_worker.py``
(clean CUDA context and CuTe DSL compile state per case) and parsing its
``RESULT,ms,tflops,tb_per_sec,...`` line. Exit code 3 (UNSUPPORTED) maps to a
skipped row.

ratio 128 is the dedicated 128-token compress kernel family
(``compressor_sm100_r128.py``); ratio {2, 4} use the generic kernels. coff=2
is the production overlapping-window form.
"""

import os
import subprocess
import sys
from typing import List, Sequence

from ..shapes import CSA_COFFS, CSA_HEAD_DIMS, CSA_PACKS, CSA_RATIOS
from ..suite_types import STATUS_FAILED, STATUS_OK, STATUS_SKIPPED, CaseResult, SuiteCase
from .base import ComponentAdapter

_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "workers", "csa_worker.py")
UNSUPPORTED_EXIT_CODE = 3  # keep in sync with workers/csa_worker.py


class CsaAdapter(ComponentAdapter):
    name = "csa"

    def cases(self, shape_sets: Sequence[str], passes: Sequence[str]) -> List[SuiteCase]:
        cases = []
        for shape_set in shape_sets:
            pack = CSA_PACKS[shape_set]
            total = sum(pack)
            for ratio in CSA_RATIOS:
                for coff in CSA_COFFS:
                    for head_dim in CSA_HEAD_DIMS:
                        for profile_pass in passes:
                            label = f"csa | r{ratio} coff{coff} d{head_dim} | {total} tok | {profile_pass}"
                            cases.append(
                                SuiteCase(
                                    component=self.name,
                                    case_tag=f"csa_r{ratio}_c{coff}_d{head_dim}_{shape_set}_{profile_pass}",
                                    shape_set=shape_set,
                                    profile_pass=profile_pass,
                                    params={"ratio": ratio, "coff": coff, "head_dim": head_dim, "seqlens": pack},
                                    label=label,
                                )
                            )
        return cases

    def run(self, case: SuiteCase, num_iterations: int, num_warmup_iterations: int) -> CaseResult:
        p = case.params
        cmd = [
            sys.executable,
            _WORKER,
            "--profile_pass",
            case.profile_pass,
            "--ratio",
            str(p["ratio"]),
            "--coff",
            str(p["coff"]),
            "--head_dim",
            str(p["head_dim"]),
            "--seqlens",
            ",".join(str(s) for s in p["seqlens"]),
            "--num_iterations",
            str(num_iterations),
            "--num_warmup_iterations",
            str(num_warmup_iterations),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        except subprocess.TimeoutExpired:
            return CaseResult(case=case, status=STATUS_FAILED, error_message="timeout")

        if proc.returncode == UNSUPPORTED_EXIT_CODE:
            line = next((l for l in proc.stdout.splitlines() if l.startswith("UNSUPPORTED,")), "UNSUPPORTED,")
            return CaseResult(case=case, status=STATUS_SKIPPED, error_message=line.split(",", 1)[1])

        line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT,")), None)
        if proc.returncode != 0 or line is None:
            tail = "\n".join((proc.stderr or proc.stdout).splitlines()[-12:])
            return CaseResult(case=case, status=STATUS_FAILED, error_message=tail[-1500:].strip() or f"exit code {proc.returncode}, no output")

        _, ms, tflops, tb_per_sec, gpu, cudnn_version, detail = line.split(",", 6)
        tflops = float(tflops)
        return CaseResult(
            case=case,
            status=STATUS_OK,
            time_ms=float(ms),
            tflops=tflops if tflops > 0 else None,
            tb_per_sec=float(tb_per_sec),
            gpu_name=gpu,
            cudnn_version=cudnn_version,
            detail=detail,
        )
