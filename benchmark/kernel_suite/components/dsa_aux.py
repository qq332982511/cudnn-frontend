# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DSA auxiliary-kernel adapter.

Benchmarks every auxiliary op of the DSA stack individually — the indexer
(dense and fused-compressed forward), the standalone radix top-K, the two
training-loss score-recompute helpers, and the indexer backward — by
subprocessing ``workers/dsa_aux_worker.py`` (clean CUDA context and CuTe DSL
compile state per case) and parsing its ``RESULT,ms,tflops,tb_per_sec,...``
line. Exit code 3 (UNSUPPORTED) maps to a skipped row.

The main sparse-attention kernel itself is covered by the ``dsa`` component;
this component's name is ``dsa_aux``.
"""

import os
import subprocess
import sys
from typing import List, Sequence

from ..shapes import DSA_AUX_OPS, DSA_AUX_RATIO, DSA_AUX_SHAPES
from ..suite_types import STATUS_FAILED, STATUS_OK, STATUS_SKIPPED, CaseResult, SuiteCase
from .base import ComponentAdapter

_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "workers", "dsa_aux_worker.py")
UNSUPPORTED_EXIT_CODE = 3  # keep in sync with workers/dsa_aux_worker.py


class DsaAuxAdapter(ComponentAdapter):
    name = "dsa_aux"

    def cases(self, shape_sets: Sequence[str], passes: Sequence[str]) -> List[SuiteCase]:
        cases = []
        for shape_set in shape_sets:
            for op, (profile_pass, heads, top_k) in DSA_AUX_OPS.items():
                if profile_pass not in passes:
                    continue
                for t in DSA_AUX_SHAPES[op][shape_set]:
                    label = f"{op} | T{t} s_k=T/{DSA_AUX_RATIO} | H{heads} K={top_k} | {profile_pass}"
                    cases.append(
                        SuiteCase(
                            component=self.name,
                            case_tag=f"dsa_aux_{op}_{shape_set}_t{t}",
                            shape_set=shape_set,
                            profile_pass=profile_pass,
                            params={"op": op, "seqlen": t, "heads": heads, "top_k": top_k, "ratio": DSA_AUX_RATIO},
                            label=label,
                        )
                    )
        return cases

    def run(self, case: SuiteCase, num_iterations: int, num_warmup_iterations: int) -> CaseResult:
        p = case.params
        cmd = [
            sys.executable,
            _WORKER,
            "--op",
            p["op"],
            "--seqlen",
            str(p["seqlen"]),
            "--heads",
            str(p["heads"]),
            "--top_k",
            str(p["top_k"]),
            "--ratio",
            str(p["ratio"]),
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
