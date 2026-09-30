# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Linear-attention auxiliary-kernel adapter.

Benchmarks the standalone per-span summary ops of GDN/KDA (the context-
parallelism state / transition kernels: ``gated_delta_net_summary(_bwd)``,
``kimi_delta_attention_summary(_bwd)``) by subprocessing
``workers/la_aux_worker.py`` (clean CUDA context and kernel compile state per
case) and parsing its ``RESULT,ms,tflops,tb_per_sec,...`` line. Exit code 3
(UNSUPPORTED) maps to a skipped row.

The main fused ops are covered by the ``gdn`` / ``kda`` components; this
component's name is ``la_aux``.
"""

import os
import subprocess
import sys
from typing import List, Sequence

from ..shapes import LA_AUX_OPS, LINEAR_ATTENTION_SHAPES, LINEAR_ATTENTION_VARIANTS
from ..suite_types import STATUS_FAILED, STATUS_OK, STATUS_SKIPPED, CaseResult, SuiteCase
from .base import ComponentAdapter

_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "workers", "la_aux_worker.py")
UNSUPPORTED_EXIT_CODE = 3  # keep in sync with workers/la_aux_worker.py


class LaAuxAdapter(ComponentAdapter):
    name = "la_aux"

    def cases(self, shape_sets: Sequence[str], passes: Sequence[str]) -> List[SuiteCase]:
        cases = []
        for shape_set in shape_sets:
            for op, (variant, profile_pass) in LA_AUX_OPS.items():
                if profile_pass not in passes:
                    continue
                geom = LINEAR_ATTENTION_VARIANTS[variant]
                for shape in LINEAR_ATTENTION_SHAPES[shape_set]:
                    b, t = shape["batch_size"], shape["seqlen"]
                    label = f"{op} | B{b} T{t} | Hq{geom['num_q_heads']} Hkv{geom['num_kv_heads']} d={geom['head_dim']} | {profile_pass}"
                    cases.append(
                        SuiteCase(
                            component=self.name,
                            case_tag=f"la_aux_{op}_{shape_set}_b{b}_t{t}",
                            shape_set=shape_set,
                            profile_pass=profile_pass,
                            params={
                                "op": op,
                                "variant": variant,
                                "batch_size": b,
                                "seqlen": t,
                                "num_q_heads": geom["num_q_heads"],
                                "num_kv_heads": geom["num_kv_heads"],
                                "head_dim": geom["head_dim"],
                            },
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
            "--variant",
            p["variant"],
            "--batch_size",
            str(p["batch_size"]),
            "--seqlen",
            str(p["seqlen"]),
            "--num_q_heads",
            str(p["num_q_heads"]),
            "--num_kv_heads",
            str(p["num_kv_heads"]),
            "--head_dim",
            str(p["head_dim"]),
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
