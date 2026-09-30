# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared data types for the kernel benchmark suite.

A :class:`SuiteCase` is one (component, shape, pass) benchmark invocation;
every component adapter expands its shape presets into cases and executes
them. A :class:`CaseResult` is the outcome of one executed case and becomes
one row in the CSV report and one cell row in the Markdown report.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

STATUS_OK = "ok"
STATUS_SKIPPED = "skipped"  # the component/arch/shape combination is honestly not served
STATUS_FAILED = "failed"


@dataclass
class SuiteCase:
    """One benchmark case: a component run at one shape for one pass.

    Attributes:
        component: Component identifier (``dsa``, ``gdn``, ``kda``, ``qsa``).
        case_tag: Unique CSV-safe label for the case.
        shape_set: ``regular`` or ``large``.
        profile_pass: ``fwd`` or ``bwd``.
        params: Component-specific shape parameters, reported verbatim.
        label: Human-readable one-liner for logs and the report.
    """

    component: str
    case_tag: str
    shape_set: str
    profile_pass: str
    params: Dict[str, Any] = field(default_factory=dict)
    label: str = ""


@dataclass
class CaseResult:
    """Outcome of one executed :class:`SuiteCase` (one report row)."""

    case: SuiteCase
    status: str
    time_ms: Optional[float] = None
    tflops: Optional[float] = None
    tb_per_sec: Optional[float] = None
    peak_mma_tflops: Optional[float] = None
    error_message: Optional[str] = None
    gpu_name: Optional[str] = None
    cudnn_version: Optional[str] = None
    detail: Optional[str] = None

    @property
    def sol_percent(self) -> Optional[float]:
        """Achieved TFLOPS as a percentage of the sampled dense-MMA peak."""
        if self.tflops and self.peak_mma_tflops:
            return self.tflops / self.peak_mma_tflops * 100.0
        return None
