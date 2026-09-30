# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Component adapter interface for the kernel benchmark suite.

Each adapter expands its shape presets into :class:`SuiteCase`s and executes
one case per call, returning a :class:`CaseResult`. Adapters must not import
torch / cudnn / the benchmarked component at module import time — the
runner's ``--dry-run`` and ``--list-cases`` paths work on a GPU-less host.
Heavy imports belong inside :meth:`ComponentAdapter.run`.
"""

from typing import List, Optional, Sequence

from ..suite_types import CaseResult, SuiteCase


class ComponentAdapter:
    """Benchmarks one component (e.g. DSA, GDN) across its shape presets."""

    name: str = ""

    def __init__(self, repo_path: Optional[str] = None):
        # Only adapters wrapping an external repository (QSA) use repo_path.
        self.repo_path = repo_path

    def cases(self, shape_sets: Sequence[str], passes: Sequence[str]) -> List[SuiteCase]:
        """Expand this component's presets for the requested shape sets/passes."""
        raise NotImplementedError

    def run(self, case: SuiteCase, num_iterations: int, num_warmup_iterations: int) -> CaseResult:
        """Execute one case and return its result (never raises for an
        unsupported shape — report STATUS_SKIPPED / STATUS_FAILED instead)."""
        raise NotImplementedError
