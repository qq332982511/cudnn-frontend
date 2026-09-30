# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Component registry for the kernel benchmark suite."""

from .csa import CsaAdapter
from .dsa import DsaAdapter
from .dsa_aux import DsaAuxAdapter
from .la_aux import LaAuxAdapter
from .linear_attention import LinearAttentionAdapter
from .qsa import QsaAdapter


def make_adapters(qsa_repo=None):
    """Instantiate every component adapter (``qsa_repo`` feeds the QSA stub)."""
    adapters = [
        DsaAdapter(),
        DsaAuxAdapter(),
        CsaAdapter(),
        LinearAttentionAdapter("gdn"),
        LinearAttentionAdapter("kda"),
        LinearAttentionAdapter("gdn2"),
        LinearAttentionAdapter("gdp"),
        LaAuxAdapter(),
        QsaAdapter(repo_path=qsa_repo),
    ]
    return {a.name: a for a in adapters}


COMPONENT_NAMES = ["dsa", "dsa_aux", "csa", "gdn", "kda", "gdn2", "gdp", "la_aux", "qsa"]
