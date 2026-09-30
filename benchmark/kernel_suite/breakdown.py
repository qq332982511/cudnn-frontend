# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-kernel profiler breakdown runner.

For every suite case (same enumeration as ``runner.py``), subprocesses
``workers/breakdown_worker.py``, collects its ``KERNEL,`` lines, and writes:

- ``breakdown_<timestamp>.csv``: one row per (case, kernel) with calls,
  average total us per op call, and the kernel's share of the case's GPU time.
- ``breakdown_<timestamp>.md``: per-component, per-case tables (top 15
  kernels + an "other" roll-up).

This is how the auxiliary kernels *inside* the fused ops become visible: the
op-level suite (runner.py) reports one number per op, here each CUDA kernel
(l2norm, chunk scan, state passing, index padding/convert, ...) gets its own
row. DSA's standalone auxiliary ops are covered as ``dsa_aux`` cases.

Usage (from the repository root):

    python -m benchmark.kernel_suite.breakdown                      # everything
    python -m benchmark.kernel_suite.breakdown --components gdn,kda --shape-set regular
    python -m benchmark.kernel_suite.breakdown --dry-run
"""

import argparse
import csv
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .components import make_adapters
from .runner import RESULTS_DIR, environment_info, gpu_slug
from .shapes import SHAPE_SETS
from .suite_types import SuiteCase

logger = logging.getLogger("kernel_suite.breakdown")

_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workers", "breakdown_worker.py")
_TOP_KERNELS = 15

BREAKDOWN_COMPONENTS = [
    "dsa",
    "dsa_aux",
    "csa",
    "gdn",
    "kda",
    "la_aux",
]  # qsa has no in-repo kernels; gdn2/gdp need different input plumbing (suite-timed only)


def worker_cmd(case: SuiteCase, iters: int) -> List[str]:
    """Rebuild the worker CLI from the case's params (same source as the timing run)."""
    p = case.params
    cmd = [sys.executable, _WORKER, "--component", case.component, "--profile_pass", case.profile_pass, "--iters", str(iters)]
    if case.component == "dsa":
        cmd += [
            "--q_seqlen",
            str(p["q_seqlen"]),
            "--kv_seqlen",
            str(p["kv_seqlen"]),
            "--num_q_heads",
            str(p["num_q_heads"]),
            "--head_dim_qk",
            str(p["head_dim_qk"]),
            "--head_dim_vo",
            str(p["head_dim_vo"]),
            "--topk",
            str(p["topk"]),
            "--indexer_topk",
            str(p["indexer_topk"]),
        ]
        if p["has_sink"]:
            cmd.append("--has_sink")
    elif case.component == "dsa_aux":
        cmd += ["--op", p["op"], "--seqlen", str(p["seqlen"]), "--heads", str(p["heads"]), "--top_k", str(p["top_k"]), "--ratio", str(p["ratio"])]
    elif case.component == "csa":
        cmd += [
            "--ratio",
            str(p["ratio"]),
            "--coff",
            str(p["coff"]),
            "--head_dim",
            str(p["head_dim"]),
            "--seqlens",
            ",".join(str(s) for s in p["seqlens"]),
        ]
    elif case.component == "la_aux":
        cmd += [
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
        ]
    else:  # gdn / kda
        cmd += [
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
        ]
    return cmd


def run_case(case: SuiteCase, iters: int) -> List[Dict[str, Any]]:
    """Profile one case; returns kernel rows (empty on failure, logged)."""
    try:
        proc = subprocess.run(worker_cmd(case, iters), capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        logger.warning("  -> TIMEOUT")
        return []
    rows = []
    for line in proc.stdout.splitlines():
        if line.startswith("KERNEL,"):
            _, profile_pass, name, count, total_us = line.split(",", 4)
            rows.append({"kernel": name, "calls": int(count), "total_us": float(total_us)})
    if not rows:
        tail = "\n".join((proc.stderr or proc.stdout).splitlines()[-8:])
        logger.warning(f"  -> no kernel rows (exit {proc.returncode}): {tail[-300:]}")
        return []
    total = sum(r["total_us"] for r in rows)
    for r in rows:
        r["share_pct"] = r["total_us"] / total * 100.0 if total else 0.0
    return rows


def write_csv(all_rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["component", "case_tag", "shape_set", "profile_pass", "label", "kernel", "calls", "total_us", "share_pct"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)


def write_markdown(all_rows: List[Dict[str, Any]], env: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Kernel Breakdown Report", "", "## Environment", ""]
    for key, value in env.items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")
    lines.append(
        "Each table is one op call profiled with torch.profiler (CUDA activity); "
        "`us/call` is averaged over the profiled iterations, `share` is the "
        "kernel's fraction of the case's total GPU kernel time."
    )
    lines.append("")
    by_case: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for r in all_rows:
        if r["case_tag"] not in by_case:
            by_case[r["case_tag"]] = []
            order.append(r["case_tag"])
        by_case[r["case_tag"]].append(r)
    for case_tag in order:
        rows = by_case[case_tag]
        lines += [f"## {rows[0]['component']} — {rows[0]['label']}", ""]
        lines += ["| kernel | calls | us/call | share |", "|---|---:|---:|---:|"]
        top = sorted(rows, key=lambda r: -r["total_us"])
        shown, rest = top[:_TOP_KERNELS], top[_TOP_KERNELS:]
        for r in shown:
            lines.append(f"| `{r['kernel']}` | {r['calls']} | {r['total_us']:.1f} | {r['share_pct']:.1f}% |")
        if rest:
            lines.append(
                f"| (other {len(rest)} kernels) | {sum(r['calls'] for r in rest)} | {sum(r['total_us'] for r in rest):.1f} | {sum(r['share_pct'] for r in rest):.1f}% |"
            )
        lines.append("")
    path.write_text("\n".join(lines))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--components", default=",".join(BREAKDOWN_COMPONENTS))
    p.add_argument("--shape-set", default="all", choices=["regular", "large", "all"])
    p.add_argument("--passes", default="fwd,bwd")
    p.add_argument("--iters", type=int, default=3, help="profiled iterations per case")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--gpu-label", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stderr)

    components = [c.strip() for c in args.components.split(",") if c.strip()]
    unknown = set(components) - set(BREAKDOWN_COMPONENTS)
    if unknown:
        logger.error(f"Unknown components: {sorted(unknown)} (available: {BREAKDOWN_COMPONENTS})")
        return 2
    shape_sets = list(SHAPE_SETS) if args.shape_set == "all" else [args.shape_set]
    passes = [p.strip() for p in args.passes.split(",") if p.strip()]

    adapters = make_adapters()
    cases = []
    for name in components:
        cases.extend(adapters[name].cases(shape_sets, passes))

    if args.dry_run:
        print(f"Would profile {len(cases)} cases:")
        for i, case in enumerate(cases, 1):
            print(f"  [{i:3d}] {case.component:7s} | {case.shape_set:7s} | {case.label}")
        return 0

    env = environment_info()
    logger.info(f"Environment: {env}")
    logger.info(f"Profiling {len(cases)} cases")

    all_rows: List[Dict[str, Any]] = []
    for i, case in enumerate(cases, 1):
        logger.info(f"[{i}/{len(cases)}] {case.component} | {case.label}")
        rows = run_case(case, args.iters)
        for r in rows:
            all_rows.append(
                {
                    "component": case.component,
                    "case_tag": case.case_tag,
                    "shape_set": case.shape_set,
                    "profile_pass": case.profile_pass,
                    "label": case.label,
                    **r,
                }
            )
        if rows:
            top = rows[0]
            logger.info(f"  -> {len(rows)} kernels; top: {top['kernel'][:60]} ({top['share_pct']:.0f}%)")

    label = args.gpu_label or gpu_slug(env)
    output_dir = args.output_dir or RESULTS_DIR / label
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"breakdown_{timestamp}.csv"
    md_path = output_dir / f"breakdown_{timestamp}.md"
    write_csv(all_rows, csv_path)
    write_markdown(all_rows, env, md_path)
    print(f"CSV:    {csv_path}")
    print(f"Report: {md_path}")
    return 0 if all_rows else 1


if __name__ == "__main__":
    sys.exit(main())
