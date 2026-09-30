# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-click kernel benchmark suite runner.

Runs the cuDNN Frontend component kernels (DSA, GDN, KDA, and the external
QSA via an adapter) over the regular and large shape presets in
``shapes.py``, one subprocess per case, and writes a CSV plus a Markdown
report under ``results/<gpu>/``.

Usage (from the repository root):

    python -m benchmark.kernel_suite.runner                      # everything
    python -m benchmark.kernel_suite.runner --dry-run            # list cases only
    python -m benchmark.kernel_suite.runner --components dsa,kda --shape-set regular
    python -m benchmark.kernel_suite.runner --passes fwd --num-iterations 50

Exit code is 1 when any case failed (skipped cases, e.g. an unserved shape
or the not-yet-integrated QSA repo, do not fail the run).
"""

import argparse
import csv
import json
import logging
import re
import sys
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, List, Optional

from .components import COMPONENT_NAMES, make_adapters
from .shapes import NUM_ITERATIONS, NUM_WARMUP_ITERATIONS, SHAPE_SETS
from .suite_types import STATUS_FAILED, STATUS_OK, STATUS_SKIPPED, CaseResult

logger = logging.getLogger("kernel_suite")

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def environment_info() -> Dict[str, Any]:
    """Software/hardware versions for the report header; tolerates a GPU-less host."""
    info: Dict[str, Any] = {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name()
            info["capability"] = ".".join(map(str, torch.cuda.get_device_capability()))
    except ImportError:
        info["torch"] = "not installed"
    try:
        import cudnn

        info["cudnn_frontend"] = cudnn.__version__
        try:
            info["cudnn_backend"] = cudnn.backend_version()
        except Exception:
            pass
    except ImportError:
        info["cudnn_frontend"] = "not installed"
    for pkg in ("nvidia-cutlass-dsl", "cuda-tile", "cuda-python"):
        try:
            info[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            pass
    return info


def gpu_slug(env: Dict[str, Any]) -> str:
    """``NVIDIA B200`` -> ``b200``; ``unknown`` on a GPU-less host."""
    name = env.get("gpu")
    if not name:
        return "unknown"
    return re.sub(r"[^a-z0-9]+", "", name.lower().replace("nvidia", "")) or "unknown"


def _fmt(value: Optional[float], digits: int = 3) -> str:
    return f"{value:.{digits}f}" if value is not None else "-"


def write_csv(results: List[CaseResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "component",
        "case_tag",
        "shape_set",
        "profile_pass",
        "label",
        "params",
        "status",
        "time_ms",
        "tflops",
        "tb_per_sec",
        "peak_mma_tflops",
        "sol_percent",
        "gpu_name",
        "cudnn_version",
        "detail",
        "error_message",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "component": r.case.component,
                    "case_tag": r.case.case_tag,
                    "shape_set": r.case.shape_set,
                    "profile_pass": r.case.profile_pass,
                    "label": r.case.label,
                    "params": json.dumps(r.case.params, sort_keys=True),
                    "status": r.status,
                    "time_ms": r.time_ms,
                    "tflops": r.tflops,
                    "tb_per_sec": r.tb_per_sec,
                    "peak_mma_tflops": r.peak_mma_tflops,
                    "sol_percent": r.sol_percent,
                    "gpu_name": r.gpu_name,
                    "cudnn_version": r.cudnn_version,
                    "detail": r.detail,
                    "error_message": r.error_message,
                }
            )


def write_markdown(results: List[CaseResult], env: Dict[str, Any], args_summary: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Kernel Benchmark Suite Report",
        "",
        "## Environment",
        "",
    ]
    for key, value in env.items():
        lines.append(f"- **{key}**: {value}")
    lines += [
        f"- **suite**: {args_summary}",
        "",
        "Timing convention differs per component: DSA/dsa_aux/CSA rows are CUDA-event wall time;"
        " GDN/KDA rows are torch.profiler CUDA device time (see"
        " benchmark/kernel_suite/README.md).",
        "",
    ]
    components = []
    for r in results:
        if r.case.component not in components:
            components.append(r.case.component)
    for component in components:
        comp_results = [r for r in results if r.case.component == component]
        lines += [f"## {component.upper()}", ""]
        for shape_set in SHAPE_SETS:
            set_results = [r for r in comp_results if r.case.shape_set == shape_set]
            if not set_results:
                continue
            lines += [
                f"### {shape_set}",
                "",
                "| case | pass | time (ms) | TFLOPS | TB/s | %SOL | status | note |",
                "|---|---|---:|---:|---:|---:|---|---|",
            ]
            for r in set_results:
                sol = _fmt(r.sol_percent, 1)
                label = r.case.label.replace("|", ";")
                note = " ".join((r.error_message or r.detail or "").split()).replace("|", ";")[:300]
                lines.append(
                    f"| {label} | {r.case.profile_pass} | {_fmt(r.time_ms)} | {_fmt(r.tflops, 1)} " f"| {_fmt(r.tb_per_sec, 2)} | {sol} | {r.status} | {note} |"
                )
            lines.append("")
    path.write_text("\n".join(lines))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--components", default=",".join(COMPONENT_NAMES), help="Comma-separated subset of: " + ",".join(COMPONENT_NAMES))
    p.add_argument("--shape-set", default="all", choices=["regular", "large", "all"], help="Which shape presets to run (default: all)")
    p.add_argument("--passes", default="fwd,bwd", help="Comma-separated passes: fwd,bwd (default: both)")
    p.add_argument("--num-iterations", type=int, default=NUM_ITERATIONS, help="Measured iterations per case")
    p.add_argument("--num-warmup-iterations", type=int, default=NUM_WARMUP_ITERATIONS, help="Warmup iterations per case (kernel compile happens here)")
    p.add_argument("--output-dir", type=Path, default=None, help="Report directory (default: results/<gpu>/)")
    p.add_argument("--gpu-label", default=None, help="GPU label for the results directory (default: auto-detect)")
    p.add_argument("--qsa-repo", default=None, help="Path to the external Qwen sparse-attention repo (QSA adapter; or set QSA_REPO_PATH)")
    p.add_argument("--dry-run", action="store_true", help="List the cases that would run, without executing (works without a GPU)")
    p.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stderr)

    components = [c.strip() for c in args.components.split(",") if c.strip()]
    unknown = set(components) - set(COMPONENT_NAMES)
    if unknown:
        logger.error(f"Unknown components: {sorted(unknown)} (available: {COMPONENT_NAMES})")
        return 2
    shape_sets = list(SHAPE_SETS) if args.shape_set == "all" else [args.shape_set]
    passes = [p.strip() for p in args.passes.split(",") if p.strip()]

    adapters = make_adapters(qsa_repo=args.qsa_repo)
    cases = []
    for name in components:
        cases.extend(adapters[name].cases(shape_sets, passes))

    if args.dry_run:
        print(f"Would run {len(cases)} benchmark cases:")
        for i, case in enumerate(cases, 1):
            print(f"  [{i:3d}] {case.component:4s} | {case.shape_set:7s} | {case.label}")
        return 0

    env = environment_info()
    logger.info(f"Environment: {env}")
    logger.info(f"Running {len(cases)} benchmark cases (components={components}, shape_sets={shape_sets}, passes={passes})")

    results: List[CaseResult] = []
    for i, case in enumerate(cases, 1):
        logger.info(f"[{i}/{len(cases)}] {case.component} | {case.label}")
        try:
            r = adapters[case.component].run(case, args.num_iterations, args.num_warmup_iterations)
        except Exception as e:
            logger.exception("case raised unexpectedly")
            r = CaseResult(case=case, status=STATUS_FAILED, error_message=f"adapter raised: {e}")
        results.append(r)
        if r.status == STATUS_OK:
            tflops_str = f"{r.tflops:.1f} TFLOPS" if r.tflops is not None else "-"
            sol = f", {r.sol_percent:.1f}% SOL" if r.sol_percent else ""
            logger.info(f"  -> {r.time_ms:.3f} ms ({tflops_str}{sol})")
        elif r.status == STATUS_SKIPPED:
            logger.info(f"  -> SKIPPED: {r.error_message}")
        else:
            lines = (r.error_message or "").splitlines()
            last_line = next((l for l in reversed(lines) if l.strip() and not l.strip().endswith(":")), "")
            logger.warning(f"  -> FAILED: {last_line[:200]}")

    label = args.gpu_label or gpu_slug(env)
    output_dir = args.output_dir or RESULTS_DIR / label
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"suite_{timestamp}.csv"
    md_path = output_dir / f"suite_{timestamp}.md"
    write_csv(results, csv_path)
    args_summary = f"components={components} shape_sets={shape_sets} passes={passes} iterations={args.num_iterations} warmup={args.num_warmup_iterations}"
    write_markdown(results, env, args_summary, md_path)

    n_ok = sum(r.status == STATUS_OK for r in results)
    n_skip = sum(r.status == STATUS_SKIPPED for r in results)
    n_fail = sum(r.status == STATUS_FAILED for r in results)
    print(f"Done: {n_ok} ok, {n_skip} skipped, {n_fail} failed out of {len(results)} cases")
    print(f"CSV:    {csv_path}")
    print(f"Report: {md_path}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
