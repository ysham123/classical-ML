"""Command line entry point: ``python -m classical_ml [projects] [--all] [--list]``."""

from __future__ import annotations

import argparse
import sys

from . import PROJECTS, REQUIRES_DOWNLOAD
from .plotting import use_house_style
from .report import format_metrics, run_project, write_results, write_results_markdown


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="classical-ml",
        description="Run the classical machine learning projects and refresh outputs/.",
    )
    parser.add_argument("projects", nargs="*", help=f"projects to run: {', '.join(PROJECTS)}")
    parser.add_argument("--all", action="store_true", help="run every project")
    parser.add_argument("--list", action="store_true", help="list the available projects and exit")
    parser.add_argument("--report", action="store_true",
                        help="rebuild outputs/RESULTS.md from outputs/results.json and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.report:
        print(f"wrote {write_results_markdown()}")
        return 0

    if args.list:
        for name, description in PROJECTS.items():
            marker = " (needs `make data`)" if name in REQUIRES_DOWNLOAD else ""
            print(f"  {name:<10} {description}{marker}")
        return 0

    selected = list(PROJECTS) if args.all else list(dict.fromkeys(args.projects))
    if not selected:
        _build_parser().print_help()
        return 1

    unknown = [name for name in selected if name not in PROJECTS]
    if unknown:
        print(f"error: no such project(s): {unknown}. Try --list.", file=sys.stderr)
        return 2

    use_house_style()
    results = []
    for name in selected:
        print(f"\n{'=' * 78}\n{PROJECTS[name]}\n{'=' * 78}")
        result = run_project(name)
        results.append(result)
        print(f"\n[{name}] finished in {result.seconds:.1f}s with {len(result.figures)} figures")
        print(f"[{name}] {format_metrics(result.metrics, limit=6)} ...")

    write_results(results)
    write_results_markdown()
    print(f"\nwrote outputs/results.json and outputs/RESULTS.md for {', '.join(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
