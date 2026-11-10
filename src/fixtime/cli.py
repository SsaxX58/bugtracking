"""Command line entry point: `fixtime run` and `fixtime report`."""

from __future__ import annotations

import argparse
import sys

from pydantic import ValidationError

from fixtime.config import load_config
from fixtime.records import STAGES, TASKS
from fixtime.runner import run, write_manifest

DEFAULT_CONFIG = "configs/default.yaml"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog="fixtime")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="fit and evaluate models (skips completed units)")
    p_run.add_argument("--config", default=DEFAULT_CONFIG)
    p_run.add_argument("--scope", nargs="+", help="projects and/or 'aggregated' (default: all)")
    p_run.add_argument("--dataset", nargs="+", help="dataset names, e.g. T0 T24 (default: all)")
    p_run.add_argument("--stage", nargs="+", choices=STAGES, default=list(STAGES))
    p_run.add_argument("--task", nargs="+", choices=TASKS, default=list(TASKS))
    p_run.add_argument("--model", nargs="+", help="model names (default: all)")
    p_run.add_argument("--force", action="store_true", help="rerun units even if cached")
    p_run.add_argument("--limit", type=int, help="subsample N rows per dataset (smoke tests)")
    p_run.add_argument("--n-jobs", type=int, help="override n_jobs from the config")
    p_run.add_argument("--run-name", help="override run_name from the config")

    p_rep = sub.add_parser("report", help="collect saved metrics into CSV tables")
    p_rep.add_argument("--config", default=DEFAULT_CONFIG)
    p_rep.add_argument("--run-name", help="override run_name from the config")

    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ValidationError as e:
        print(f"invalid configuration {args.config}:\n{e}", file=sys.stderr)
        return 2
    if args.run_name:
        cfg.run_name = args.run_name

    if args.command == "run":
        if args.n_jobs is not None:
            cfg.n_jobs = args.n_jobs
        if args.limit:
            # subsampled runs get their own splits and results, never mixed with full runs
            cfg.run_name = f"{cfg.run_name}_limit{args.limit}"
        known_models = {*cfg.models.regression.enabled(), *cfg.models.classification.enabled()}
        unknown = set(args.scope or []) - set(cfg.scopes)
        unknown |= set(args.dataset or []) - set(cfg.datasets)
        unknown |= set(args.model or []) - known_models
        if unknown:
            parser.error(f"unknown scope/dataset/model: {sorted(unknown)}")
        print(f"run dir: {cfg.run_dir}")
        write_manifest(cfg, ["fixtime", *argv])
        failures = run(
            cfg,
            scopes=args.scope,
            datasets=args.dataset,
            stages=args.stage,
            tasks=args.task,
            models=args.model,
            force=args.force,
            limit=args.limit,
        )
        if failures:
            print(f"{len(failures)} units failed")
            return 1
        return 0

    from fixtime.report import build_report

    for path in build_report(cfg):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
