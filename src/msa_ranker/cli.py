"""`msa-ranker` CLI (C11). `ingest` (S-1) · `train` (S-4) · `report` + `deploy` (S-5);
`eval` is a stub (eval runs inside `train`)."""

from __future__ import annotations

import argparse
import sys

from .config import DEFAULT_DB_PATH, DEFAULT_EVAL_FRAC, DEFAULT_K, DEFAULT_MIN_LABELED_SEARCHES
from .datasets import InsufficientDataError
from .db import open_db
from .deploy import NotDeployableError, deploy
from .ingest import ingest
from .report import render_report, report_json
from .train import run_training

_STUBS = ("eval",)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="msa-ranker", description="msa-ranker lifecycle CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="fold the JSONL event ledger into the training SoR")
    p_ingest.add_argument("--ledger-dir", required=True, help="directory of events-*.jsonl[.gz]")
    p_ingest.add_argument("--db", default=str(DEFAULT_DB_PATH), help="training SoR path")

    p_train = sub.add_parser(
        "train", help="freeze -> baseline -> train -> eval -> register (spec 05)"
    )
    p_train.add_argument("--db", default=str(DEFAULT_DB_PATH), help="training SoR path")
    p_train.add_argument("--ledger-dir", help="optional: ingest this ledger dir before training")
    p_train.add_argument(
        "--out", required=True, help="output dir for the model artifact + manifest"
    )
    p_train.add_argument("--algo", default="logreg", help="model algorithm (v1: logreg)")
    p_train.add_argument("--k", type=int, default=DEFAULT_K, help="NDCG@k / MRR cutoff")
    p_train.add_argument(
        "--seed",
        type=int,
        default=0,
        help="dataset-split seed (recorded in model params; the v1 logreg fit is "
        "deterministic regardless of seed)",
    )
    p_train.add_argument(
        "--eval-frac", type=float, default=DEFAULT_EVAL_FRAC, help="held-out fraction"
    )
    p_train.add_argument(
        "--min-searches",
        type=int,
        default=DEFAULT_MIN_LABELED_SEARCHES,
        help="min labelled searches before training is allowed",
    )

    p_report = sub.add_parser("report", help="show the model registry + eval lineage (FR-17)")
    p_report.add_argument("--db", default=str(DEFAULT_DB_PATH), help="training SoR path")
    p_report.add_argument("--json", action="store_true", help="emit JSON instead of a table")

    p_deploy = sub.add_parser("deploy", help="copy a beats-baseline model to the serving dir")
    p_deploy.add_argument("--db", default=str(DEFAULT_DB_PATH), help="training SoR path")
    p_deploy.add_argument("--model-id", required=True, help="model_id to deploy")
    p_deploy.add_argument("--dest", required=True, help="serving dir (MSA's ltr_model_dir)")
    p_deploy.add_argument(
        "--force", action="store_true", help="deploy even if it did not beat baseline"
    )

    for name in _STUBS:
        sub.add_parser(name, help=f"{name} — runs inside `train`; no standalone command")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "ingest":
        conn = open_db(args.db)
        try:
            stats = ingest(conn, args.ledger_dir)
        finally:
            conn.close()
        print(
            f"ingested: files={stats['files']} lines={stats['lines']} "
            f"inserted={stats['inserted']} skipped={stats['skipped']}"
        )
        return 0

    if args.cmd == "train":
        conn = open_db(args.db)
        try:
            if args.ledger_dir:
                ingest(conn, args.ledger_dir)
            result = run_training(
                conn,
                out_dir=args.out,
                algo=args.algo,
                k=args.k,
                seed=args.seed,
                eval_frac=args.eval_frac,
                min_searches=args.min_searches,
            )
        except InsufficientDataError as exc:
            print(f"train refused: {exc}", file=sys.stderr)
            return 1
        except ValueError as exc:  # bad flag value (e.g. --eval-frac out of range)
            print(f"train error: {exc}", file=sys.stderr)
            return 2
        finally:
            conn.close()
        print(
            f"trained model={result['model_id']} dataset={result['dataset_id']}\n"
            f"  NDCG@{result['k']}: model={result['model_ndcg']:.4f} "
            f"baseline={result['baseline_ndcg']:.4f} "
            f"beats_baseline={result['beats_baseline']} "
            f"(eval queries={result['n_eval_queries']})\n"
            f"  artifact: {result['artifact']}"
        )
        return 0

    if args.cmd == "report":
        conn = open_db(args.db)
        try:
            print(report_json(conn) if args.json else render_report(conn))
        finally:
            conn.close()
        return 0

    if args.cmd == "deploy":
        conn = open_db(args.db)
        try:
            dest = deploy(conn, args.model_id, args.dest, force=args.force)
        except NotDeployableError as exc:
            print(f"deploy refused: {exc}", file=sys.stderr)
            return 1
        finally:
            conn.close()
        print(f"deployed {args.model_id} -> {dest}")
        return 0

    print(f"'{args.cmd}' runs inside `train`; no standalone command.", file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
