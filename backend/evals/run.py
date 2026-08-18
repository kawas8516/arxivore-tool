"""Regression harness for the Arxivore pipeline.

Records what each stage actually produced so a prompt or model change can be
compared against a committed baseline instead of judged by eye. It measures
*health*, not relevance: no human labels are involved, so it answers "did this
change break or slow anything" rather than "did this change find better papers".

Usage:
    python -m evals.run --offline            # zero tokens, CI-safe
    python -m evals.run --topics 3           # live, 3 topics
    python -m evals.run --full               # live, all 15 topics (expensive)
    python -m evals.run --offline --write-baseline
"""

import argparse
import contextlib
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path
from unittest.mock import patch

import yaml

_HERE = Path(__file__).parent
_TOPICS_FILE = _HERE / "topics.yaml"
_BASELINE_FILE = _HERE / "baseline.json"

# Stages whose token spend and latency are tracked individually.
_STAGES = ("expand", "retrieve", "rerank", "extract", "synthesize")


def _load_topics() -> list[str]:
    with _TOPICS_FILE.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)["topics"]


@contextlib.contextmanager
def _offline():
    """Swap both external clients for deterministic fakes."""
    from evals.fakes import FakeArxivClient, FakeOpenAI

    targets = [
        patch("app.pipeline.expand.OpenAI", FakeOpenAI),
        patch("app.pipeline.rerank.OpenAI", FakeOpenAI),
        patch("app.pipeline.extract.OpenAI", FakeOpenAI),
        patch("app.pipeline.synthesize.OpenAI", FakeOpenAI),
        patch("app.pipeline.retrieve.arxiv.Client", FakeArxivClient),
    ]
    for target in targets:
        target.start()
    try:
        yield
    finally:
        for target in targets:
            target.stop()


def _run_topic(topic: str) -> dict:
    from app.service import PipelineError, run_pipeline

    start = time.monotonic()
    try:
        result = run_pipeline(topic)
    except PipelineError as exc:
        return {
            "topic": topic,
            "hard_failure": exc.stage,
            "total_ms": int((time.monotonic() - start) * 1000),
        }

    retained = result.papers_returned
    extracted_ok = retained - result.extract_errors
    unscored = sum(1 for p in result.papers if p.relevance_score is None)

    return {
        "topic": topic,
        "hard_failure": None,
        "queries_issued": len(result.queries),
        "candidates_retrieved": result.candidates_retrieved,
        "papers_retained": retained,
        "papers_unscored": unscored,
        "extract_ok": extracted_ok,
        "extract_errors": result.extract_errors,
        # The headline quality-of-service number: RELEASE.md recorded 7/18 and
        # 16/18 on the first live runs, so this is the figure to watch.
        "extract_completion_pct": round(100 * extracted_ok / retained, 1) if retained else 0.0,
        "landscape": result.landscape is not None,
        "clusters": len(result.landscape.clusters) if result.landscape else 0,
        "relationships": len(result.landscape.relationships) if result.landscape else 0,
        "tensions": len(result.landscape.tensions) if result.landscape else 0,
        "open_problems": len(result.landscape.open_problems) if result.landscape else 0,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.prompt_tokens + result.completion_tokens,
        "ms": {stage: getattr(result, f"{stage}_ms", 0) for stage in _STAGES},
        "total_ms": int((time.monotonic() - start) * 1000),
    }


def _summarise(runs: list[dict]) -> dict:
    ok = [r for r in runs if not r["hard_failure"]]

    def _mean(key: str) -> float:
        values = [r[key] for r in ok]
        return round(statistics.mean(values), 1) if values else 0.0

    return {
        "topics_run": len(runs),
        "hard_failures": len(runs) - len(ok),
        "landscapes_produced": sum(1 for r in ok if r["landscape"]),
        "mean_candidates_retrieved": _mean("candidates_retrieved"),
        "mean_papers_retained": _mean("papers_retained"),
        "total_papers_unscored": sum(r["papers_unscored"] for r in ok),
        "total_extract_errors": sum(r["extract_errors"] for r in ok),
        "mean_extract_completion_pct": _mean("extract_completion_pct"),
        "mean_total_tokens": _mean("total_tokens"),
        "total_tokens": sum(r["total_tokens"] for r in ok),
        "mean_total_ms": _mean("total_ms"),
    }


def _print_report(report: dict) -> None:
    summary = report["summary"]
    print(f"\n  mode: {report['mode']}    topics: {summary['topics_run']}\n")
    header = f"  {'topic':<40} {'cand':>5} {'kept':>5} {'extract':>9} {'clust':>6} {'tokens':>8} {'ms':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for run in report["runs"]:
        if run["hard_failure"]:
            print(f"  {run['topic']:<40} FAILED at {run['hard_failure']}")
            continue
        extracted = f"{run['extract_ok']}/{run['papers_retained']}"
        print(
            f"  {run['topic']:<40} {run['candidates_retrieved']:>5} {run['papers_retained']:>5} "
            f"{extracted:>9} {run['clusters']:>6} "
            f"{run['total_tokens']:>8} {run['total_ms']:>7}"
        )
    print()
    for key, value in summary.items():
        print(f"  {key:<32} {value}")
    print()


def _compare(report: dict) -> None:
    """Print deltas against the committed baseline, if one exists."""
    if not _BASELINE_FILE.exists():
        print("  no baseline committed yet - run with --write-baseline to create one\n")
        return
    baseline = json.loads(_BASELINE_FILE.read_text(encoding="utf-8"))
    if baseline.get("mode") != report["mode"]:
        print(
            f"  baseline is mode={baseline.get('mode')}, this run is mode="
            f"{report['mode']} - skipping diff\n"
        )
        return

    print("  vs baseline:")
    for key, current in report["summary"].items():
        previous = baseline["summary"].get(key)
        if previous is None or not isinstance(current, (int, float)):
            continue
        delta = round(current - previous, 1)
        # Wall-clock metrics move on every run regardless of code changes, so they
        # are reported but never flagged — otherwise every diff looks like a
        # regression and the marker stops meaning anything.
        noisy = key.endswith("_ms")
        marker = "  <-- changed" if delta != 0 and not noisy else ""
        sign = "+" if delta > 0 else ""
        print(f"    {key:<32} {previous} -> {current} ({sign}{delta}){marker}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Arxivore pipeline regression harness")
    parser.add_argument(
        "--topics",
        type=int,
        default=3,
        help="how many topics to run (default 3; a live full run is ~315 provider calls)",
    )
    parser.add_argument("--full", action="store_true", help="run every topic in topics.yaml")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use deterministic fakes instead of arXiv and the LLM provider (zero tokens)",
    )
    parser.add_argument("--write-baseline", action="store_true", help="overwrite baseline.json")
    parser.add_argument("--out", type=Path, default=None, help="also write the report here")
    parser.add_argument("--verbose", action="store_true", help="show pipeline logs")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.ERROR)

    # Settings require an API key even when nothing will call a provider, so
    # offline runs work on a machine (or CI runner) that has no .env.
    if args.offline:
        os.environ.setdefault("LLM_API_KEY", "offline-no-key-required")

    # Eval runs get their own database, never the developer's real reading-map
    # data: offline runs use synthetic papers that shouldn't pollute a real
    # cache, and a live eval run's own reading-map writes shouldn't reshuffle a
    # real one either. init_db() also only runs via the FastAPI lifespan in
    # normal operation — the harness calls run_pipeline() directly, so it must
    # create the schema itself.
    os.environ.setdefault("DATABASE_URL", f"sqlite:///{_HERE / 'eval_data' / 'app.db'}")
    from app.config import get_settings
    from app import storage

    get_settings.cache_clear()
    storage.init_db()

    topics = _load_topics()
    if not args.full:
        topics = topics[: max(1, args.topics)]

    runner = _offline() if args.offline else contextlib.nullcontext()
    runs: list[dict] = []
    with runner:
        for topic in topics:
            print(f"  running: {topic} ...", flush=True)
            runs.append(_run_topic(topic))

    report = {
        "mode": "offline" if args.offline else "live",
        "summary": _summarise(runs),
        "runs": runs,
    }

    _print_report(report)
    _compare(report)

    payload = json.dumps(report, indent=2) + "\n"
    if args.write_baseline:
        _BASELINE_FILE.write_text(payload, encoding="utf-8")
        print(f"  baseline written to {_BASELINE_FILE}\n")
    if args.out:
        args.out.write_text(payload, encoding="utf-8")

    # Non-zero exit on a hard failure so CI notices.
    return 1 if report["summary"]["hard_failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
