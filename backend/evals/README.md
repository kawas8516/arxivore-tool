# Pipeline regression harness

Answers *"did this change break or slow anything"* — not *"did this change find
better papers"*. There are no human relevance labels here, so it measures
pipeline **health**: parse failures, extraction completion, token spend, latency.

Add a labelled set later to layer `recall@50` / `precision@18` on top; the runner
already reports everything those metrics would build on.

## Usage

```bash
cd backend

python -m evals.run --offline              # zero tokens, no key needed, CI-safe
python -m evals.run --offline --write-baseline

python -m evals.run --topics 3             # live, first 3 topics
python -m evals.run --full                 # live, all 15 topics
```

## Why offline mode exists

A live `--full` run is roughly:

```
15 topics x (1 expand + 1 rerank + 18 extract + 1 synthesize) = ~315 provider calls
```

On the free tier that exhausts the daily quota in a single invocation. Offline
mode swaps `arxiv.Client` and `OpenAI` for deterministic fakes (`fakes.py`) that
still exercise the real fence-stripping, schema-validation, pruning, and token
accounting paths — so wiring regressions get caught for free, and live runs are
reserved for questions that genuinely need a real model.

`--topics` defaults to **3** for the same reason. Topic order in `topics.yaml` is
fixed, so a 3-topic run is comparable against a previous 3-topic run.

## Determinism

`fakes.py` uses `zlib.crc32`, never `hash()`: Python randomises string hashing
per process, which would make offline token counts drift between runs and render
the committed baseline undiffable.

Metrics ending in `_ms` are wall-clock and move on every run. They are reported
but never flagged as changed, so the `<-- changed` marker keeps meaning something.

## What to watch

| Metric | Why |
|---|---|
| `hard_failures` | Non-zero means retrieve or rerank died — exits non-zero for CI |
| `mean_extract_completion_pct` | The 7/18 vs 16/18 figure from `RELEASE.md`. Free-tier rate limiting shows up here first |
| `total_papers_unscored` | Papers the reranker skipped; they sort last and get silently dropped |
| `landscapes_produced` | Below `topics_run` means synthesis is failing or returning empty |
| `total_tokens` | Cost regression signal. Prompt changes show up immediately |
| `mean_candidates_retrieved` | Query-expansion recall. Should sit at the `max_candidates` cap |

## Files

| Path | Role |
|---|---|
| `topics.yaml` | Fixed topic set, stable order |
| `run.py` | Runner, reporting, baseline diffing |
| `fakes.py` | Deterministic arXiv + LLM stand-ins for `--offline` |
| `baseline.json` | Committed offline snapshot to diff against |
