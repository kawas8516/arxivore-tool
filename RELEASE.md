# Release Notes — Arxivore

Build-by-build record of what shipped, the security lapses found and fixed, the
models used, the tests run, and the Claude tokens spent building it.

## Claude Build Token Usage

Tokens consumed by **Claude Code** (the AI pair-programmer) to build the product,
covering both builds combined (the `/cost` report is a cumulative session total
and can't be cleanly split per build).

| Claude model | Input | Output | Cache read | Cache write | Cost |
|--------------|------:|-------:|-----------:|------------:|-----:|
| Opus 4.8 | 21.4k | 101.5k | 10.8M | 357.7k | $11.62 |
| Sonnet 4.6 | 1.2k | 56.7k | 12.0M | 462.1k | $7.23 |
| Haiku 4.5 | 46 | 1.6k | 395.0k | 98.4k | $0.24 |
| **Total** | **22.6k** | **159.8k** | **23.2M** | **918.2k** | **$19.09** |

**Session at a glance:**

- **Total cost:** $19.09
- **API time:** 3h 28m 48s · **wall time:** ~1d 7h
- **Code changes:** 2,895 lines added · 229 removed
- **Models used:** Opus 4.8 (primary), Sonnet 4.6, Haiku 4.5

---

# Initial Build — v0.1.0 (M1–M3)

The first working build: the full four-stage pipeline, implemented and unit-tested.

## Scope

| Stage | What it does |
|-------|--------------|
| 1 · Retrieve | Pull up to 50 candidate papers from arXiv for a topic |
| 2 · Rerank | LLM scores candidates by semantic relevance; keeps top 18 |
| 3 · Extract | Per-paper `{problem, method, results, contribution}`, concurrent, partial-failure tolerant |
| 4 · Synthesize | Cross-paper landscape: clusters, relationships, tensions, open problems |

- **Delivery:** single-server — FastAPI serves both `/api/*` and the static
  Alpine.js + Tailwind UI.
- **No Node, no build step** for the frontend.

## Models Used

| Role | Model |
|------|-------|
| Development | Claude Sonnet 4.6 |
| Rerank + Extract | `meta-llama/llama-3.3-70b-instruct:free` (OpenRouter) |
| Synthesis | `nvidia/nemotron-3-ultra-550b-a55b:free` (OpenRouter) |

## Tests Run

- **11 unit tests** across all four pipeline stages — **11 passed**.
- LLM and arXiv calls are **mocked**, so no live pipeline run in this build.

---

# Security & Rename Build — v0.1.1

Renamed the project to **Arxivore**, ran the first live end-to-end searches, and
reviewed the codebase against [`security.md`](security.md).

## Security Lapses Found

| # | Area (security.md) | Lapse |
|---|--------------------|-------|
| 1 | 3.2 Cost / token abuse | No **rate limiting** on `/api/search` — wallet-DoS risk |
| 2 | 3.2 Cost / token abuse | **Concurrency cap** (`MAX_CONCURRENT_RUNS`) defined but never enforced |
| 3 | 3.2 Cost / token abuse | **Spend ceiling** (`DAILY_TOKEN_BUDGET`) defined but no token accounting |
| 4 | 3.3 Prompt injection | System prompts didn't mark paper text as *data, not instructions* |
| 5 | 3.6 CORS & transport | **No security headers** (CSP / X-Content-Type-Options / Referrer-Policy) |
| 6 | 3.6 CORS & transport | Stale CORS default (`localhost:3000`, the dead Next.js dev server) |
| 7 | 3.7 Error handling | `rerank.py` could `500` on null model content (extract/synthesize were guarded) |

## Fixes Applied

| Lapse | Fix | File |
|-------|-----|------|
| 1 | Per-IP sliding-window limiter, `RATE_LIMIT_PER_MINUTE` (default 10/min) → `429` | `app/api/search.py` |
| 2 | `BoundedSemaphore(MAX_CONCURRENT_RUNS)` enforced → `503` when exceeded | `app/api/search.py` |
| 4 | System prompts now treat paper text as untrusted data, ignore embedded directives | `app/pipeline/*.py` |
| 5 | CSP, `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy` via middleware | `app/main.py` |
| 6 | CORS default corrected to same-origin host; no wildcard | `app/config.py` |
| 7 | Empty model content raises cleanly instead of crashing | `app/pipeline/rerank.py` |

### Still open (accepted for single-user v1)

- **Spend ceiling (#3)** — needs token accounting from `response.usage`. Bounded
  for now by candidate/token caps + rate limit + concurrency cap. Do before any
  public, multi-user exposure.
- **No CI** secret scanning / dependency audit (`gitleaks`, `pip-audit`).

## Models Used

| Role | Model |
|------|-------|
| Development | Claude Sonnet 4.6 → Opus 4.8 |
| Rerank + Extract + Synthesis | `openrouter/free` (OpenRouter free-tier routing) |

> **Why the switch:** the originally-pinned model IDs hit account data-policy /
> 404 errors. `openrouter/free` routes to any available free model under the
> account's privacy settings.

## Tests Run

- **Unit:** 11/11 passed after all changes.
- **Live end-to-end** (first real runs):

  | Topic | Retrieved | Ranked | Extracted | Landscape |
  |-------|----------:|-------:|----------:|-----------|
  | retrieval-augmented generation | 50 | 18 | 7/18 | synthesized |
  | diffusion policy learning | 50 | 18 | 16/18 | 7 clusters · 3 tensions · 5 open problems |

- **Security controls verified live:**
  - Security headers present on responses.
  - Rate limiter allows exactly 10/min per IP, then `429`, with per-IP isolation.

---

## Notes

- Free-tier models are rate-limited and slow (**~3–5 min per full run**); timings
  reflect the free tier, not a paid provider.
- Wiring `response.usage` through the three LLM stages is the next instrumentation
  target — it also unlocks enforcement of `DAILY_TOKEN_BUDGET` (lapse #3).

---

# Multi-Model Failover & Extract Batching — v0.1.2

Replaced the single-model LLM setup with a **hybrid failover pool system** and
**batched extraction**, reducing rate-limit failures and cutting extract token
usage by ~60%.

## Problem

Free-tier OpenRouter models have per-minute and daily rate limits. With a single
model pinned per stage, any rate-limit hit returned a generic `502` and aborted
the run. Extract called the LLM 18× (one paper per call), burning rate-limit
quota fast and paying the system prompt cost 18 times.

## What Shipped

### `backend/app/llm.py` — new shared LLM module

Single entry point for all LLM calls (previously each stage created its own
`OpenAI` client). Key behaviours:

- **One reused client** across all stages and threads — memory-optimal.
- **Hybrid failover:** each request carries OpenRouter's native `models[]` array
  (in-request fallback, no wasted output tokens if primary is rate-limited) **plus**
  an in-process per-model cooldown registry so a rate-limited model is skipped on
  subsequent calls without even attempting it.
- **Auto-cool on silent fallback:** when OpenRouter's inner fallback serves a
  response from a different model than requested, the primary is immediately cooled
  — so the next call goes straight to the working model without waiting for
  OpenRouter to re-route.
- **`AllModelsRateLimited` exception** raised when every model in a pool is
  unavailable, surfaced as HTTP `429` ("try again in a minute") at the API layer —
  distinct from the per-IP `429` and generic `502`.
- **`auto` discovery mode:** set `LLM_*_MODELS=auto` to fetch OpenRouter's free
  model catalog at runtime (cached 1 h); your account's allowed-models list gates
  which actually serve. True zero-touch when you add/remove models on OpenRouter.

### Two ordered failover pools

Models listed strongest-first. Position 0 is what runs on a healthy request —
accuracy is unchanged until rate-limit pressure forces failover.

| Pool | Models (in order) |
|------|------------------|
| Rerank + Extract | `meta-llama/llama-3.3-70b-instruct:free` → `openai/gpt-oss-120b:free` → `google/gemma-4-31b-it:free` → `nvidia/nemotron-3-nano-30b-a3b:free` → `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` |
| Synthesis | `nvidia/nemotron-3-ultra-550b-a55b:free` → `nvidia/nemotron-3-super-120b-a12b:free` → `nousresearch/hermes-3-llama-3.1-405b:free` → `openai/gpt-oss-120b:free` → `openrouter/owl-alpha` |

> All synthesis-pool models have ≥ 131K context — failover can never silently
> truncate the synthesis prompt (18 extracted papers).

> "Llama Nemotron Rerank VL 1B" (embedding-style reranker) was excluded — it is
> not a chat/JSON-completion model and cannot serve these prompts.

### Batched extraction (`backend/app/pipeline/extract.py`)

| Before | After |
|--------|-------|
| 1 paper per LLM call | 4 papers per LLM call |
| 18 calls per run | 5 calls per run |
| `_MAX_WORKERS = 2` (9 serial rounds) | `_MAX_WORKERS = 3` (2 serial rounds) |
| System prompt paid 18× | System prompt paid 5× |

Token savings: ~2,600 input tokens per run on system prompts alone (~60%
reduction on extract stage). Rate-limit pressure drops from 18 calls to 5
calls per run.

Batch failure isolation: if a batch fails (parse error or model error), the
papers in that batch are marked `extract_status = "error"` and the rest
continue — same per-paper resilience guarantee as before.

### `.bat` launcher

`start.bat` added to repo root — double-click to start the backend with one
click (activates `.venv`, runs `uvicorn --reload`, keeps the window open on
crash).

### OpenRouter constraint discovered

OpenRouter caps the native `models[]` fallback array at **3 models per
request**. The pool can be longer (cooldown memory spans calls), but only the
first 3 non-cooling models are sent per request.

## Config Changes

```env
### New in .env / .env.example
LLM_RERANK_MODELS=<comma-separated pool or "auto">
LLM_SYNTHESIS_MODELS=<comma-separated pool or "auto">
LLM_COOLDOWN_SECONDS=60
LLM_MODELS_CACHE_TTL=3600
```

Legacy `LLM_RERANK_MODEL` / `LLM_SYNTHESIS_MODEL` (single-model vars) are kept
as documented fallback defaults so existing `.env` files still boot.

## Models Used

| Role | Model |
|------|-------|
| Development | Claude Opus 4.8 / Sonnet 4.6 |
| Rerank + Extract (primary) | `meta-llama/llama-3.3-70b-instruct:free` |
| Rerank + Extract (failover) | `openai/gpt-oss-120b:free` (auto-triggered on rate limit) |
| Synthesis | `nvidia/nemotron-3-ultra-550b-a55b:free` |

## Tests Run

- **20 unit tests — 20 passed.**
- 9 new tests added: failover, cooldown, all-exhausted, auto-discovery (2 pool
  variants), OpenRouter silent-fallback cooling, per-stage mock repointing.
- **Live end-to-end** (first run with failover active):

  | Topic | Retrieved | Ranked | Extracted | Result |
  |-------|----------:|-------:|----------:|--------|
  | retrieval-augmented generation | 50 | 18 | 18/18 ✓ | synthesized |

  Failover triggered on nearly every extract call (llama rate-limited); gpt-oss-120b
  served all 18 papers. Extract time: ~105 s (pre-batching). With batching and
  auto-cooling: expected ~2–3× improvement on next run.

---

# Correctness & Instrumentation Build — v0.2.0

Audit-driven batch. Full findings list in [`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md)
(33 findings: 24 from code review, 6 from a DeepWiki docs audit, 3 from the live
OpenRouter catalog).

## Root cause found for the v0.1.1 model failures

`RELEASE.md` above recorded *"the originally-pinned model IDs hit account
data-policy / 404 errors"* without identifying which. Querying all 414 models in
the OpenRouter catalog settled it:

| Configured ID | Status |
|---|---|
| `meta-llama/llama-3.3-70b-instruct:free` | **withdrawn — does not exist** |
| `meta-llama/llama-3.3-70b-instruct` (paid) | exists, $0.10 / $0.32 per 1M |
| `nvidia/nemotron-3-ultra-550b-a55b:free` | valid, free, 1M context |

The `:free` variant of Llama 3.3 70B had been retired. Because `extract.py`
reused `LLM_RERANK_MODEL`, a fresh clone had a broken rerank **and** extract
stage while the docs still shipped the dead ID.

## Fixed

| # | Area | Issue | Fix |
|---|---|---|---|
| P0-1 | Correctness | `rerank.py` alone had no markdown-fence stripping, so a fenced response became a hard `502` that discarded the whole run | Shared `app/pipeline/_json.py`; all four stages now use one parse path |
| P0-2 | Config | Default rerank/extract model withdrawn from the provider catalog | Defaults → `nvidia/nemotron-3-super-120b-a12b:free`; opt-in `pytest -m live_models` asserts every configured ID still exists |
| P1-1 | Retrieval | Plain-English topic passed straight into arXiv's boolean API — the exact weakness `PRD.md` was written to solve | New stage 0 `expand.py`: one call → ~5 arXiv-syntax queries, unioned and deduped. Falls back to the raw topic, so it can only widen recall |
| P1-2 | Correctness | `.get()` with silent defaults meant a garbage response produced empty fields marked `extract_status="done"` | `RerankItem`, `ExtractionOut`, `ExpandOut` Pydantic contracts; failures become honest per-paper errors |
| P1-4 | Correctness | `{}` from synthesis validated as a successful-but-blank landscape | `Landscape.clusters` now requires ≥1 entry |
| P1-5 | Correctness | Prompt demanded grounded `arxiv_id`s and real cluster names; nothing verified it | Post-validation prunes invented ids and dangling relationship endpoints, and logs what it dropped |
| P1-6 | Testing | No way to tell whether a change helped | `backend/evals/` regression harness — offline mode, deterministic fixtures, committed baseline |
| P2-2 | Performance | `_MAX_WORKERS = 2` hardcoded while docs claimed 5 | `EXTRACT_CONCURRENCY` setting (default 3) |
| P2-3 | Observability | `response.usage` discarded by every stage, so `DAILY_TOKEN_BUDGET` was unenforceable | Usage threaded through all stages into `SearchResponse`; new `app/budget.py` ledger enforces the ceiling → `429`. **Closes security lapse #3** |
| P2-5 | Correctness | `arxiv_id` kept its `v2` suffix, so revisions were distinct ids | Canonical id strips the version; the versioned form still links to arXiv |
| P3-3 | Robustness | Rate-limiter dict grew one key per unique IP forever | Periodic sweep of inactive IPs |
| P5-6 | Config | Extraction silently reused the rerank model | Separate `LLM_EXTRACT_MODEL` / `LLM_EXPAND_MODEL`, as `PRD.md` §12 asked |

## Still open (accepted for single-user v1)

- **No CI** secret scanning / dependency audit (`gitleaks`, `pip-audit`).
- **No persistence** — SQLite cache + reading map (FR10/FR11) blocked on the
  cross-run dedup decision in `PRD.md` §12. `arxiv_id` normalisation landed now so
  the cache has a correct key when it is built.
- **Sync pipeline** — `POST /api/search` still blocks for the whole run; async +
  SSE (FR3/FR4) not yet done, so the `<90s` NFR is still missed on free models.
- **CDN supply chain** — Alpine loads from a floating `@3.x.x` tag with no
  integrity hash.

## Models Used

| Role | Model |
|------|-------|
| Development | Claude Sonnet 5 |
| Expand + Rerank + Extract | `nvidia/nemotron-3-super-120b-a12b:free` |
| Synthesis | `nvidia/nemotron-3-ultra-550b-a55b:free` |

All defaults remain free-tier. Switching rerank/extract/expand to
`openai/gpt-oss-120b` costs ~$0.0025/run (~30k in, ~9.4k out) and removes the
3–5 minute wall — documented in `.env.example` but not the default.

## Tests Run

- **54 unit tests, 54 passed** (up from 11), plus 1 opt-in live catalog check.
- New modules: `test_json.py`, `test_expand.py`, `test_budget.py`,
  `test_search_api.py`, `test_config.py`.
- New regression coverage for each fixed defect: fenced rerank JSON, off-schema
  extraction, empty landscape, hallucinated cross-references, `arxiv_id` version
  stripping, multi-query dedup, rate-limiter sweep, budget `429`.
- **Offline eval** — `python -m evals.run --offline`: 3 topics, 50 candidates each,
  18/18 extraction, 2 clusters, byte-identical token totals across runs.

> Determinism note: the fixtures use `zlib.crc32` rather than `hash()`. Python
> randomises string hashing per process, which made offline token counts drift
> between runs and left the baseline undiffable.

---

# Remaining-Backlog Batch — v0.3.0

Closes the rest of `IMPROVEMENT_PLAN.md`'s findings that were still worth
doing, given two constraints from this session: the UI is now a separate
Gradio (Hugging Face Space) front end calling this backend, so UI-side work
(P3-2, P4-3, P4-4) and deployment-topology work depending on an undecided
CORS/proxy setup (P3-4) were explicitly left as-is. Persistence dedup was
resolved: one row per paper (`arxiv_id` primary key), a join table tags which
runs surfaced which papers.

## Closed

| # | Area | What changed |
|---|---|---|
| P4-1, P3-5, P3-6 | Hygiene | `.github/workflows/ci.yml` (pytest, pip-audit, gitleaks); `requirements.txt` pinned to exact verified versions; `PATCH` dropped from CORS methods until a PATCH route existed |
| P2-6 | Resilience | `LLM_FALLBACK_MODEL`: a 429 (`RateLimitExceeded`) retries once against a fallback instead of burning the OpenAI client's own backoff against a daily quota that can't recover mid-run — the likely real cause of the recorded 7/18-vs-16/18 extraction gap |
| P1-3, P1-7 | Retrieval quality | Rerank batches candidates in groups of 15 (concurrent), reducing per-call prompt size and truncation risk; `SearchRequest.published_after` filters out papers older than a cutoff, since `SortCriterion.Relevance` alone let old papers outrank new SOTA |
| P2-1, P5-1, P5-4, P5-3, P2-5 | Persistence | SQLite-backed reading map (`storage.py`): a paper's extraction is cached once and reused by every later run that retrieves it (proven end-to-end in `test_service.py`) — this is where the token savings actually land. 5 of 6 PRD API endpoints now implemented (`GET /api/runs`, `/{id}`, `/{id}/papers`, `/{id}/landscape`, `PATCH /api/papers/{id}`) |
| P3-1, P5-2 | Latency (UX) | `POST /api/search` no longer blocks for the full 3–5 min run — returns `{run_id}` immediately; `GET /api/runs/{id}/stream` (SSE) streams the `ARCHITECTURE.md` state machine live. **Breaking change** to `POST /api/search`'s response shape |
| P2-4 | Token efficiency | Embedding prefilter before rerank — **opt-in**, off by default. OpenRouter's `/embeddings` coverage is unverified against the configured model, so any failure falls back to unfiltered candidates automatically |
| P5-7 | Extraction depth | Full-text PDF excerpt for the top `FULL_TEXT_TOP_N` papers by rerank score — **opt-in**, off by default (adds real per-paper latency; would otherwise make the offline eval harness hit arxiv.org for synthetic ids) |

## Still open (deliberately, this session)

- P3-2, P4-3, P4-4 — UI-side; dead work now that Gradio is the front end.
- P3-4 — depends on whether the Gradio Space ends up calling this API directly
  (needs CORS) or through a proxy; deployment topology explicitly not decided.
- P5-5 — evaluated, not applicable: `synthesize_landscape` only ever ingests
  the current run's `<=18` papers, already bounded. Nothing to cap until a
  cross-run/whole-map synthesis feature exists.
- P5-8 (Next.js) — lowest value now; UI is Gradio.

## Two deviations from the original schema sketches

Both documented in `storage.py`'s module docstring:

1. `read_status` lives on `papers`, not `run_papers` — `PATCH
   /api/papers/{arxiv_id}` has no `run_id` to resolve which row to update, so
   read state has to be global to the paper.
2. Embeddings live in their own table, not a column on `papers` — the
   prefilter runs before extraction, before a candidate has ever earned a
   `papers` row.

## Tests Run

- **129 unit tests, 129 passed** (up from 54), plus 1 opt-in live catalog
  check. New modules: `test_storage.py`, `test_run_status.py`,
  `test_runs_api.py`, `test_service.py`, `test_embed.py`, `test_fulltext.py`.
- Every new feature has an explicit "falls back safely / off by default"
  regression test — rate-limit fallback, all-batches-failing rerank,
  cache-hit-skips-LLM, embed-prefilter-disabled, full-text-fetch-failure.
- **Offline eval**, `python -m evals.run --offline`: baseline refreshed twice
  (rerank batching changed the token shape; persistence caching reduced it via
  genuine cross-topic reuse in the fixture data) and unchanged for both
  opt-in features, since neither is exercised while disabled.

---

# Static UI Hardening — v0.4.0

Closes P3-2, P4-3, P4-4 and — found in the course of fixing them — repairs a
regression the v0.3.0 async-pipeline change introduced but never applied to
`backend/app/static/index.html`.

## The regression this batch actually opened with

`POST /api/search` had already changed to return `{run_id, state}`
immediately instead of the full result (v0.3.0), but the static page's
`search()` still did `this.result = await res.json()` expecting the old
synchronous `SearchResponse`. **The static UI was non-functional on this
branch** independent of P3-2/P4-3/P4-4 — fixed as part of this batch rather
than left as a separate surprise.

## Closed

| # | What changed |
|---|---|
| — (regression) | `static/app.js` (new): `search()` now follows the real flow — `POST /api/search` → `{run_id}` → subscribe to `GET /api/runs/{run_id}/stream` (SSE) for live stage progress → on `COMPLETE`, fetch `GET /api/runs/{run_id}` (the storage-backed result) and render. The stats bar's `candidates_retrieved`/`papers_returned`/`extract_errors` fields don't exist on the storage-backed `RunDetail` (only on the old `SearchResponse`) — replaced with `papers.length` and a client-side count of `extract_status === "error"` |
| P3-2 | A `setTimeout` ceiling (8 min — above the documented 3–5 min free-tier run time) closes the SSE stream and shows a timeout error if no terminal state arrives; a separate `AbortController` bounds the initial POST itself |
| P4-4 | Alpine (3.16.2) and the Tailwind Play script (3.4.17) vendored locally under `static/vendor/` — pinned exact versions, no CDN network dependency, no floating tag to drift on |
| P4-3 | CSP: both external CDN origins dropped; `unsafe-inline` dropped from `script-src` (the page's JS is now external `app.js`, not an inline block). `unsafe-eval` (Alpine's `new Function()`-based expression evaluation) and `style-src`'s `unsafe-inline` (Tailwind Play's runtime JIT style injection) stay — both are architectural to the tools themselves, not a CDN-loading artifact, and can't be dropped without a real build step that would contradict this UI's "no Node build" design |

## A bug found by actually loading the page, not by reading the diff

Browser-console verification (not just `curl` status codes) caught a real
defer-ordering bug `curl` could never see: Alpine's `<script defer>` sits in
`<head>`, and deferred scripts execute in **document order**, not head/body
position. With `app.js`'s tag at the bottom of `<body>`, Alpine's script ran
*first* and called `mapper()` before it existed — `ReferenceError: mapper is
not defined`, cascading into every Alpine-bound expression on the page.
Fixed by moving `app.js`'s tag before Alpine's in document order.

Also fixed while verifying this exact region: `<template x-for="p in
result.papers">` had no null guard, unlike its sibling `x-show="result?.
papers?.length"` right next to it — threw on every initial page load (before
any search) since `result` starts `null`. One-token fix:
`x-for="p in (result?.papers ?? [])"`.

## Still open (deliberately)

- **P3-4** (trusted `X-Forwarded-For`) — depends on whether the Gradio Space
  calls this API directly or through a proxy; deployment topology still
  undecided.
- **P5-8** (Next.js) — UI framework choice; dead work now that Gradio is the
  real front end.

## Tests Run

- **129 unit tests, 129 passed** — unchanged; this batch touched only static
  assets (`index.html`, `app.js`, `vendor/`) and `main.py`'s CSP string, no
  Python logic.
- **Live browser verification** (not just `pytest`): server started locally,
  page loaded in a real Chrome tab via `claude-in-chrome`, console read for
  errors before and after each fix. This is what caught both bugs above —
  neither would show up in `curl -I` status-code checks or in a diff review.
- Vendor assets and `app.js` confirmed serving `200` from the local static
  mount (not a CDN redirect) via direct request.

---

# Branch Alignment & Production Hardening — v0.5.0

The two branches had drifted apart, and the live Space had been quietly broken
for months. This build fixes both, and settles what each branch is *for*.

## Branch roles, finally explicit

| Branch | Purpose |
|---|---|
| `main` | the full backend — run locally, deploy to GCP/AWS. FastAPI + OpenRouter, SQLite persistence, SSE progress, embedding prefilter and full-text extraction |
| `hf-spaces-prototype` | the free-tier Gradio demo at [`kawas8516/arxivore`](https://huggingface.co/spaces/kawas8516/arxivore) — same four-stage pipeline, sized for a public CPU Space and for recruiters to click |

They share the pipeline and the LLM transport. They differ only where the
deployment target genuinely differs. Before this build they differed everywhere,
by accident.

## 1 · The Space was broken, and nothing said so

`microsoft/Phi-4-mini-instruct` had been the Space's model since it was written.
It was never served by any provider — and the `-mini-instruct` variant does not
exist at all (plain `microsoft/phi-4` does).

The Space **built green and ran green**. Retrieve worked. Rerank worked, because
it is a local CPU cross-encoder that never calls an API. Extract and Synthesize
failed on every single call, and the only symptom was two empty tabs.

Now `google/gemma-3-12b-it`: served, returns valid JSON, small enough to stay
cheap on free-tier credits.

## 2 · A withdrawn model id killed the backend too

Five of nine configured OpenRouter ids had been withdrawn from the catalog,
including **position 0 of the rerank pool** — the model every healthy request
hit first.

The failover pools did not help, because an unknown id returns **400, not 429**,
and `complete()` only descended the pool on a rate limit or a 5xx. `app/llm.py`
now fails over on an unservable id too, dropping only the ids the provider names
and parking them for the catalog TTL rather than the 60-second rate-limit
cooldown — a retired model is not coming back in a minute.

## 3 · So model ids are now checked, not assumed

`scripts/check_models.py` validates every id the project can dial, against the
right provider, and exits non-zero so it can gate a release.

```bash
python scripts/check_models.py            # both providers
python scripts/check_models.py --suggest  # live replacements for anything dead
```

The HF check **calls** the model rather than looking it up: catalog membership
is not the property that matters, being servable *for this account* is — and
that is precisely the distinction Phi-4-mini fell through.

## 4 · HF force-upgraded the Space SDK mid-flight

Hugging Face bumped the Space from Gradio 5.9.1 to 6.26.0 on its own. Gradio 6
moved `theme` off the `Blocks` constructor onto `launch()`, and signals removed
arguments with a **UserWarning, not an error** — so the Space kept serving while
silently dropping its theme and rendering unstyled.

Fixed, then pinned: `gradio==6.26.0` exactly, and `app.py` now checks the Gradio
major it actually loaded, logging ERROR *and* rendering a banner into the page
header on a mismatch. A Space log nobody opens is how this got missed once.

## 5 · One LLM transport, on both branches

Both branches had independently grown failover. `main` retried once against a
single `LLM_FALLBACK_MODEL`; the prototype had ordered pools with cooldowns and
runtime catalog discovery. The pools won.

`app/llm.py` is now the only thing that issues an LLM request. `_json.py` keeps
parsing, validation, and token accounting, and imports nothing from `app` — which
is what lets `app.llm` build on it without a cycle. `complete()` returns
`(content, prompt_tokens, completion_tokens)`, because `budget.add()` cannot
count what a bare `str` return threw away.

## 6 · Repo layout now mirrors the Space

`hf_space/` was promoted to the repo root as pure renames, so the branch and the
Space repo can be diffed with `git cat-file -s` instead of the HF API. All nine
code files are byte-identical to what is deployed.

> **Watch the line endings.** This repo runs `core.autocrlf=true`, so working-tree
> files are CRLF while the committed blobs are LF. Copying a worktree file into
> the Space pushes CRLF and silently breaks the mirror. Stage from the blob
> (`git show HEAD:<path>`), never from the checkout.

## What each branch kept

Left off the Space branch, deliberately — a PDF fetch per paper and an extra
embedding call per run do not suit a free-tier demo:

| Not on the Space branch | Why |
|---|---|
| `pipeline/embed.py` | OpenRouter's `/embeddings` coverage is thin; costs a call per run |
| `pipeline/fulltext.py` | a PDF fetch and parse per paper, plus a `pypdf` dependency |

Both are default-off on `main`, so excluding them changes no behavior. Bringing
`main` to parity therefore had to *re-apply* their wiring on top of the new
transport rather than merge over it.

## Models

| Where | Stage | Model |
|---|---|---|
| Backend | synthesize | `nemotron-3-ultra-550b-a55b:free` → 3 more in pool |
| Backend | rerank · extract · expand | `gemma-4-31b-it:free` → 3 more in pool |
| Space | extract · synthesize | `google/gemma-3-12b-it` (HF Inference API) |
| Space | rerank | `BAAI/bge-reranker-v2-m3` (local CPU, no API, no quota) |

Either backend pool also accepts the literal `auto`, which discovers free models
from OpenRouter's catalog at runtime.

## Tests

**138 passing, 1 skipped** on `main`; **127 passing** on `hf-spaces-prototype`.
The suite runs identically from the repo root and from `backend/` — that
dual-cwd behaviour is what root-level `pytest.ini` exists for.

| Area | Coverage |
|---|---|
| `test_llm.py` (16) | pool failover, cooldown registry, all-exhausted, auto-discovery, OpenRouter silent-fallback cooling, JSON mode, the 400-only retry without `response_format`, schema rejection, and failover past a withdrawn id |
| `test_config.py` | asserts on **shipped defaults**, not `get_settings()` — reading loaded config is how a developer's own `.env` masked retired ids from everyone else |
| `test_rerank.py` (11) | batching at 15/group, partial-batch failure, all-batches-fail raising, token accounting, pool failover |
| `test_extract.py` (12) | per-paper resilience, extraction cache hits, storage write-through and its failure path |
| `test_service.py` | full pipeline orchestration; its three per-module mocks collapsed into one client dispatching on the system prompt, since the stages no longer have separate clients to mock apart |
| `test_embed.py` · `test_fulltext.py` | `main` only — the opt-in features the Space branch does not carry |

Two opt-in network checks, excluded by default:

```bash
pytest -m live_models
```

They verify every configured OpenRouter id against the live catalog *and* call
the Space's HF model, rejecting empty content — thinking models return `None`
and would break the JSON-only extraction prompt. On `main` the Space tests skip
themselves, because `main` ships no Gradio app and "no HF models here" is the
correct answer rather than a failure.

**Verified end to end** against the live HF API through the Space's own code
path: 2/2 papers extracted in 4.2 s, synthesis returned 2 clusters,
2 relationships, 1 tension, 3 open problems. The Space is `RUNNING` on
Gradio 6.26.0 / Python 3.11.
