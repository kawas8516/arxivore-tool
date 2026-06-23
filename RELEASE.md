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
