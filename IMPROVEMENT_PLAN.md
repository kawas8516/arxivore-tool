# Improvement Plan — Arxivore

Working backlog for making the project **more efficient, robust, optimized, and
token-efficient**. Every row cites a real `file:line` so a fix can start without
re-investigation.

**Status:** the audit below is the original snapshot and is left unedited for
history. See `RELEASE.md` v0.2.0 through v0.4.0 for what's actually landed —
**19 of the original 33 findings are now closed**: P0-1, P0-2, P1-1, P1-2,
P1-3, P1-4, P1-5, P1-6, P1-7, P2-1, P2-2, P2-3, P2-4, P2-5, P2-6, P3-1, P3-2,
P3-3, P3-5, P3-6, P4-1, P4-2, P4-3, P4-4, P5-1, P5-2, P5-3, P5-4, P5-6, P5-7.
(P4-2 — spend-ceiling enforcement — was actually closed in the very first
batch, before the session that produced this plan's other tracking; it was
never marked here until this pass caught the gap.) P5-5 was evaluated and
found not-yet-applicable (no cross-run synthesis feature exists yet to bound —
see the Phase 1 note in `RELEASE.md`). Remaining open, by explicit decision
this session (deployment topology and UI framework choice are still not
decisions to make in-repo): **P3-4** (trusted X-Forwarded-For — depends on
whether the Gradio Space ends up calling this API directly or through a
proxy, undecided) and **P5-8** (Next.js migration — dead work now that
Gradio is the real front end).

**How findings were sourced:**

| Source | Method | Findings | Blind spot |
|--------|--------|---------:|------------|
| Code review (this repo, all 12 `.py` + `index.html`) | Read implementation | 24 | Skipped docs — missed empirical run data + CI gap |
| DeepWiki (Cognition) | Read `*.md` only | 6 | Zero code citations; can only surface gaps already documented |
| OpenRouter model catalog (live API) | `GET /api/v1/models`, 414 models | 3 | Key-specific access unverified (see §8) |

Overlap between code review and DeepWiki: **zero**. Both were needed.

---

## 0. Scorecard

| Area | State | Worst offender |
|------|-------|----------------|
| Correctness | 🔴 2 latent crash/corruption bugs | `rerank.py:66` |
| Retrieval quality | 🔴 raw NL query into a boolean API | `retrieve.py:13-17` |
| Config validity | 🔴 default rerank model does not exist | `config.py:17` |
| Token efficiency | 🟠 no caching, whole pipeline re-runs every time | `service.py:21` |
| Latency | 🔴 3–5 min vs `<90s` NFR target | `PRD.md:103-104` vs `RELEASE.md:124` |
| Robustness | 🟠 partial-failure handling good, retries naive | `extract.py:96-104` ✅ / `:86` 🟠 |
| Persistence | 🔴 none; `database_url` configured but unused | `config.py:33` |
| Security posture | 🟢 headers/CORS/limits done · 🟠 no CI scanning | `main.py:50-57` ✅ |
| XSS safety | 🟢 `x-text` throughout, no `x-html` | `index.html` ✅ |
| Observability | 🟠 timings logged, tokens never | `service.py:66-79` |
| Test coverage | 🟠 11 unit tests, all mocked; no eval set | `backend/tests/` |

---

## 1. P0 — Correctness blockers

Fix before anything else. Both are live defects, documented nowhere.

| # | Finding | File:line | Why it breaks | Fix |
|---|---------|-----------|---------------|-----|
| P0-1 | **Rerank has no markdown-fence stripping.** `extract.py:43-49` and `synthesize.py:48-53` both define `_strip_fences`; `rerank.py` defines none and calls `json.loads(content.strip())` directly. | `rerank.py:66-67` | Any model that wraps JSON in ```` ```json ```` → `JSONDecodeError` → `PipelineError("rerank")` at `service.py:46-50` → **hard 502, entire run lost**. Rerank is a hard-fail stage, so there is no partial-result fallback. | Extract `_strip_fences` into `app/pipeline/_json.py`, import in all three stages. Removes the duplication at `extract.py:43` / `synthesize.py:48` at the same time. |
| P0-2 | **Default rerank model does not exist.** `meta-llama/llama-3.3-70b-instruct:free` is absent from the OpenRouter catalog (414 models checked). Only the **paid** `meta-llama/llama-3.3-70b-instruct` exists. | `config.py:17`, `.env.example:11` | This is the exact cause of the "404 / data-policy errors" recorded at `RELEASE.md:102-104`. A fresh clone using shipped defaults gets a broken rerank **and** extract stage (extract reuses `llm_rerank_model` — see `extract.py:93`). Synthesis default `nvidia/nemotron-3-ultra-550b-a55b:free` **is** still valid. | Change default to a model verified present in §8. Add a startup existence check against `GET /models`. |

### P0-2 evidence

```
meta-llama/llama-3.3-70b-instruct:free       → MISSING
meta-llama/llama-3.3-70b-instruct            → present, $0.10/M in, $0.32/M out
nvidia/nemotron-3-ultra-550b-a55b:free       → present, free, 1M ctx
openrouter/free                              → present, free, 200k ctx
```

---

## 2. P1 — Output quality

| # | Finding | File:line | Why it matters | Fix |
|---|---------|-----------|----------------|-----|
| P1-1 | **Raw natural-language topic passed straight to arXiv.** `arxiv.Search(query=topic)` — arXiv's API is field-based boolean (`all:`, `ti:`, `abs:`, `AND`/`OR`). | `retrieve.py:13-17` | `PRD.md:21-22` names the problem: *"keyword search on arXiv returns hundreds of weakly-ordered hits; relevance ranking is poor."* The code then **reproduces that exact problem** by feeding English into a boolean engine. Every downstream stage inherits this recall ceiling — the best reranker cannot recover a paper retrieval never returned. **Single highest-leverage fix in this document.** | Add stage 0 `expand_query`: one cheap LLM call → 4-6 arXiv-syntax queries (synonyms, method names, `cat:` filters). Union results, dedupe on `arxiv_id`. |
| P1-2 | **No schema validation on rerank or extract output.** `synthesize.py:103` correctly calls `Landscape.model_validate(data)`. Rerank and extract do not — they use `.get()` with silent defaults. | `rerank.py:71-74`, `extract.py:72-75` | `CLAUDE.md` mandates *"validate/parse LLM output against strict schemas."* Currently a garbage response yields `problem=""`, `method=""` and `extract_status="done"` — a **silently empty card marked successful**. Worse than an error. | Add `RerankItem` and `ExtractionOut` Pydantic models; validate. Malformed output becomes an honest per-paper `extract_status="error"`. |
| P1-3 | **Rerank prompt can truncate silently.** 50 abstracts in, 50 objects + rationales out, `max_tokens=4096`. | `rerank.py:49`, `:41-44` | ~15k input tokens; 50 scored objects with 20-word rationales does not reliably fit 4096 output. Truncated JSON → P0-1 crash. If a paper is merely *omitted*, `score_map.get` at `rerank.py:71` leaves `relevance_score=None`, which `rerank.py:76-79` coerces to `0.0` and sorts to the bottom — **relevant papers silently discarded, nothing logged**. | Batch 15 candidates/call, or drop `rationale` from the rerank schema. Log any `arxiv_id` the model failed to score. |
| P1-4 | **Empty landscape validates as success.** `Landscape` fields all default to `[]` (`models.py:55-58`), so a model returning `{}` passes `model_validate` and renders as a blank landscape with no error. | `synthesize.py:102-103`, `models.py:54-58` | User sees an empty page, logs say `synthesized=True` (`service.py:74`). Undiagnosable. | Require `min_length=1` on `clusters`, or raise when `clusters` is empty. |
| P1-5 | **Landscape cross-references never verified.** Prompt at `synthesize.py:43` demands *"Every arxiv_id you use must come from the papers given"* — code never checks. `Relationship.from_cluster/to_cluster` are free strings (`models.py:48-49`). | `synthesize.py:102-103`, `models.py:44-49` | LLM can invent `arxiv_id`s and reference non-existent cluster names. UI at `index.html:109-111` renders them as real. Prompt-level constraint with no code-level enforcement is not a constraint. | Post-validate: drop `arxiv_ids` not in the input set; drop relationships whose endpoints aren't in `clusters`. |
| P1-6 | **No evaluation harness.** `PRD.md:140` lists *"rerank quality: human spot-check agreement on top-10"* as a success metric. No mechanism exists. | `backend/tests/` (absent) | Every fix in P1/P2 is unverifiable. You cannot tell whether query expansion helped or hurt. **Unmeasurable = unimprovable.** Build this before optimizing, not after. | 15-20 fixed topics checked into repo, ~10 hand-labelled relevant papers each. Metrics: `recall@50` post-retrieve, `precision@18` post-rerank. Run before/after every prompt change. |
| P1-7 | **No recency control.** `SortCriterion.Relevance` only; no date window, no `cat:` filter. | `retrieve.py:16`, `:13-17` | A 2016 paper outranks 2025 SOTA. For a field-orientation tool this is a correctness issue, not a preference. | Optional `since` param; default `cat:cs.LG OR cat:cs.CL OR cat:cs.AI`. |

---

## 3. P2 — Token & cost efficiency

Stated goal: token-efficient. Ranked by tokens saved per unit of work.

| # | Finding | File:line | Token cost today | Fix / saving |
|---|---------|-----------|------------------|--------------|
| P2-1 | **No caching. Nothing persists.** `run_pipeline` returns `SearchResponse` and discards everything. `database_url` is configured and never read. | `service.py:21-91`, `config.py:33` | Every run re-extracts every paper from scratch. Two searches over overlapping fields pay twice for identical work. **Abstracts are immutable — this is pure waste.** | SQLite cache `arxiv_id → extraction`. On overlapping topics saves ~100% of extract tokens for repeat papers. Same table implements FR10/FR11 reading map (`PRD.md:97-99`) — **one schema, two features**. Resolve dedup strategy (`PRD.md:155`) first. |
| P2-2 | **`_MAX_WORKERS = 2`** while README claims 5 concurrent. | `extract.py:13` | 18 papers ÷ 2 = **9 sequential rounds**. Self-inflicted share of the 3–5 min runtime that `RELEASE.md:124` attributes solely to free-tier limits. | Raise to 5. One-line latency win, zero token cost. Gate on P0-2 (needs a model that won't rate-limit). |
| P2-3 | **`response.usage` never read.** All three stages discard it. | `rerank.py:61-63`, `extract.py:67`, `synthesize.py:96-98` | `DAILY_TOKEN_BUDGET` (`config.py:22`) is unenforceable — open security lapse #3 (`RELEASE.md:71`, `:90-92`). `PRD.md:108` requires *"every stage logs timing, token usage"*; only timing is logged (`service.py:66-79`). | Return `usage` from each stage; accumulate; log per stage; refuse new runs at ceiling. Closes the last v1 security gap. |
| P2-4 | **Full abstracts sent to rerank.** | `rerank.py:41-44` | 50 × ~250 words ≈ 15k tokens per run just to *order* candidates. | Embedding prefilter: embed topic + abstracts, cosine, cut 50→20 before the LLM sees anything. Cuts rerank prompt ~60%, fixes P1-3, and embeddings cache permanently per `arxiv_id`. |
| P2-5 | **`arxiv_id` retains version suffix.** `entry_id.split("/")[-1]` yields `2301.12345v2`. | `retrieve.py:23` | `v1` and `v2` of one paper are distinct cache keys → duplicate extraction, duplicate spend, duplicate cards. Breaks P2-1 before it ships. | Strip trailing `v\d+` for the cache key; keep the versioned id for display/URL. |
| P2-6 | **Naive retry strategy.** `max_retries=5` on default backoff. | `extract.py:86`, `rerank.py:38`, `synthesize.py:64` | Client-side retry cannot beat an exhausted **daily** free-tier quota. This is the real driver of the 7/18 extraction result at `RELEASE.md:111-114` — not transient failure. Retrying a hard quota wall just burns wall-clock. | Detect `429` + `Retry-After`; fall back to a secondary model rather than retrying the exhausted one. Combined with P2-1 caching, quota pressure mostly disappears. |

### Cost math — why free tier is the expensive option

Approximate tokens for one default run (50 candidates → 18 retained):

| Stage | Input | Output |
|-------|------:|-------:|
| Rerank | ~15,000 | ~2,000 |
| Extract (18 papers) | ~9,000 | ~5,400 |
| Synthesize | ~6,000 | ~2,000 |
| **Total** | **~30,000** | **~9,400** |

| Option | Cost / run | 100 runs | Wall time |
|--------|-----------:|---------:|-----------|
| Free tier (today) | $0.00 | $0.00 | **3–5 min** (`RELEASE.md:124`) |
| `openai/gpt-oss-120b` ($0.03 / $0.17) | ~$0.0025 | ~$0.25 | seconds |
| `meta-llama/llama-3.3-70b-instruct` ($0.10 / $0.32) | ~$0.0060 | ~$0.60 | seconds |

**Verdict:** the free tier is not saving money, it is spending 3–5 minutes of
user time per run to avoid ~half a cent. DeepWiki's "move to a paid tier"
recommendation is directionally right but reaches for the credit card before
the free wins — do **P2-1 caching** and **P2-2 concurrency** first, then spend
$0.25 to erase the rest.

---

## 4. P3 — Robustness

| # | Finding | File:line | Risk | Fix |
|---|---------|-----------|------|-----|
| P3-1 | **Synchronous blocking pipeline.** `def run_pipeline` (sync) called directly in the request handler. | `service.py:21`, `api/search.py:44`, `:54` | One request occupies a threadpool worker for 3–5 min. Violates FR3/FR4 (`PRD.md:85-86`) and the `<90s` NFR (`PRD.md:103-104`). User stares at a dead page, then everything appears at once. | `POST /search` → `run_id` immediately; background task; SSE stream (`PRD.md:130`). Papers appear as they extract. |
| P3-2 | **No client-side timeout.** `fetch` with no `AbortController`. | `index.html:200-204` | Against a 3–5 min backend, a stalled run hangs the UI indefinitely with a spinner and no recourse. | `AbortController` + explicit timeout + retry affordance. Largely moot once P3-1 lands. |
| P3-3 | **Rate-limiter dict grows without bound.** `_ip_hits` is a `defaultdict(deque)`; empty deques are pruned of entries but the **keys are never deleted**. | `api/search.py:24`, `:31-40` | One key per unique IP, forever. Slow memory leak; trivially amplified by spoofed source IPs. | Drop the key when its deque empties; or periodic sweep. |
| P3-4 | **Client IP taken raw from socket.** No `X-Forwarded-For` handling. | `api/search.py:45` | Behind any proxy/CDN every client collapses to one IP — the rate limit becomes global and one user locks out everybody. | Trust `X-Forwarded-For` only from a configured proxy allowlist. |
| P3-5 | **Unpinned dependencies, no lockfile.** All `>=` constraints. | `backend/requirements.txt:1-8` | Non-reproducible builds; a breaking minor release lands silently. Contradicts `PRD.md:109` reproducibility goal. | Pin exact versions + `requirements.lock`. |
| P3-6 | **`allow_methods` includes `PATCH` with no PATCH route.** | `main.py:46` | Harmless today, but signals the unimplemented `PATCH /api/papers/{id}` (`PRD.md:134`). Remove or implement. | Tighten to actual methods. |

---

## 5. P4 — Security & CI

Live controls are in good shape. Gaps are process-level.

| # | Finding | File:line | Status | Fix |
|---|---------|-----------|--------|-----|
| P4-1 | **No CI secret scanning or dependency audit** (`gitleaks`, `pip-audit`). No `.github/` directory exists. | `RELEASE.md:93` | Open, accepted for v1 | GitHub Actions: `gitleaks` + `pip-audit` + `pytest` on push. Public repo — worth the 20 minutes. *(Found by DeepWiki; code review missed it.)* |
| P4-2 | **Spend ceiling unenforced** — see P2-3. | `config.py:22` | Open lapse #3 (`RELEASE.md:71`) | Blocked on P2-3. Must land before any multi-user exposure. |
| P4-3 | **CSP permits `unsafe-inline` + `unsafe-eval` + two external CDNs.** | `main.py:17-26`, `:19-21` | Accepted (Alpine.js requires eval) | Vendor Alpine + a built Tailwind stylesheet locally → drop `unsafe-eval` and both CDN origins from `script-src`/`style-src`. Also retires the Tailwind **Play** CDN (`index.html:8`), which is explicitly not for production use, and closes P4-4. |
| P4-4 | **CDN assets unpinned and unverified.** Tailwind Play CDN at `index.html:8`; Alpine at `index.html:9` using a **floating tag** — `alpinejs@3.x.x` — with no `integrity` on either. | `index.html:8-9` | Open | `3.x.x` resolves to *whatever 3.x is latest at page load*: arbitrary third-party JS executes in the app origin, and a compromised or broken release ships to users with no repo change. Pin an exact version + add `integrity`/`crossorigin`, or vendor locally per P4-3 (preferred — kills P4-3 and P4-4 together). |

---

## 6. P5 — Architecture gaps (planned, not built)

| # | Gap | Spec | Reality | Priority |
|---|-----|------|---------|----------|
| P5-1 | Persistence / reading map (FR10, FR11) | `PRD.md:97-99`, `:119-120` | Nothing stored; `config.py:33` unused | High — see P2-1 |
| P5-2 | Async run + SSE progress (FR2-FR5) | `PRD.md:84-87`, `:130` | Sync blocking (`service.py:21`) | High — see P3-1 |
| P5-3 | 6 of 7 documented API endpoints missing | `PRD.md:126-134` | Only `POST /api/search` (`api/search.py:43`) | Medium |
| P5-4 | Dedup across overlapping runs — unresolved | `PRD.md:155` | Undecided | **Blocks P2-1** — decide first |
| P5-5 | Bounding synthesis token cost as map grows | `PRD.md:156` | Undecided | Medium |
| P5-6 | Per-stage model split (cheap rerank / strong synthesis) | `PRD.md:153-154` | Partially done: `config.py:16-17` splits them, but `extract.py:93` reuses the *rerank* model | Medium — see §8 |
| P5-7 | Full-text extraction | `PRD.md:46` (v1 non-goal) | Abstract-only (`extract.py:60-64`) | Low, high payoff — abstracts rarely contain numbers, so the `results` field is weak across the board. Pull LaTeX/PDF for top ~5. |
| P5-8 | Next.js frontend | `PRD.md:115-116`, `CLAUDE.md` | Static Alpine.js SPA (`RELEASE.md:41-43`) | Lowest — current UI works and is XSS-safe. Only migrate when the reading map needs real routing. |

---

## 7. Doc ↔ code drift

Public-repo credibility issues. All cheap to fix.

| Claim | Where | Reality | Action |
|-------|-------|---------|--------|
| 7 API endpoints incl. SSE, `/api/runs`, `PATCH /api/papers` | `README.md` API table, `PRD.md:126-134` | Only `POST /api/search` exists (`api/search.py:43`) | Mark planned rows explicitly |
| "extracted concurrently… 5 concurrent" | `README.md` features + pipeline table | `_MAX_WORKERS = 2` (`extract.py:13`) | Fix code (P2-2), not the doc |
| "An Anthropic API key" in prerequisites | `README.md` Prerequisites | Uses OpenRouter (`config.py:15`); instructions two lines later say `sk-or-...` | Correct to OpenRouter |
| `git clone` → `cd patch-search` | `README.md` Quick start | Repo is `arxivore-tool` | Fix path |
| Rerank model `llama-3.3-70b-instruct:free` | `config.py:17`, `.env.example:11`, `RELEASE.md:50` | **Does not exist** (§8) | P0-2 |
| "Rerank + Extract + Synthesis → `openrouter/free`" | `RELEASE.md:100` | `config.py:16-17` still ships the dead pinned IDs | Reconcile |
| Frontend: Next.js + Tailwind | `PRD.md:115-116`, `CLAUDE.md` Stack | Alpine.js static page | Mark as planned |

---

## 8. Model availability — live OpenRouter catalog

Queried `GET https://openrouter.ai/api/v1/models` (public endpoint, no key sent):
**414 models total, 17 free.**

> **Note on key-specific access:** reading `LLM_API_KEY` out of `.env` to call
> the authenticated `/api/v1/key` endpoint was blocked by a local safety
> classifier, so per-key entitlements are **unverified**. Free-model
> availability also depends on your account's privacy/data-policy setting —
> exactly the failure mode recorded at `RELEASE.md:102-104`. To check your own
> tier, run this yourself (never commit the output):
> ```bash
> curl -s -H "Authorization: Bearer $LLM_API_KEY" https://openrouter.ai/api/v1/key
> ```

### 8.1 All 17 free models

Sorted by context length. `structured_outputs` = supports enforced JSON schema,
which eliminates P0-1 and P1-2 at the source.

| Model ID | Context | Max output | `response_format` | `structured_outputs` |
|----------|--------:|-----------:|:-----------------:|:--------------------:|
| `nvidia/nemotron-3.5-lightning:free` | 1,000,000 | 65,536 | no | no |
| `nvidia/nemotron-3-ultra-550b-a55b:free` ← current synthesis default | 1,000,000 | 65,536 | no | no |
| `dots-studio/dots-3-note-preview:free` | 512,000 | 512,000 | **yes** | **yes** |
| `poolside/laguna-s-2.1:free` | 262,144 | 32,768 | no | no |
| `poolside/laguna-xs-2.1:free` | 262,144 | 32,768 | no | no |
| `google/gemma-4-26b-a4b-it:free` | 262,144 | 32,768 | **yes** | **yes** |
| `google/gemma-4-31b-it:free` | 262,144 | 32,768 | **yes** | no |
| `nvidia/nemotron-3-super-120b-a12b:free` | 262,144 | 262,144 | **yes** | **yes** |
| `cohere/north-mini-code:free` | 256,000 | 64,000 | no | no |
| `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` | 256,000 | 65,536 | no | no |
| `nvidia/nemotron-3-nano-30b-a3b:free` | 256,000 | — | no | no |
| `openai/gpt-oss-20b:free` | 131,072 | 32,768 | **yes** | **yes** |
| `liquid/lfm-2.5-2.6b:free` | 128,000 | 8,192 | **yes** | **yes** |
| `z-ai/glm-5.2:free` | 128,000 | 128,000 | no | no |
| `nvidia/nemotron-3.5-content-safety:free` | 128,000 | 8,192 | no | no |
| `nvidia/nemotron-nano-12b-v2-vl:free` | 128,000 | 128,000 | no | no |
| `nvidia/nemotron-nano-9b-v2:free` | 128,000 | — | **yes** | **yes** |
| `openrouter/free` (auto-router) | 200,000 | — | **yes** | **yes** |

**Not present:** `meta-llama/llama-3.3-70b-instruct:free` — the P0-2 root cause.

### 8.2 Cheap paid models worth considering

Filtered to ≤$0.60/M input, ≥64k context, `structured_outputs` supported
(163 matched; cheapest 12 shown). Prices per 1M tokens.

| Model ID | Context | Input | Output | Note |
|----------|--------:|------:|-------:|------|
| `inclusionai/ling-2.6-flash` | 262,144 | $0.010 | $0.030 | cheapest viable |
| `mistralai/mistral-nemo` | 131,072 | $0.019 | $0.030 | cheapest output |
| `nex-agi/nex-n2-mini` | 262,144 | $0.025 | $0.100 | |
| `upstage/solar-pro4` | 524,288 | $0.030 | $0.120 | |
| `openai/gpt-oss-120b` | 131,072 | $0.030 | $0.170 | **recommended rerank/extract** |
| `openai/gpt-oss-20b` | 131,072 | $0.030 | $0.130 | free tier also exists |
| `cohere/command-r7b-12-2024` | 128,000 | $0.037 | $0.150 | |
| `qwen/qwen3-30b-a3b-instruct-2507` | 262,144 | $0.048 | $0.193 | |
| `ibm-granite/granite-4.1-8b` | 131,072 | $0.050 | $0.100 | |
| `google/gemini-2.5-flash-lite:batch` | 1,048,576 | $0.050 | $0.200 | batch only |
| `meta-llama/llama-3.1-8b-instruct` | 131,072 | $0.050 | $0.080 | |
| `meta-llama/llama-3.3-70b-instruct` | 131,072 | $0.100 | $0.320 | paid form of the dead default |

### 8.3 Recommended configuration

Resolves P0-2 and P5-6 together. Two options:

**Stay free** — zero cost, still 3–5 min/run:
```env
LLM_RERANK_MODEL=nvidia/nemotron-3-super-120b-a12b:free   # structured_outputs
LLM_SYNTHESIS_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free # already valid
```

**Go paid** — ~$0.0025/run, seconds not minutes:
```env
LLM_RERANK_MODEL=openai/gpt-oss-120b
LLM_SYNTHESIS_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
```

Also worth adding a third setting: `extract.py:93` currently reuses
`llm_rerank_model`. Split it (`LLM_EXTRACT_MODEL`) so rerank can go cheap while
extraction stays accurate — `PRD.md:153-154` asks for exactly this.

**Pick models with `structured_outputs: yes.`** Passing a JSON schema makes P0-1
(fence stripping) and P1-2 (schema validation) structurally impossible rather
than defensively handled.

---

## 9. Already good — do not regress

| Strength | Where |
|----------|-------|
| Partial-failure tolerance: one paper's extraction failure never kills the run | `extract.py:96-104`, `service.py:52-64` |
| Prompt-injection defence: all three system prompts mark paper text as untrusted data | `rerank.py:12-13`, `extract.py:18-19`, `synthesize.py:16-17` |
| XSS-safe rendering: `x-text` everywhere, zero `x-html`/`innerHTML` | `index.html:94-166` |
| Security headers via middleware | `main.py:50-57` |
| CORS locked to known origins, no wildcard | `config.py:31`, `main.py:43-48` |
| Rate limit + concurrency cap enforced | `api/search.py:19`, `:28-40` |
| Input validation incl. control-char stripping | `models.py:4-15` |
| Synthesis sends distilled fields, not raw abstracts — already token-frugal | `synthesize.py:66-79` |
| Fail-fast on missing env at startup | `main.py:34-37` |
| Zero-candidate early return skips all LLM calls | `service.py:35-44` |
| Empty-content guards on all three stages (lapse #7 fix held) | `rerank.py:63-65`, `extract.py:67-69`, `synthesize.py:98-100` |
| Structured stage logging with timings | `service.py:66-79` |

---

## 10. Execution order

Dependency-aware. Do not reorder 1-3.

| Step | Task | Refs | Effort | Payoff |
|-----:|------|------|--------|--------|
| 1 | Fix rerank fence bug; share `_strip_fences` | P0-1 | 30 min | Stops hard 502s |
| 2 | Fix dead model default; add startup existence check | P0-2 | 30 min | Fresh clone works |
| 3 | **Eval harness** — 15 topics, labels, `recall@50` / `precision@18` | P1-6 | half day | Makes 4-11 measurable |
| 4 | Pydantic validation on rerank + extract; prefer `structured_outputs` models | P1-2, §8.3 | 2 h | Kills silent corruption |
| 5 | Query expansion stage | P1-1 | half day | **Largest quality gain** |
| 6 | `_MAX_WORKERS` 2→5 | P2-2 | 1 min | Free latency |
| 7 | Decide dedup strategy, then SQLite cache | P5-4, P2-1, P2-5 | 1 day | Token savings + unlocks FR10/11 |
| 8 | Wire `response.usage`; enforce `DAILY_TOKEN_BUDGET` | P2-3, P4-2 | half day | Closes last v1 security lapse |
| 9 | Async run + SSE | P3-1, P3-2, P5-2 | 1-2 days | Meets `<90s` NFR |
| 10 | CI: gitleaks + pip-audit + pytest | P4-1 | 20 min | Public-repo hygiene |
| 11 | Reconcile README/PRD drift | §7 | 1 h | Credibility |
| 12 | Embedding prefilter | P2-4 | 1 day | Token + quality |
| 13 | Rate-limiter leak, XFF handling, pin deps | P3-3, P3-4, P3-5 | 2 h | Pre-multi-user |
| 14 | Full-text extraction, top-5 | P5-7 | 1-2 days | Depth |
| 15 | Vendor Alpine/Tailwind, tighten CSP | P4-3, P4-4 | half day | Supply chain |
| 16 | Next.js migration | P5-8 | — | Defer |

**Totals:** 24 code-level findings, 6 doc-level, 3 catalog-level.
Steps 1, 2, 6 together are under an hour and fix a crash, a broken default, and
half the latency.
