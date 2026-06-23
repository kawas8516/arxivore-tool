# PRD — Arxivore

## 1. Overview

**Working name:** Arxivore (codename: `patch-search`)

**One-liner:** An AI agent that maps an entire ML research field from a single
plain-English search.

A user enters a topic (e.g. "retrieval-augmented generation"). The system pulls
candidate papers from arXiv, reranks them by semantic relevance with an LLM,
extracts structured information per paper, and synthesizes a cross-paper research
landscape — clusters, relationships, tensions, and open problems — surfaced as an
interactive reading map that grows over time.

## 2. Problem

Entering a new ML subfield is slow and lossy. A newcomer (or a researcher moving
laterally) faces:

- **Discovery cost** — keyword search on arXiv returns hundreds of weakly-ordered
  hits; relevance ranking is poor.
- **Reading cost** — every paper must be read to extract the same four things:
  problem, method, results, contribution.
- **Synthesis cost** — the hardest part. Understanding how papers relate (who
  builds on whom, who disagrees, what's unsolved) requires holding dozens of
  papers in your head at once.

Existing tools (Google Scholar, Semantic Scholar, Connected Papers) handle
retrieval and citation graphs but do **not** read and synthesize content into a
navigable conceptual landscape.

## 3. Goals & Non-Goals

### Goals
- Turn a single topic query into a structured, explorable research landscape.
- Make the four-stage pipeline (retrieve → rerank → extract → synthesize)
  observable in real time.
- Produce per-paper structured summaries (problem / method / results /
  contribution).
- Produce a landscape view: clusters, inter-cluster relationships, tensions,
  open problems.
- Persist results so a map grows across sessions.

### Non-Goals (v1)
- Full-text PDF parsing of figures/tables (abstract + available metadata only).
- Sources beyond arXiv (no PubMed, ACL Anthology, OpenReview in v1).
- Multi-user accounts, sharing, or collaboration.
- Citation-graph analysis (we synthesize from content, not citation edges).
- Mobile-native apps.

## 4. Target Users

| Persona | Need |
|---------|------|
| Grad student entering a field | Fast, trustworthy orientation map |
| Researcher exploring adjacent area | Identify clusters, tensions, gaps |
| ML engineer evaluating an approach | Compare methods and reported results |
| Technical PM / analyst | High-level landscape without deep math |

## 5. The Pipeline (core product)

A four-stage pipeline runs in the background while the user watches each stage
complete.

1. **Retrieve** — Query arXiv via the `arxiv` API for candidate papers
   (title, abstract, authors, categories, date, links). Cap at N candidates
   (default 50).
2. **Rerank** — LLM scores each candidate for semantic relevance to the user's
   intent (not just keyword overlap), returning a ranked, trimmed set (default
   top 15–20).
3. **Extract** — For each retained paper, LLM produces a structured record:
   `{problem, method, results, contribution}` plus tags.
4. **Synthesize** — LLM reads all extractions and emits a landscape:
   - **Clusters** — groups of papers sharing an approach/theme.
   - **Relationships** — how clusters/papers connect (builds-on, alternative-to).
   - **Tensions** — competing claims or unresolved methodological disagreements.
   - **Open problems** — gaps the literature itself flags.

## 6. Functional Requirements

### Search & Pipeline
- FR1: User submits a free-text topic; system validates and starts a pipeline run.
- FR2: Each stage emits progress/status to the UI (queued → running → done/error).
- FR3: Pipeline runs asynchronously; UI streams stage completion.
- FR4: Partial results are viewable as soon as a stage completes.
- FR5: A run is identified by a stable `run_id` and is persisted.

### Per-paper view
- FR6: Show structured extraction for each paper with link back to arXiv.
- FR7: Show the relevance score / rationale from rerank.

### Landscape view
- FR8: Render clusters, relationships, tensions, open problems.
- FR9: Clicking a cluster reveals its member papers.

### Reading map (growth over time)
- FR10: Persisted runs are retrievable; the map accumulates papers across runs.
- FR11: Mark papers as read / to-read / skipped.

## 7. Non-Functional Requirements

- **Performance:** First candidates visible < 5s; full landscape for default
  paper count target < 90s.
- **Resilience:** A single paper's extraction failure must not fail the run.
- **Cost control:** Bounded paper count and token budgets per run; caching of
  arXiv results and extractions.
- **Observability:** Every stage logs timing, token usage, and errors.
- **Reproducibility:** Same `run_id` re-renders identical results from storage.

## 8. System Architecture (summary)

- **Backend:** FastAPI (Python). Wraps the `arxiv` client and the LLM calls.
  Exposes pipeline endpoints and streams progress (SSE or polling).
- **Frontend:** Next.js + Tailwind. Search entry, live pipeline view, paper
  cards, landscape, reading map.
- **LLM:** Used for rerank, extraction, and synthesis. Provider configured via
  API key.
- **Storage:** Run metadata, extractions, and landscapes persisted (start with
  SQLite/JSON; see ARCHITECTURE.md).

See `ARCHITECTURE.md` for state transitions and `security.md` for the threat model.

## 9. API Surface (v1 draft)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/search` | Start a pipeline run; returns `run_id` |
| GET | `/api/runs/{run_id}` | Run status + available results |
| GET | `/api/runs/{run_id}/stream` | SSE progress stream |
| GET | `/api/runs/{run_id}/papers` | Per-paper extractions |
| GET | `/api/runs/{run_id}/landscape` | Synthesized landscape |
| GET | `/api/runs` | List prior runs (reading map) |
| PATCH | `/api/papers/{id}` | Update read/to-read status |

## 10. Success Metrics

- Time-to-first-insight (query → first landscape) under target.
- % of runs completing all four stages without manual retry.
- Rerank quality: human spot-check agreement on top-10 relevance.
- Return usage: % of users who run ≥2 searches (map growth).

## 11. Milestones

1. **M1 — Pipeline spine:** retrieve + rerank, JSON out, no UI polish.
2. **M2 — Extraction:** structured per-paper records + paper cards.
3. **M3 — Synthesis:** landscape generation + landscape view.
4. **M4 — Live progress:** streaming stage status in UI.
5. **M5 — Persistence & reading map:** runs persist and accumulate.

## 12. Open Questions

- Which LLM provider/model as default, and a cheaper model for rerank vs.
  synthesis? (Default to latest Claude for synthesis; smaller model for rerank.)
- Dedup strategy across overlapping runs (same paper, different topics).
- How to bound synthesis token cost as the map grows large.
