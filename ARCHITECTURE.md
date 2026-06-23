# ARCHITECTURE — User Journey & Architecture

Maps how a user moves through the app and how application/pipeline state
transitions, so implementation stays consistent across frontend and backend.

## 1. High-Level Architecture

```
┌─────────────────────────┐        ┌──────────────────────────────┐
│  Next.js + Tailwind UI  │        │        FastAPI backend        │
│                         │  HTTP  │                              │
│  - Search entry         │ <────> │  /api/search                 │
│  - Pipeline progress    │  SSE   │  /api/runs/{id}[/stream]     │
│  - Paper cards          │        │  /api/runs/{id}/papers       │
│  - Landscape view       │        │  /api/runs/{id}/landscape    │
│  - Reading map          │        │  Pipeline orchestrator       │
└─────────────────────────┘        └───────┬──────────┬──────────┘
                                            │          │
                                  ┌─────────▼──┐   ┌───▼─────────┐
                                  │  arXiv API │   │ LLM provider│
                                  └────────────┘   └─────────────┘
                                            │
                                     ┌──────▼───────┐
                                     │   Storage    │
                                     │ (SQLite/JSON)│
                                     └──────────────┘
```

- **Frontend** owns presentation and per-stage rendering.
- **Backend** owns orchestration of the four-stage pipeline, external calls,
  and persistence.
- **Pipeline** runs async; the UI observes via SSE (preferred) or polling.

## 2. Core User Journey

```
Landing/Search → Submit topic → Live pipeline → Explore landscape
       │                              │                  │
       │                              │                  ├─ open paper card
       │                              │                  ├─ open cluster
       └──────────── Reading map ◄────┴──────────────────┘ (persisted runs)
```

1. **Landing / Search** — User sees a single prominent search box with example
   topics ("retrieval-augmented generation", "diffusion policy learning").
2. **Submit** — Frontend POSTs to `/api/search`; receives a `run_id`; navigates
   to the run view.
3. **Live pipeline** — UI subscribes to the run stream and renders each stage's
   status as it progresses.
4. **Explore** — On completion, user explores the landscape, drills into
   clusters and individual paper cards.
5. **Reading map** — Prior runs persist; the user revisits and the map grows
   over time; papers can be marked read/to-read.

## 3. Pipeline State Machine (per run)

```
            ┌──────────┐
            │  QUEUED  │
            └────┬─────┘
                 ▼
     ┌────────────────────────┐   stage error (fatal)
     │   RETRIEVING (arXiv)   │ ───────────────┐
     └───────────┬────────────┘                │
                 ▼                              │
     ┌────────────────────────┐                │
     │   RERANKING (LLM)       │ ───────────────┤
     └───────────┬────────────┘                │
                 ▼                              ▼
     ┌────────────────────────┐          ┌──────────┐
     │  EXTRACTING (LLM, N×)   │ ───────► │  FAILED  │
     └───────────┬────────────┘          └──────────┘
                 ▼
     ┌────────────────────────┐
     │  SYNTHESIZING (LLM)     │
     └───────────┬────────────┘
                 ▼
            ┌──────────┐
            │ COMPLETE │
            └──────────┘
```

**Per-stage status:** each stage is `pending → running → done | error`.

**Partial failure rule:** in EXTRACTING, an individual paper may fail
(`paper.status = error`) without failing the run — synthesis proceeds with the
papers that succeeded. Only retrieval/rerank/synthesis hard-failures move the run
to `FAILED`.

**Run object (conceptual):**
```jsonc
{
  "run_id": "uuid",
  "topic": "retrieval-augmented generation",
  "state": "EXTRACTING",
  "stages": {
    "retrieve":   { "status": "done",    "count": 50,  "ms": 1200 },
    "rerank":     { "status": "done",    "kept": 18,   "ms": 8400 },
    "extract":    { "status": "running", "done": 11, "total": 18 },
    "synthesize": { "status": "pending" }
  },
  "created_at": "...", "updated_at": "..."
}
```

## 4. Frontend State Transitions (run view)

```
IDLE ──submit──► STARTING ──run_id──► STREAMING
                                         │
   ┌─── stage events update progress ────┤
   │                                     ▼
   │                            (state == COMPLETE) ──► READY
   │                            (state == FAILED)   ──► ERROR
   └──────────────── reconnect on dropped stream ───┘
```

- **STREAMING:** progressively reveal results — show candidates after retrieve,
  ranked list after rerank, paper cards as extractions land, landscape after
  synthesize.
- **READY:** full landscape + reading map interactions enabled.
- **ERROR:** show which stage failed and offer retry.
- **Reconnect:** on SSE drop, refetch `/api/runs/{id}` to resync, then resubscribe.

## 5. Sequence: a search run

```
User → UI:        enter topic, click Search
UI → Backend:     POST /api/search { topic }
Backend → UI:     202 { run_id, state: QUEUED }
UI → Backend:     GET /api/runs/{run_id}/stream  (SSE)
Backend → arXiv:  query candidates
Backend → UI:     event: retrieve.done { count }
Backend → LLM:    rerank candidates
Backend → UI:     event: rerank.done { kept }
loop per paper:
  Backend → LLM:  extract structured record
  Backend → UI:   event: extract.progress { done, total }
Backend → LLM:    synthesize landscape
Backend → UI:     event: synthesize.done
Backend → Store:  persist run + papers + landscape
Backend → UI:     event: run.complete
UI:               render landscape + reading map
```

## 6. Key Screens & Data Needs

| Screen | Data source | Notes |
|--------|-------------|-------|
| Search | — | Topic input + examples |
| Pipeline | `/runs/{id}/stream` | Four stage cards with live status |
| Paper card | `/runs/{id}/papers` | problem/method/results/contribution + arXiv link |
| Landscape | `/runs/{id}/landscape` | clusters, relationships, tensions, open problems |
| Reading map | `/runs` | persisted runs; read/to-read marks |

## 7. Persistence Model (v1)

Start simple (SQLite or JSON files), shaped so it can move to Postgres later.

- `runs(run_id, topic, state, created_at, updated_at)`
- `papers(paper_id, run_id, arxiv_id, title, authors, url, score,
   problem, method, results, contribution, status, read_state)`
- `landscapes(run_id, clusters_json, relationships_json, tensions_json,
   open_problems_json)`

The **reading map** is the union of `papers` across runs, deduped by `arxiv_id`.

## 8. Error & Edge Cases

- Empty/too-broad topic → backend returns validation error before any LLM call.
- arXiv returns zero results → run completes with empty landscape + guidance.
- LLM returns malformed JSON → repair/retry once, else mark stage error.
- Stream disconnects → UI resyncs via GET; backend run continues independently.
- Duplicate paper across runs → deduped by `arxiv_id` in the reading map.
