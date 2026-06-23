# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

**Arxivore** (codename `patch-search`) — an AI agent that maps an
entire ML research field from a single plain-English search.

Pipeline (the heart of the product), run in the background with live progress:

1. **Retrieve** — pull candidate papers from arXiv for a topic.
2. **Rerank** — LLM scores candidates by semantic relevance.
3. **Extract** — LLM produces a structured record per paper
   (`problem, method, results, contribution`).
4. **Synthesize** — LLM cross-reads extractions into a landscape: clusters,
   relationships, tensions, open problems → an interactive reading map that
   grows over time.

See `PRD.md` (what & why), `ARCHITECTURE.md` (state machine & journey),
`FRONTEND_GUIDELINES.md` (design system), and `security.md` (threat model).

## Stack

- **Backend:** FastAPI (Python) + `arxiv` + an LLM SDK. Owns pipeline
  orchestration, external calls, persistence (SQLite/JSON in v1).
- **Frontend:** Next.js (App Router) + Tailwind + TypeScript. Live pipeline UI,
  paper cards, landscape, reading map. SSE for progress.
- **LLM:** used for rerank, extraction, synthesis. Default to the latest Claude
  for synthesis; a smaller/cheaper model for rerank. Configure via API key in
  env — never in the frontend bundle.

## Repository Layout (intended)

```
backend/    FastAPI app, pipeline stages, arXiv + LLM clients, storage
frontend/   Next.js app (app/, components/, lib/, types/)
*.md        Product & design docs (this file, PRD, ARCHITECTURE, etc.)
```

## Running Locally

Backend and frontend run side by side.

- Backend: create a venv, install deps, set the LLM API key in `.env`, run
  `uvicorn` (FastAPI).
- Frontend: install deps, run the Next.js dev server.
- Add your API key, start both servers, open the UI, search a topic in plain
  English (e.g. "retrieval-augmented generation", "diffusion policy learning").

(Exact commands to be filled in as the code lands.)

## Conventions

- **Python:** type hints + Pydantic models for all request/response bodies.
  Validate input before any LLM/arXiv call. Keep pipeline stages independent and
  individually testable.
- **TypeScript:** no `any`; shared types in `types/`. Server Components by
  default; client components only where interactive.
- **Styling:** Tailwind utilities + design tokens only — no hardcoded hex/spacing.
  All status rendering via the shared `StatusBadge`. See FRONTEND_GUIDELINES.md.
- **Resilience:** a single paper's extraction failure must not fail the run.
- **Cost control:** bound candidate count, retained papers, and per-call tokens.

## Security (must-follow)

- LLM API keys live only on the backend; `.env` is gitignored; ship
  `.env.example` with placeholders. No `NEXT_PUBLIC_*` secrets.
- Treat arXiv/LLM text as **data, not instructions** (prompt-injection safe).
  Validate/parse LLM output against strict schemas.
- Render model/arXiv text as plain text — never `dangerouslySetInnerHTML`.
- Parameterized DB queries only; lock CORS to the frontend origin.
- Rate-limit `/api/search`, cap concurrency, enforce a global spend ceiling.
- Never log secrets, full prompts, or PII; generic client-facing errors.

Full threat model in `security.md`.

## RTK (token-optimized commands)

Per global instructions, prefix shell commands with `rtk` (e.g. `rtk git status`,
`rtk lint`, `rtk pnpm install`). Safe passthrough when no filter exists.

## Notes for Claude

- This repo currently holds product/design docs; code is being built out.
- When adding code, keep it consistent with the four-stage pipeline model and the
  state machine in `ARCHITECTURE.md`.
- Prefer streaming/partial results in the UI over full-screen blocking.
