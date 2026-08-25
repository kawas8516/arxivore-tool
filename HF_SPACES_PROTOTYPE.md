# Arxivore — Hugging Face Spaces Prototype Plan

**Branch:** `hf-spaces-prototype`  
**Goal:** A self-contained, recruiter-ready demo deployed on HF Spaces, replacing the Next.js frontend with Gradio and wiring the pipeline to a free HF-hosted model.

---

## 1. What changes vs. the main branch

| Layer | Main branch | This prototype |
|---|---|---|
| UI | Next.js + Tailwind | **Gradio** (Python, runs in same process as backend) |
| LLM provider | OpenRouter (free-tier failover) | **HF Inference API** — one free model, no key needed for public models |
| Deploy target | Local / self-hosted | **Hugging Face Spaces** (Gradio SDK) |
| Frontend build step | `npm` / `pnpm` | None — Gradio UI is pure Python |
| Auth | `.env` API key | HF token for private models; public model = zero config |

Everything else stays the same: the four-stage pipeline (`retrieve → rerank → extract → synthesize`), arXiv calls, Pydantic models, per-paper resilience.

---

## 2. Model selection from Hugging Face

### Synthesis + Extraction (main LLM)

**Originally chosen: `microsoft/Phi-4-mini-instruct` — superseded, see below.**

> **It was never served.** The HF Inference API answers
> `microsoft/Phi-4-mini-instruct` with `model_not_supported` — no provider hosts
> it, and the `-mini-instruct` variant does not exist (plain `microsoft/phi-4`
> does). The Space still built and ran green, so this surfaced only as empty
> Papers and Landscape tabs at runtime.
>
> **Now: `google/gemma-3-12b-it`** — served, returns valid JSON, and small
> enough to be cheap on free-tier inference credits. It fences its JSON, which
> `strip_fences()` already handles. `Qwen/Qwen2.5-72B-Instruct` also tested
> clean if extraction quality ever needs the upgrade.

| Property | Value |
|---|---|
| HF repo | `google/gemma-3-12b-it` |
| Parameters | 12B |
| Served by | featherless-ai, deepinfra (via the HF router) |
| Link | https://hf.co/google/gemma-3-12b-it |

**Why gemma-3-12b-it:** it is actually served — the property the original pick
turned out to lack. Inference runs provider-side, so Space CPU and cold start no
longer bound the model choice; what matters instead is credits per run, which a
12B keeps low. Verified end to end: 2/2 papers extracted in 4.2 s, synthesis
returned 2 clusters, 2 relationships, 1 tension, 3 open problems.

**Tested alternatives:** `Qwen/Qwen2.5-72B-Instruct` emitted bare JSON and is the
quality upgrade if extraction ever needs it, at more credits per run.
`microsoft/phi-4` is served but ignored the JSON-only instruction.
`Qwen/Qwen3-14B` returned empty content (a thinking model).

Extraction is done by prompting the model with strict JSON-only instructions —
no separate model needed. Output arrives fenced; `strip_fences()` handles it.

### Rerank (dedicated cross-encoder — no LLM cost)

**Chosen: `BAAI/bge-reranker-v2-m3`**

| Property | Value |
|---|---|
| HF repo | `BAAI/bge-reranker-v2-m3` |
| Parameters | 568M (XLM-RoBERTa base) |
| License | Apache 2.0 |
| Downloads | 16.2M |
| Library | `sentence-transformers` |
| Link | https://hf.co/BAAI/bge-reranker-v2-m3 |

**Why bge-reranker-v2-m3:** Most downloaded reranker on HF (16.2M), multilingual, runs on CPU in ~0.5s for 50 papers, no API cost. Significantly stronger than ms-marco-MiniLM while still being CPU-viable.

---

## 3. Architecture for the Spaces prototype

Repo root (this branch) — HF Spaces expects a flat layout with `app.py` at root,
so these files sit at the root rather than under a subdirectory:

```
app.py                  ← Gradio app entry point (replaces main.py + Next.js)
pipeline/
  retrieve.py           ← unchanged from backend/app/pipeline/retrieve.py
  rerank.py             ← swapped: BAAI/bge-reranker-v2-m3 via sentence-transformers
  extract.py            ← unchanged, but calls HF Inference API instead of OpenRouter
  synthesize.py         ← unchanged, but calls HF Inference API instead of OpenRouter
llm.py                  ← thin wrapper around huggingface_hub.InferenceClient
models.py               ← unchanged Pydantic models
requirements.txt        ← gradio, arxiv, sentence-transformers, huggingface_hub, pydantic
README.md               ← HF Spaces card in YAML frontmatter, project README below
```

`backend/` and the design docs stay in the repo alongside these. HF only reads
`README.md`'s frontmatter, `app_file`, and `requirements.txt`, and ignores
everything the card doesn't reference — so the FastAPI backend is never part of
the Space build.

**Keeping the mirror honest.** The Space is its own git repo and is updated by
hand, so the nine code files above must stay byte-identical to it. Verify with:

```bash
git cat-file -s $(git rev-parse HEAD:app.py)    # 7695
git cat-file -s $(git rev-parse HEAD:llm.py)    # 2078
# models.py 1268 · requirements.txt 312
# pipeline/{__init__.py 0, extract.py 3393, rerank.py 982,
#           retrieve.py 970, synthesize.py 2819}
```

These drift every time a Space file changes. Re-run them after any push to the
Space and update this block — a stale number here reads as a broken mirror.

`README.md` is the one deliberate divergence: the Space carries the card alone,
this branch carries the card plus the full project README beneath it.

---

## 4. Gradio UI design

Three tabs in one `gr.Blocks` interface:

### Tab 1 — Search
- `gr.Textbox` — plain-English query (e.g. "diffusion policy learning")
- `gr.Slider` — number of papers (5–20, default 10)
- `gr.Button` — "Map this field"
- `gr.Markdown` — live status (SSE replaced by `gr.Progress` + generator `yield`)

### Tab 2 — Papers
- `gr.Dataframe` — ranked papers: title, score, problem, method, contribution
- `gr.Textbox` (read-only) — selected paper's full extraction JSON

### Tab 3 — Landscape
- `gr.Markdown` — synthesized landscape: clusters, tensions, open problems
- `gr.JSON` — raw synthesis output for inspecting structure

The UI streams progress via a Python generator yielding status strings — no SSE, no WebSocket, no JS build step.

---

## 5. `llm.py` rewrite for HF Inference API

```python
from huggingface_hub import InferenceClient

_MODEL = "google/gemma-3-12b-it"
_client = InferenceClient()  # uses HF_TOKEN env var if set, else public API

def complete(prompt: str, model: str = _MODEL) -> str:
    response = _client.text_generation(
        prompt,
        model=model,
        max_new_tokens=1024,
        temperature=0.2,
        return_full_text=False,
    )
    return response
```

No failover pool needed for the demo — single model, single call. Add a retry decorator for transient 429s.

---

## 6. `rerank.py` rewrite (cross-encoder, no LLM)

```python
from sentence_transformers import CrossEncoder

_model = CrossEncoder("BAAI/bge-reranker-v2-m3")

def rerank(query: str, papers: list[Paper], top_k: int = 10) -> list[Paper]:
    pairs = [(query, f"{p.title}. {p.abstract}") for p in papers]
    scores = _model.predict(pairs)
    ranked = sorted(zip(scores, papers), reverse=True)
    return [p for _, p in ranked[:top_k]]
```

This runs on CPU in ~0.5 s for 50 papers. No API cost. `bge-reranker-v2-m3` is multilingual and the most downloaded reranker on HF (16.2M).

---

## 7. HF Spaces `README.md` card

```yaml
---
title: Arxivore
emoji: 🔬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
python_version: "3.11"
app_file: app.py
pinned: true
license: mit
---
```

HF Spaces reads this frontmatter to know it's a Gradio app and which file to run.
It sits at the top of the repo-root `README.md`, with the project README below it.

`sdk_version` moved 4.44.0 → 5.9.1 during build fixes: Gradio 4.x pulls `audioop`,
which Python 3.13 removed. Pinning `python_version: "3.11"` and Gradio 5.x settles
it from both directions.

---

## 8. `requirements.txt`

```
gradio>=5.0.0
arxiv>=2.1.0
sentence-transformers>=3.0.0
huggingface_hub>=0.23.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
httpx>=0.27.0
tenacity>=8.2.0
```

No FastAPI, no uvicorn, no npm, no Next.js.

---

## 9. Implementation steps (ordered)

- [x] **Step 1** — Scaffold `app.py` with the three-tab Gradio skeleton
- [x] **Step 2** — Copy and adapt `models.py`, `retrieve.py`
- [x] **Step 3** — Write new `llm.py` wrapping `InferenceClient` (now `gemma-3-12b-it`)
- [x] **Step 4** — Write new `rerank.py` using `BAAI/bge-reranker-v2-m3`
- [x] **Step 5** — Adapt `extract.py` to call the new `llm.py`
- [x] **Step 6** — Adapt `synthesize.py` the same way
- [x] **Step 7** — Wire full pipeline into `app.py` generator with live status updates
- [x] **Step 8** — Write the HF Spaces `README.md` card
- [x] **Step 9** — Space live at [`huggingface.co/spaces/kawas8516/arxivore`](https://huggingface.co/spaces/kawas8516/arxivore)
- [x] **Step 10** — Validate the live Space: run one query, confirm all three tabs populate
- [x] **Step 11** — Promote the Space files from `hf_space/` to the repo root so this
      branch mirrors the Space repo and the two can be diffed without the HF API

### Updating the Space

The Space is a **separate git repo** and is **not** linked to GitHub — pushing
this branch does not deploy. To ship a change, copy the nine root files into a
clone of the Space repo and push there. The card is the one file that differs:
the Space carries frontmatter only, this branch carries frontmatter plus the
project README.

---

## 10. What to show recruiters

1. Open the live Space URL (no install, no API key needed).
2. Type a research topic (e.g. "reinforcement learning from human feedback").
3. Watch the pipeline progress in real time — retrieve → rerank → extract → synthesize.
4. Show the ranked paper table with structured extractions.
5. Show the synthesized landscape: clusters, open problems, tensions.
6. Point to the code: flat Python, one `app.py`, four pipeline stages, `BAAI/bge-reranker-v2-m3` cross-encoder for reranking, `gemma-3-12b-it` for synthesis/extraction — demonstrating end-to-end ML pipeline design with two distinct HF models.

**Key talking points:**
- Four-stage agentic pipeline (not just a wrapper around one LLM call)
- Free, open-source model serving via HF Inference API
- Cross-encoder reranking (retrieval + ranking = classic IR + ML)
- Structured extraction with schema validation (Pydantic)
- Streaming UX (generator-based, not blocking)
- Deployed live — anyone can use it

---

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| HF Inference API rate limits Phi-4-mini on free tier | Add `tenacity` retry with exponential backoff; fall back to `HuggingFaceH4/zephyr-7b-beta` as secondary |
| Synthesis output slow on shared Spaces CPU (3.8B still needs RAM) | Cap `max_new_tokens=512` for demo; Phi-4-mini at 3.8B is significantly faster than 7B alternatives |
| bge-reranker-v2-m3 download on first Spaces cold start (~1.1 GB) | Load at module level so it caches after first run; subsequent requests are instant |
| arXiv API flakiness | Existing retry logic in `retrieve.py` carries over unchanged |
