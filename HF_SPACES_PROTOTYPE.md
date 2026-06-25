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

**Chosen: `microsoft/Phi-4-mini-instruct`**

| Property | Value |
|---|---|
| HF repo | `microsoft/Phi-4-mini-instruct` |
| Parameters | 3.8B |
| License | MIT |
| Downloads | 765K |
| HF Inference API | Yes — free tier, no gating |
| Link | https://hf.co/microsoft/Phi-4-mini-instruct |

**Why Phi-4-mini-instruct:** Fastest cold start on Spaces CPU (3.8B vs 7B), MIT license, strong structured JSON output for extraction, and Microsoft's newest Phi-4 generation outperforms older 7B models on instruction benchmarks despite being smaller. Ideal for a live recruiter demo where latency matters.

Extraction is done by prompting Phi-4-mini-instruct with strict JSON-only instructions — no separate model needed.

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

```
huggingface_hub/
  app.py                  ← Gradio app entry point (replaces main.py + Next.js)
  pipeline/
    retrieve.py           ← unchanged from backend/app/pipeline/retrieve.py
    rerank.py             ← swapped: BAAI/bge-reranker-v2-m3 via sentence-transformers
    extract.py            ← unchanged, but calls HF Inference API instead of OpenRouter
    synthesize.py         ← unchanged, but calls HF Inference API instead of OpenRouter
  llm.py                  ← thin wrapper around huggingface_hub.InferenceClient
  models.py               ← unchanged Pydantic models
  requirements.txt        ← gradio, arxiv, sentence-transformers, huggingface_hub, pydantic
  README.md               ← HF Spaces card (title, sdk: gradio, python_version: 3.11)
```

No `backend/` nesting — HF Spaces expects a flat layout with `app.py` at root.

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

_MODEL = "microsoft/Phi-4-mini-instruct"
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
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: mit
---
```

HF Spaces reads this frontmatter to know it's a Gradio app and which file to run.

---

## 8. `requirements.txt`

```
gradio>=4.44.0
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

- [ ] **Step 1** — Scaffold `app.py` with the three-tab Gradio skeleton (no pipeline logic yet; confirm it launches on Spaces)
- [ ] **Step 2** — Copy and adapt `models.py`, `retrieve.py` (unchanged)
- [ ] **Step 3** — Write new `llm.py` wrapping `InferenceClient` with `Phi-4-mini-instruct`; smoke-test with a single call
- [ ] **Step 4** — Write new `rerank.py` using `BAAI/bge-reranker-v2-m3`; unit-test with 5 fake papers
- [ ] **Step 5** — Adapt `extract.py` to call the new `llm.py` (swap `complete()` import only)
- [ ] **Step 6** — Adapt `synthesize.py` the same way
- [ ] **Step 7** — Wire pipeline into `app.py` generator; test end-to-end locally with `gradio app.py`
- [ ] **Step 8** — Write the HF Spaces `README.md` card
- [ ] **Step 9** — Push branch; create Space at `huggingface.co/spaces/<username>/arxivore`; link repo
- [ ] **Step 10** — Validate the live Space: run one query, confirm all three tabs populate

---

## 10. What to show recruiters

1. Open the live Space URL (no install, no API key needed).
2. Type a research topic (e.g. "reinforcement learning from human feedback").
3. Watch the pipeline progress in real time — retrieve → rerank → extract → synthesize.
4. Show the ranked paper table with structured extractions.
5. Show the synthesized landscape: clusters, open problems, tensions.
6. Point to the code: flat Python, one `app.py`, four pipeline stages, `BAAI/bge-reranker-v2-m3` cross-encoder for reranking, `Phi-4-mini-instruct` for synthesis/extraction — demonstrating end-to-end ML pipeline design with two distinct HF models.

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
