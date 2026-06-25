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

# Arxivore

Map an entire ML research field from a single plain-English search.

**Pipeline:**
1. **Retrieve** — pull candidate papers from arXiv
2. **Rerank** — score by relevance using `BAAI/bge-reranker-v2-m3` (cross-encoder, CPU, no API cost)
3. **Extract** — structured record per paper (problem / method / results / contribution) via `microsoft/Phi-4-mini-instruct`
4. **Synthesize** — cross-read extractions into a landscape: clusters, relationships, tensions, open problems

No API key needed. All models served via HF Inference API.
