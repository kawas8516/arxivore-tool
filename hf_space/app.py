"""Arxivore — HF Spaces entry point.

Three-tab Gradio interface over the four-stage pipeline:
  retrieve → rerank (bge-reranker-v2-m3) → extract → synthesize (Phi-4-mini-instruct)
"""

import logging
import sys
import os

# Make pipeline imports work from this flat directory
sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr

from pipeline.retrieve import retrieve_candidates
from pipeline.rerank import rerank_candidates
from pipeline.extract import extract_papers
from pipeline.synthesize import synthesize_landscape

logging.basicConfig(level=logging.INFO)

_HEADER = """
# Arxivore
**Map an entire ML research field from a single search.**

Enter a plain-English topic and the pipeline will retrieve papers from arXiv,
rerank them by relevance, extract structured information from each, then
synthesize a landscape of clusters, relationships, tensions, and open problems.
"""

_COLS = ["Title", "Score", "Problem", "Method", "Contribution", "URL"]


def run_pipeline(query: str, num_papers: int):
    query = query.strip()
    if len(query) < 3:
        yield "Query must be at least 3 characters.", None, "", {}
        return

    # --- Stage 1: Retrieve ---
    yield "Retrieving papers from arXiv...", None, "", {}
    try:
        candidates, retrieve_ms = retrieve_candidates(query)
    except Exception as e:
        yield f"Retrieve failed: {e}", None, "", {}
        return

    if not candidates:
        yield "No papers found for that topic. Try a broader query.", None, "", {}
        return

    yield (
        f"Retrieved {len(candidates)} candidates ({retrieve_ms} ms). Reranking with bge-reranker-v2-m3...",
        None, "", {},
    )

    # --- Stage 2: Rerank ---
    try:
        top_papers, rerank_ms = rerank_candidates(query, candidates, top_k=int(num_papers))
    except Exception as e:
        yield f"Rerank failed: {e}", None, "", {}
        return

    yield (
        f"Ranked to top {len(top_papers)} papers ({rerank_ms} ms). Extracting with Phi-4-mini-instruct...",
        None, "", {},
    )

    # --- Stage 3: Extract ---
    try:
        papers, extract_ms, errors = extract_papers(top_papers)
    except Exception as e:
        yield f"Extract failed: {e}", None, "", {}
        return

    rows = _build_rows(papers)
    yield (
        f"Extracted {len(papers) - errors}/{len(papers)} papers ({extract_ms} ms, {errors} errors). Synthesizing...",
        rows, "", {},
    )

    # --- Stage 4: Synthesize ---
    try:
        landscape, synth_ms = synthesize_landscape(query, papers)
    except Exception as e:
        yield f"Synthesis failed: {e}", rows, "", {}
        return

    total_ms = retrieve_ms + rerank_ms + extract_ms + synth_ms
    status = (
        f"Done in {total_ms / 1000:.1f}s — "
        f"{len(papers)} papers · {len(landscape.clusters)} clusters · "
        f"{len(landscape.tensions)} tensions · {len(landscape.open_problems)} open problems"
    )
    yield status, rows, _build_landscape_md(query, landscape), landscape.model_dump()


def _build_rows(papers) -> list[list]:
    rows = []
    for p in papers:
        score = f"{p.relevance_score:.3f}" if p.relevance_score is not None else "—"
        rows.append([
            p.title,
            score,
            p.problem or ("(extraction failed)" if p.extract_status == "error" else ""),
            p.method or "",
            p.contribution or "",
            p.url,
        ])
    return rows


def _build_landscape_md(topic: str, landscape) -> str:
    md = f"## Research Landscape: *{topic}*\n\n"

    if landscape.clusters:
        md += "### Clusters\n\n"
        for c in landscape.clusters:
            md += f"**{c.name}**\n{c.summary}\n\n"

    if landscape.relationships:
        md += "### Relationships\n\n"
        for r in landscape.relationships:
            md += f"- **{r.from_cluster}** → *{r.kind}* → **{r.to_cluster}**: {r.description}\n"
        md += "\n"

    if landscape.tensions:
        md += "### Tensions\n\n"
        for t in landscape.tensions:
            md += f"- {t}\n"
        md += "\n"

    if landscape.open_problems:
        md += "### Open Problems\n\n"
        for op in landscape.open_problems:
            md += f"- {op}\n"

    return md


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Arxivore", theme=gr.themes.Soft()) as demo:
    gr.Markdown(_HEADER)

    with gr.Tabs():

        # Tab 1 — Search & status
        with gr.TabItem("Search"):
            with gr.Row():
                with gr.Column(scale=4):
                    query_box = gr.Textbox(
                        label="Research topic",
                        placeholder="e.g. retrieval-augmented generation, diffusion policy learning",
                        lines=1,
                    )
                with gr.Column(scale=1):
                    num_slider = gr.Slider(
                        minimum=5, maximum=20, value=10, step=1,
                        label="Papers to analyse",
                    )
            run_btn = gr.Button("Map this field", variant="primary")
            status_md = gr.Markdown("Enter a topic above and click **Map this field**.")

        # Tab 2 — Papers table
        with gr.TabItem("Papers"):
            papers_df = gr.Dataframe(
                headers=_COLS,
                datatype=["str", "str", "str", "str", "str", "str"],
                wrap=True,
                label="Ranked papers with structured extractions",
            )

        # Tab 3 — Landscape
        with gr.TabItem("Landscape"):
            landscape_md = gr.Markdown("Landscape will appear here after the pipeline runs.")
            with gr.Accordion("Raw JSON", open=False):
                landscape_json = gr.JSON(label="Landscape JSON")

    run_btn.click(
        fn=run_pipeline,
        inputs=[query_box, num_slider],
        outputs=[status_md, papers_df, landscape_md, landscape_json],
    )

if __name__ == "__main__":
    demo.launch()
