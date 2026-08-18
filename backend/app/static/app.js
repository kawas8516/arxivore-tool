// Externalized from index.html's inline <script> so CSP's script-src can drop
// 'unsafe-inline' (see main.py's _CSP comment for what still can't be dropped
// and why).
//
// POST /api/search now returns {run_id, state} immediately — the pipeline
// runs in the background (see ARCHITECTURE.md section 3's state machine).
// This subscribes to its SSE stream for live progress, then fetches the final
// result from the storage-backed run endpoint once it completes.
function mapper() {
  const STAGE_LABELS = {
    QUEUED: "Queued…",
    RETRIEVING: "Retrieving candidates from arXiv…",
    RERANKING: "Reranking by relevance…",
    EXTRACTING: "Extracting problem / method / results / contribution…",
    SYNTHESIZING: "Synthesizing the research landscape…",
  };
  // Comfortably above the documented 3-5 min free-tier run time (RELEASE.md).
  const RUN_TIMEOUT_MS = 8 * 60 * 1000;
  const REQUEST_TIMEOUT_MS = 15 * 1000;

  return {
    topic: "",
    loading: false,
    error: "",
    stageLabel: "",
    result: null,
    examples: [
      "retrieval-augmented generation",
      "diffusion policy learning",
      "state space models for language",
    ],
    _eventSource: null,
    _runTimeoutId: null,

    _cleanupStream() {
      if (this._eventSource) {
        this._eventSource.close();
        this._eventSource = null;
      }
      if (this._runTimeoutId) {
        clearTimeout(this._runTimeoutId);
        this._runTimeoutId = null;
      }
    },

    async search() {
      const topic = this.topic.trim();
      if (topic.length < 3 || this.loading) return;

      this._cleanupStream();
      this.loading = true;
      this.error = "";
      this.result = null;
      this.stageLabel = STAGE_LABELS.QUEUED;

      // Timeout on the initial POST: is the server even reachable — separate
      // concern from whether the run itself finishes in time (below).
      const controller = new AbortController();
      const requestTimeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

      let runId;
      try {
        const res = await fetch("/api/search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ topic }),
          signal: controller.signal,
        });
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          throw new Error(data.detail || `Request failed (${res.status})`);
        }
        const accepted = await res.json();
        runId = accepted.run_id;
      } catch (e) {
        this.loading = false;
        this.error =
          e.name === "AbortError"
            ? "Request timed out. Please try again."
            : e.message || "Something went wrong.";
        return;
      } finally {
        clearTimeout(requestTimeout);
      }

      this._streamProgress(runId);
    },

    _streamProgress(runId) {
      // P3-2: a stalled connection must not hang the UI forever.
      this._runTimeoutId = setTimeout(() => {
        this._cleanupStream();
        this.loading = false;
        this.error = "This run is taking far longer than expected. Please try again.";
      }, RUN_TIMEOUT_MS);

      const source = new EventSource(`/api/runs/${runId}/stream`);
      this._eventSource = source;

      source.onmessage = (event) => {
        let snapshot;
        try {
          snapshot = JSON.parse(event.data);
        } catch {
          return; // malformed event — wait for the next one rather than fail the run
        }

        const stage = STAGE_LABELS[snapshot.state];
        if (stage) {
          const extract = snapshot.stages && snapshot.stages.extract;
          this.stageLabel =
            snapshot.state === "EXTRACTING" && extract && extract.total
              ? `${stage} (${extract.done || 0}/${extract.total})`
              : stage;
        }

        if (snapshot.state === "COMPLETE") {
          this._cleanupStream();
          this._fetchFinalResult(runId);
        } else if (snapshot.state === "FAILED") {
          this._cleanupStream();
          this.loading = false;
          this.error = snapshot.error || "The run failed.";
        }
      };

      source.onerror = () => {
        // The browser's EventSource retries transient drops on its own; only
        // treat this as fatal if the connection is fully closed.
        if (source.readyState === EventSource.CLOSED) {
          this._cleanupStream();
          this.loading = false;
          this.error = "Lost connection to the server. Please try again.";
        }
      };
    },

    extractErrorCount() {
      // RunDetail (the storage-backed result) doesn't carry a precomputed
      // extract_errors count the way the old synchronous SearchResponse did —
      // it's derived here from the papers themselves instead.
      return this.result?.papers?.filter((p) => p.extract_status === "error").length ?? 0;
    },

    async _fetchFinalResult(runId) {
      try {
        const res = await fetch(`/api/runs/${runId}`);
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          throw new Error(data.detail || `Request failed (${res.status})`);
        }
        this.result = await res.json();
      } catch (e) {
        this.error = e.message || "Something went wrong fetching the result.";
      } finally {
        this.loading = false;
      }
    },
  };
}
