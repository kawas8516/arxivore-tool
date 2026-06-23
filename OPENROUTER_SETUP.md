# OpenRouter Dashboard Setup — Prompt for Claude

Copy and paste the prompt below into Claude.ai (or any Claude chat) to get
step-by-step help configuring your OpenRouter account safely for this project.

---

## Prompt to paste into Claude chat

```
I'm a student building a side project called "Arxivore" — a FastAPI
+ Next.js app that uses an OpenAI-compatible LLM API via OpenRouter. I've already
generated an API key and wired it into my .env file. I want you to walk me through
configuring the OpenRouter dashboard so I never get surprise bills, rate-limit
errors, or account-abuse emails.

Here's my project situation:
- Backend: FastAPI (Python), single server, local dev right now
- Models in use: google/gemini-flash-1.5 (reranking, called once per search run)
  and google/gemini-2.5-pro (synthesis, called once per run, later)
- Usage pattern: I'm the only user, running maybe 5–20 searches per day during
  development
- I'm on the free tier / very low budget — I want hard limits, not soft warnings

Please walk me through each of the following in the OpenRouter dashboard
(app.openrouter.ai), one step at a time, with the exact setting names and where
to find them:

1. **Credit limit / spend cap** — how to set a hard monthly or total spending
   limit so the key is automatically disabled if I hit it. I want $0 surprise
   bills. Where exactly is this in the dashboard?

2. **Rate limit configuration** — what rate limits are set by default on a free
   account, and how do I check or adjust them? I want to understand what happens
   when I hit the limit (error code, response body) so my FastAPI app can handle
   it gracefully.

3. **Key restrictions** — can I lock the API key to specific models only
   (gemini-flash-1.5 and gemini-2.5-pro)? Can I restrict it to requests coming
   from localhost only during dev? Show me where to configure this.

4. **Usage monitoring** — where do I see per-model token usage and cost in
   real time? I want to watch spend during a test run.

5. **Email alerts** — how do I set up an email notification before I hit my
   limit (e.g. at 80% of budget), so I'm never caught off guard?

6. **Free models** — which models on OpenRouter are genuinely free (no per-token
   cost)? Are gemini-flash-1.5 and gemini-2.5-pro free or paid? If they have a
   cost, recommend the best free alternatives for:
   - Reranking (fast, follows JSON schema instructions reliably)
   - Synthesis (can read 10–20 short paper summaries and produce structured output)

7. **What to do if I hit a rate limit** — show me the exact error response
   OpenRouter returns (HTTP status + JSON body) so I can handle it in my Python
   code with a proper error message to the user rather than a 500.

After each step, pause and ask if I've found the setting before moving to the
next one.
```

---

## What to do with the answer

Once Claude walks you through the dashboard, come back and update your `.env`
and `backend/app/pipeline/rerank.py` with any model changes.

If you switch to a free model (step 6), change these two lines in `.env`:

```bash
LLM_RERANK_MODEL=<free-model-id-from-openrouter>
LLM_SYNTHESIS_MODEL=<free-model-id-from-openrouter>
```

No code changes needed — the OpenAI-compatible client picks up the new model
name automatically.

## Rate-limit error to handle in code (add when known)

When OpenRouter hits a rate limit it returns HTTP `429`. The FastAPI endpoint
at `backend/app/api/search.py` currently returns a generic 502 for any LLM
failure. Once you know the exact error shape from step 7 above, add a specific
check:

```python
# In rerank_candidates / future extract / synthesize stages:
# Catch openai.RateLimitError and return HTTP 429 to the frontend
# so the UI can show "Rate limit hit — try again in X seconds"
# rather than a generic error.
```

Mark this as a TODO in the code until M2/M3 when the other stages are built.
