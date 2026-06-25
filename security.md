# Security — Arxivore

Scope: a FastAPI backend that calls arXiv and an LLM provider, plus a Next.js
frontend. The app accepts free-text user input, makes outbound calls to external
services with a secret API key, and persists results. This document covers the
threat model and the controls for v1.

## 1. Assets to Protect

- **LLM API key(s)** — the highest-value secret; pay-per-token abuse risk.
- **Provider/account billing** — uncontrolled token spend.
- **Persisted run data** — extractions, landscapes, run history.
- **Backend availability** — the pipeline is expensive; easy to DoS.

## 2. Trust Boundaries

```
[Browser / Next.js]  --HTTP-->  [FastAPI backend]  --HTTPS-->  [arXiv API]
                                       |             --HTTPS-->  [LLM provider]
                                       +--> [Storage: SQLite/JSON/DB]
```

- The browser is **untrusted**. All input from it is validated server-side.
- Secrets live **only** on the backend, never shipped to the frontend bundle.
- arXiv and the LLM provider are trusted services but treated as untrusted
  **data sources** (their text flows into prompts and the UI).

## 3. Top Threats & Controls

### 3.1 Secret leakage (LLM API key)
- **Never** expose the key to the frontend. No `NEXT_PUBLIC_*` for secrets.
- Load from environment / `.env` (gitignored). Provide `.env.example` with
  placeholders only.
- Add `.env`, `*.key`, and credential files to `.gitignore`.
- Scan commits for secrets (e.g. `gitleaks`/`trufflehog`) in CI.
- Backend never echoes the key in logs, errors, or API responses.

### 3.2 Cost / token abuse (DoS by wallet)
- **Per-run bounds:** max candidate papers (default 50), max retained papers,
  max tokens per LLM call, max output length.
- **Rate limiting:** per-IP and global request limits on `/api/search`.
- **Concurrency cap:** limited number of in-flight pipeline runs.
- **Daily/global spend ceiling:** hard stop when a configurable budget is hit.
- **Input length cap:** reject overlong topic strings before any LLM call.

### 3.3 Prompt injection (via paper content)
arXiv abstracts are attacker-influenceable text that we feed into LLM prompts
for rerank/extract/synthesize.
- Treat all retrieved text as **data, not instructions**. Use clear delimiters
  and system prompts that state retrieved content must never be executed as
  commands.
- Constrain LLM output with strict schemas (JSON) and validate/parse server-side;
  reject or repair malformed output.
- Do not let model output trigger side effects (no tool execution, no shell, no
  file writes driven by extracted text).
- Render extracted text in the UI as **plain text**, never as raw HTML.

### 3.4 Injection into the app (XSS / SQLi)
- **XSS:** Never `dangerouslySetInnerHTML` with model/arXiv text. React escapes
  by default — keep it that way. Sanitize any markdown rendering.
- **SQLi:** Use parameterized queries / an ORM. Never string-format SQL with
  user input or `run_id`.
- **Path/SSRF:** Do not fetch arbitrary URLs from user/model input. arXiv calls
  use the library with validated query params only.

### 3.5 Input validation
- Validate the topic string: length bounds, strip control chars, reject empty.
- Validate `run_id` / `paper_id` as expected formats (UUID/slug) before lookup.
- Use Pydantic models for all request/response bodies.

### 3.6 CORS & transport
- Restrict CORS to the known frontend origin(s); no wildcard with credentials.
- Enforce HTTPS in production (TLS termination at proxy/host).
- Set security headers (CSP, X-Content-Type-Options, Referrer-Policy) on the
  Next.js app.

### 3.7 Error handling & logging
- Return generic error messages to clients; log details server-side only.
- Never include stack traces, secrets, or full prompts in client responses.
- Log token usage and cost per run for monitoring abuse.

## 4. Data Handling & Privacy

- Stored data is non-personal (public paper metadata + derived summaries).
- User queries may be logged for debugging — document retention and avoid
  logging anything sensitive a user might type.
- No third-party analytics that exfiltrate query content without disclosure.

## 5. Dependencies

- Pin dependency versions; enable Dependabot/`pip-audit`/`npm audit` in CI.
- Vet the `arxiv` and LLM SDK versions; keep them updated.

## 6. Deployment Checklist

Status as of the v1 backend (single-server FastAPI + Alpine UI):

- [x] Secrets in env, not in repo or frontend bundle (`.env` untracked; no
      `NEXT_PUBLIC_*` secret; key never echoed in errors/logs).
- [x] `.env` gitignored; `.env.example` present.
- [x] Rate limiting (per-IP, `RATE_LIMIT_PER_MINUTE`) + concurrency cap
      (`MAX_CONCURRENT_RUNS`) enforced on `/api/search`.
- [ ] Global spend ceiling — `DAILY_TOKEN_BUDGET` is defined but token
      accounting is **not yet wired**; bounded for now by candidate/token caps,
      rate limit, and concurrency cap. See "Known gaps" below.
- [x] CORS locked to known origins (no wildcard); UI is served same-origin.
- [x] Security headers set (CSP, X-Content-Type-Options, X-Frame-Options,
      Referrer-Policy). HTTPS to be enforced at the proxy/host in production.
- [x] Input validation (Pydantic) on every endpoint (topic length/charset).
- [ ] Parameterized DB access — N/A until persistence (M5) lands.
- [~] Secret scanning — a local **pre-commit secret guard** is in place
      (`scripts/git-hooks/pre-commit`, activated via `core.hooksPath`); CI-side
      scanning + dependency audit still pending.
- [x] No secrets/prompts/PII in client-facing errors (generic messages; details
      logged server-side only). Note: search topics are logged for debugging.

### Known gaps (tracked, accepted for single-user v1)

- **Spend ceiling not enforced.** Implement token accounting from LLM
  `response.usage` and hard-stop when `DAILY_TOKEN_BUDGET` is hit (do before any
  public, multi-user exposure).
- **No CI secret scanning / dependency audit.** A local pre-commit guard exists
  (`scripts/git-hooks/pre-commit`); still add `gitleaks` + `pip-audit` in CI
  before opening the repo to external contributors.
- **No auth** (see §7) — assumes single-user / trusted deployment.

## 6a. Before Going Public

Run through this once before flipping the GitHub repo to public visibility.

- [ ] **Confirm repo visibility** is what you intend (GitHub → Settings).
- [ ] **No secret in history:** `git grep "sk-or-v1" $(git rev-list --all)` and
      `git log --all -S "<your key>"` both return nothing. (`.env` must be
      untracked — verify with `git check-ignore .env`.)
- [ ] **Rotate the LLM API key** if it was ever pasted into chat, screenshots, or
      a non-`.env` file — cheap insurance. Update `.env`, then revoke the old key
      at https://openrouter.ai/settings/keys.
- [ ] **Set an OpenRouter spend / credit cap** on the key so exposure can never
      produce a runaway bill, regardless of leaks.
- [ ] **Pre-commit secret guard active:** `git config core.hooksPath` returns
      `scripts/git-hooks`. New clones must run this once (it is not auto-applied
      on clone).
- [ ] **Skim the diff that goes public** for stray tokens, internal URLs, or PII.

> The hook is local. For real enforcement on a public repo, also add a
> server-side secret scan (GitHub secret scanning / `gitleaks` Action).

## 7. Out of Scope (v1)

- Authentication / authorization (single-user / trusted-deployment assumption).
  If exposed publicly, add auth before launch.
- Multi-tenant data isolation.
