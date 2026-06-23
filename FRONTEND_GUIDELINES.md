# FRONTEND_GUIDELINES — Design & Implementation

> **Status: aspirational design notes.** The shipped UI is a single static
> `backend/app/static/index.html` (Alpine.js + Tailwind, served by FastAPI — no
> Node). This document describes the fuller Next.js + Tailwind design system we'd
> grow into if the project warrants it; treat it as the north star, not the
> current implementation.

Locks down the design system and frontend tech stack so components and layouts
stay consistent. Applies to the Next.js + Tailwind app.

## 1. Tech Stack

- **Framework:** Next.js (App Router) + React, TypeScript.
- **Styling:** Tailwind CSS. Utility-first; no competing CSS-in-JS.
- **Components:** Headless/accessible primitives (e.g. Radix) + custom styling.
  Keep a small, owned component library in `components/`.
- **Icons:** One icon set only (e.g. `lucide-react`).
- **Data fetching:** Server Components for static loads; `fetch` + a small
  client hook (or SWR/React Query) for the live run; `EventSource` for SSE.
- **State:** Local component state + URL state (`run_id` in the route). Avoid a
  global store unless the reading map demands it.

## 2. Design Principles

1. **The pipeline is the hero.** The four stages must feel alive — progress is a
   feature, not a spinner. Show what's happening, not just that something is.
2. **Progressive disclosure.** Reveal results as each stage completes; never
   block the whole screen waiting for synthesis.
3. **Calm, dense, scannable.** This is a research tool. Favor clarity and
   information density over decoration.
4. **Plain text is safe text.** Never render model/arXiv content as raw HTML.

## 3. Design Tokens

Define once in `tailwind.config` / CSS variables; never hardcode hex in JSX.

### Color (semantic)
```
--bg          base background
--surface     card / panel background
--border      hairline borders
--text        primary text
--text-muted  secondary text
--primary     brand / primary action
--accent      highlights, active cluster
--success     stage done
--warning     stage degraded / partial
--danger      stage error
```
- Support light and dark via CSS variables. Default to a focused, low-chroma
  palette; reserve saturated color for stage status and the active selection.

### Stage status colors
| Status | Token |
|--------|-------|
| pending | `--text-muted` |
| running | `--primary` (animated) |
| done | `--success` |
| error | `--danger` |
| partial | `--warning` |

### Typography
- One sans family (e.g. Inter) for UI; one mono (e.g. JetBrains Mono) for
  arXiv IDs / scores.
- Scale: `xs 12 / sm 14 / base 16 / lg 18 / xl 20 / 2xl 24 / 3xl 30`.
- Body 16px, line-height ~1.6. Don't go below 12px.

### Spacing & radius
- Spacing scale: 4 / 8 / 12 / 16 / 24 / 32 / 48 (Tailwind defaults).
- Radius: `sm` inputs, `md` cards, `lg` modals. Be consistent.
- Max content width ~1200px; landscape view may go wider.

## 4. Layout

- **App shell:** top bar (logo + search) + main content; optional left rail for
  the reading map on wide screens.
- **Search screen:** centered, single large input, example-topic chips below.
- **Run screen:** stage tracker (top) + results area (below) that fills in.
- **Landscape:** cluster grid/graph; selecting a cluster opens a side panel of
  member papers.
- **Responsive:** mobile-first; stage tracker stacks vertically on small screens;
  landscape degrades to a scannable list of clusters.

## 5. Core Components (owned)

| Component | Responsibility |
|-----------|----------------|
| `SearchBar` | Topic input, submit, example chips |
| `StageTracker` | Four stages with live status + timing |
| `StageCard` | Single stage: label, status color, count/progress |
| `PaperCard` | Title, authors, score, problem/method/results/contribution, arXiv link |
| `ClusterCard` | Cluster name, summary, member count |
| `LandscapePanel` | Clusters + relationships + tensions + open problems |
| `ReadingMap` | Persisted runs; read/to-read toggles |
| `StatusBadge` | Reusable status pill (pending/running/done/error/partial) |
| `EmptyState` / `ErrorState` | Consistent zero/error rendering |

Rules:
- One component = one responsibility. Compose, don't fork.
- All status rendering goes through `StatusBadge` — no ad-hoc colored text.
- Props typed with TypeScript; no `any`.

## 6. Live Pipeline UX

- Each `StageCard` shows: name, status badge, and a metric
  (e.g. "50 candidates", "kept 18", "11 / 18 extracted", elapsed ms).
- Running stage gets a subtle animation (pulse/indeterminate bar), not a
  full-screen blocker.
- As `extract.progress` events arrive, paper cards appear incrementally.
- On `run.complete`, smoothly transition to the landscape (no hard reload).
- On stream drop, show a quiet "reconnecting…" indicator, not an error.

## 7. Accessibility

- Semantic HTML; keyboard-navigable; visible focus rings.
- Color is never the *only* status signal — pair with text/icon.
- Respect `prefers-reduced-motion` (disable pulse animations).
- Color contrast ≥ WCAG AA. Alt text / aria-labels on icon-only buttons.

## 8. Content & Copy

- arXiv IDs and scores in mono.
- Truncate long abstracts/titles with a "show more" affordance, not hard cutoff.
- Empty states explain what to do next; error states name the failed stage and
  offer retry.

## 9. Code Conventions

- File structure: `app/` routes, `components/` UI, `lib/` fetch/SSE helpers,
  `types/` shared types.
- Tailwind utilities in markup; extract repeated clusters into components, not
  `@apply` soup.
- No inline hardcoded colors/spacing outside the token system.
- Keep client components minimal; prefer Server Components where no interactivity
  is needed.
- Format with Prettier; lint with ESLint (run `rtk lint`, `rtk prettier --check`).

## 10. Performance

- Stream/lazy-render paper cards; virtualize long lists if needed.
- Memoize landscape rendering; avoid re-rendering all cards on each SSE event.
- Code-split the landscape/graph view.
- Optimize fonts (subset, `next/font`); avoid layout shift.

## 11. Don'ts

- ❌ `dangerouslySetInnerHTML` with model/arXiv text.
- ❌ Hardcoded hex colors or magic spacing in JSX.
- ❌ Secrets or API keys in client code / `NEXT_PUBLIC_*`.
- ❌ Full-screen spinners that hide already-available partial results.
- ❌ One-off status colors outside `StatusBadge`.
