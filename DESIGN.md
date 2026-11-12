# Documentation design

Visual reference for the Astro Starlight documentation site at
<https://macsermkiat.github.io/blood_audit/>. Companion to `PRODUCT.md`.

## Color

Use warm neutral backgrounds and a red accent for active states and the scope notice. Classification labels combine text with muted colors. Hue base is 28; neutral chroma ranges from 0.005 to 0.01. Keep the existing token values below when editing layout.

### Light theme

| Role | OKLCH | Hex reference |
| --- | --- | --- |
| Background, primary | `oklch(99% 0.005 28)` | near-white, warm-tinted |
| Background, raised | `oklch(97% 0.008 28)` | nav and card surfaces |
| Background, muted | `oklch(94% 0.01 28)` | code blocks, table head |
| Border, hairline | `oklch(90% 0.012 28)` | 1px card and table borders |
| Border, strong | `oklch(80% 0.02 28)` | divider lines |
| Text, primary | `oklch(22% 0.015 28)` | body copy |
| Text, muted | `oklch(48% 0.02 28)` | captions, secondary |
| Accent | `oklch(43% 0.18 28)` (≈ `#b91c1c`) | active states, QI banner border |
| Accent, soft | `oklch(94% 0.05 28)` | QI banner background |

### Dark theme

| Role | OKLCH | Hex reference |
| --- | --- | --- |
| Background, primary | `oklch(18% 0.015 28)` | warm near-black, never `#000` |
| Background, raised | `oklch(22% 0.018 28)` | nav, card surfaces |
| Background, muted | `oklch(26% 0.02 28)` | code blocks, table head |
| Border, hairline | `oklch(34% 0.025 28)` | 1px card and table borders |
| Border, strong | `oklch(44% 0.03 28)` | divider lines |
| Text, primary | `oklch(94% 0.01 28)` | body copy, never `#fff` |
| Text, muted | `oklch(70% 0.015 28)` | captions, secondary |
| Accent | `oklch(64% 0.20 28)` (≈ `#ef4444`) | active states, QI banner border |
| Accent, soft | `oklch(24% 0.06 28)` | QI banner background |

### Pill palette

Status pills are text labels, not buttons. Keep their colors muted and retain the classification text in both themes.

| Pill | Light bg / fg | Dark bg / fg | Use |
| --- | --- | --- | --- |
| `ok` | `oklch(94% 0.04 150)` / `oklch(34% 0.12 150)` | `oklch(28% 0.05 150)` / `oklch(86% 0.10 150)` | `APPROPRIATE`, `auto-classify` |
| `warn` | `oklch(94% 0.05 70)` / `oklch(40% 0.14 70)` | `oklch(28% 0.06 70)` / `oklch(88% 0.10 70)` | `NEEDS_REVIEW`, opt-in, model review, deployment prerequisite |
| `err` | `oklch(94% 0.05 28)` / `oklch(38% 0.16 28)` | `oklch(28% 0.06 28)` / `oklch(88% 0.10 28)` | `POTENTIALLY_INAPPROPRIATE`, required input |

## Typography

Two families. EN body uses a system sans (Inter falls back to system). Thai
body uses Noto Sans Thai, self-hosted via `@fontsource/noto-sans-thai`.
Monospace is JetBrains Mono falling back to system mono.

### Font stack

- **Sans** (`--ba-font-sans`): `"Inter", -apple-system, "Segoe UI", "Roboto",
  "Noto Sans Thai", "Helvetica Neue", sans-serif`. The Thai font in the
  middle of the stack means Thai glyphs render in Noto Sans Thai while Latin
  glyphs use Inter, in mixed-language paragraphs.
- **Mono** (`--ba-font-mono`): `"JetBrains Mono", "Fira Code", ui-monospace,
  "SF Mono", Menlo, Consolas, monospace`. Code blocks and inline code only.

### Scale

Ratio 1.250 (major third). Base 16px. Cap body line length at 70ch.

| Step | Size | Weight | Use |
| --- | --- | --- | --- |
| Display | 2.441rem | 700 | h1 on splash hero only |
| H1 | 1.953rem | 700 | page title |
| H2 | 1.563rem | 700 | section heading |
| H3 | 1.25rem | 600 | sub-section |
| H4 | 1.0rem | 600 | component heading inside cards |
| Body | 1.0rem | 400 | paragraph |
| Small | 0.875rem | 400 | captions, pill, footer |
| Code | 0.875rem | 500 | inline and block code |

### Thai diacritic handling

- Line-height 1.65 in Thai content (vs 1.55 in EN) so stacked diacritics
  (saraI + maiHanAkat, etc.) do not collide with the line above.
- Letter-spacing 0 in Thai. No negative tracking.
- Font-feature-settings disable Latin liga in Thai to avoid odd substitutions.

## Layout

Keep Starlight's sidebar, content column, and table of contents. The landing page places audience links immediately after the short scope notice. Processing diagrams belong in the architecture guide.

Task pages start with prerequisites and commands. Reference pages use tables where readers need to compare fields or rules. A summary must add information rather than repeat the introduction. Long paths and identifiers must wrap or scroll within their own container without widening the page.

Thai content uses the same navigation and layout. Its English-reference notice stays short. Check narrow English and Thai views separately because wrapping differs.

## Components

Shared components live in `docs/src/components/`. Reuse them when they clarify the content. Their text must describe current behavior.

| Component | Contract |
| --- | --- |
| `AudienceTiles` | Direct links for operators, developers, and clinical reviewers |
| `QiNotSamdBanner` | Retrospective quality-improvement scope, with a compact inner-page variant |
| `PipelineDiagram` | Processing stages and boundaries without obsolete implementation-status labels |
| `EnvVarChecklist` | Settings grouped by the task that needs them; no credential values in checks |
| `HbTierMatrix` | Inclusive implemented boundaries, routing, and preceding-rule caveats |
| `AlgorithmDecisionFlow` | Ordered current rules with separate legacy flag-off conditions |
| `QuoteGroundingLayers` | Implemented checks, stopping behavior, and verification limits |
| `ThaiAbridgedBanner` | Concise link to the matching full English reference |

## Motion

Disable motion when `prefers-reduced-motion: reduce`. Scope and processing components can reveal with opacity from 0 to 1 and vertical translation from 0.5rem to 0 over 180ms. Use `cubic-bezier(0.22, 1, 0.36, 1)`. Required instructions must remain readable.

Hover transitions on environment rows and audience links use 120ms with the same easing. Keep them limited to color and border or background color. Focus uses an unanimated 2px accent outline with a 2px offset. Do not animate layout properties such as width, height, margin, padding, or position.

## Accessibility and constraints

Use visible keyboard focus, semantic headings, and descriptive links. Convey classification with text as well as color. Check text contrast against WCAG AA when changing tokens; token tables alone do not prove rendered contrast.

Honor `prefers-reduced-motion`. Keep instructional content readable when JavaScript is unavailable. Avoid gradients, glass effects, urgent alert styling, decorative metrics, nested cards, and modals in the documentation.

## Tooling

- Astro 5 + Starlight 0.34.
- Tailwind 4 via the `@tailwindcss/vite` plugin (scoped to `docs/`).
  Tailwind utilities carry layout and composition; CSS variables in
  `custom.css` carry tokens.
- `@fontsource/noto-sans-thai` for self-hosted Thai font (subsets: `thai`,
  `latin`).
- Node 20 in CI (`deploy-docs.yml`). Local dev on Node 20+.
