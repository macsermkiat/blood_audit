# DESIGN — blood_audit docs

Locked decisions for the Astro Starlight documentation site at
<https://macsermkiat.github.io/blood_audit/>. Companion to `PRODUCT.md`.

## Color

Restrained color strategy. Tinted neutrals carry the surface; one accent role
is reserved for evidence (active states, critical pills, the QI scope banner).
Pills carry tier semantics and never compete with the accent for the eye.

Hue base: 28 (warm red). Neutrals are tinted toward that hue at chroma 0.005
to 0.01 so the page never reads as cold white.

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

Pills are evidence, not decoration. Chroma is deliberately low so they read
as labels, not buttons. Same role colors in both themes; lightness inverts.

| Pill | Light bg / fg | Dark bg / fg | Use |
| --- | --- | --- | --- |
| `ok` | `oklch(94% 0.04 150)` / `oklch(34% 0.12 150)` | `oklch(28% 0.05 150)` / `oklch(86% 0.10 150)` | `APPROPRIATE`, `auto-classify` |
| `warn` | `oklch(94% 0.05 70)` / `oklch(40% 0.14 70)` | `oklch(28% 0.06 70)` / `oklch(88% 0.10 70)` | `NEEDS_REVIEW`, Phase 1.5, opt-in, `LLM review`, integration seam |
| `err` | `oklch(94% 0.05 28)` / `oklch(38% 0.16 28)` | `oklch(28% 0.06 28)` / `oklch(88% 0.10 28)` | `POTENTIALLY_INAPPROPRIATE`, required (Phase 1) |

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

Starlight's three-column layout is kept (sidebar, content, table of contents).
Component-level spacing is rhythmic, not uniform.

- Container max-width for content: 75ch (Starlight default holds).
- Vertical rhythm: `space-y` of 1.5rem between component blocks, 0.75rem
  inside cards. No same-spacing everywhere.
- Cards are used only when the affordance is genuinely needed
  (`CliCommandCard`, `IntegrationSeamCard`, `EnvVarChecklist` rows). No
  nested cards. No card-grid of identical icon+heading+text.
- The `PipelineDiagram` is a grid of 12 stages but is grouped visually into
  the **ingest leg** (stages 1 to 5, Phase 1 today) and the **analysis leg**
  (stages 6 to 12, wired in Phase 1.5), with a subtle leg boundary. The
  index numbers carry the visual rhythm. This breaks the identical-card-grid
  ban via grouping plus index prominence.

## Components

Seven shipped Astro components under `docs/src/components/`. All are semantic
shells with typed props. After polish they share these contracts:

| Component | Role | Polish target |
| --- | --- | --- |
| `QiNotSamdBanner` | Above-the-fold scope statement | Stronger presence on landing, calmer presence on inner pages via `variant="compact"` |
| `PipelineDiagram` | 12-stage architectural map | Grouped into 2 legs, ordinal index prominent, subtle reveal-on-scroll |
| `CliCommandCard` | CLI invocation reference | Code-first layout, terminal-style example block, semantic sections, monospace command treated as primary label |
| `EnvVarChecklist` | 3 env vars with phase pills | Compact rows, phase pill aligned right, code example block tight |
| `HbTierMatrix` | 3-tier Hb policy table | Tier number column, rule-output pill, routing pill, note paragraph kept |
| `IntegrationSeamCard` | Warning callout for 4 by-design CLI errors | Replace `border-left` side-stripe with full hairline border + soft warn tint + warn pill |
| `QuoteGroundingLayers` | 7-layer stack, layer 7 dashed | Ordered stack with leading index, layer 7 marked dashed + opt-in pill |

## Motion

Allowed and subtle. Disabled when `prefers-reduced-motion: reduce`.

- **Reveal-on-scroll**: opacity 0 → 1 plus translate-y 0.5rem → 0. Duration
  180ms. Easing `cubic-bezier(0.22, 1, 0.36, 1)` (ease-out-quart). Used on
  the landing page only, on `QiNotSamdBanner`, `PipelineDiagram`, and the
  three audience entries.
- **Hover transitions**: border-color and background-color only. Duration
  120ms. Same easing. No transform. Used on `CliCommandCard`,
  `EnvVarChecklist` rows, audience entry tiles.
- **Focus rings**: 2px outline in accent color, offset 2px. No animation.
- **Animations on layout properties**: forbidden. Don't animate width,
  height, margin, padding, top, left, right, bottom.

## Absolute bans (project-specific)

In addition to the impeccable shared bans:

- **Side-stripe borders.** The existing `.ba-card--warning` uses
  `border-left: 4px solid` and must be rewritten with a full 1px border in
  the warn pill hue plus a soft warn background tint.
- **Gradient text or gradient backgrounds.** None.
- **Glassmorphism.** None.
- **Hero metric template.** No "98% appropriate, 12 wards audited" big-number
  hero. We are a QI tool, not a vanity dashboard.
- **Identical card grids.** The pipeline diagram uses grouping plus ordinal
  indices to escape this; nothing else uses a card grid.
- **Modals.** None.

## Accessibility

- WCAG AA contrast on every text-on-background pair (verified for the OKLCH
  values above).
- All interactive elements reachable by keyboard. Focus visible.
- `aria-label` on each component's outer landmark element (already in place).
- Pills use semantic text plus background color; never color alone conveys
  meaning.
- `prefers-reduced-motion` honored.

## Tooling

- Astro 5 + Starlight 0.34.
- Tailwind 4 via `@astrojs/tailwindcss` Vite plugin (scoped to `docs/`).
  Tailwind utilities carry layout and composition; CSS variables in
  `custom.css` carry tokens.
- `@fontsource/noto-sans-thai` for self-hosted Thai font (subsets: `thai`,
  `latin`).
- Node 20 in CI (`deploy-docs.yml`). Local dev on Node 20+.
