# PRODUCT — blood_audit

## What it is

A post-hoc quality-improvement audit pipeline for inpatient adult red blood cell
(RBC) transfusions at King Chulalongkorn Memorial Hospital (KCMH). It reads
finished HOSxP exports, classifies each completed transfusion against a 3-tier
hemoglobin policy plus KCMH PR 17.2 and AABB 2023, and writes one stored row
per transfusion to the audit store. Output feeds the monthly QI committee
report.

The Astro Starlight site at <https://macsermkiat.github.io/blood_audit/> is the
documentation surface for that pipeline. It is bilingual (English, Thai) and
serves three audiences from one tree.

## Register

`brand`. The site is the product surface for every visitor. Operators read it
to install and run the tool; developers read it before touching the code;
clinical reviewers read it to understand the audit scope. The landing page is
the visual anchor. Product pages (operator how-tos) stay restrained; the
landing is allowed slightly more presence.

## Users

1. **Operators (KCMH IT).** Install the package, set three environment
   variables, run `bba ingest` then `bba audit`, hand the dataset to the QI
   committee. Reading on a desktop, usually a hospital workstation. They need
   working CLI examples, exact env-var purposes, and clear exit-code
   semantics. They do not need theory.
2. **Developers maintaining `src/bba/`.** Reading on a desktop or laptop while
   editing the code. They need the architectural map (12 pipeline stages),
   the module glossary, and the deterministic-classifier rules. They need
   precise vocabulary, not friendly summaries.
3. **Clinical reviewers and QI committee.** Reading on tablet or laptop in a
   meeting. They need to understand that this is **not** Software as a
   Medical Device, what the five stored classifications mean, and what
   appears in the monthly report. They never run the CLI.

## Strategic principles

- **Quality improvement, not clinical decision support.** This is the
  load-bearing framing. The site must not adopt a SaMD or point-of-care
  aesthetic. No urgent red alerts, no "act now" CTAs, no real-time
  dashboards-as-hero. The `QiNotSamdBanner` anchors the landing page and
  appears on every page where someone might misread scope.
- **Restraint reads as competence in a hospital context.** Tools that influence
  clinical work earn trust by looking calm and verifiable. Confidence without
  showiness. Hover effects are subtle. Color is deployed as evidence, not
  decoration.
- **Bilingual by design.** Thai content uses the same layout but a font stack
  that handles Thai diacritics (Noto Sans Thai). Line-height generous enough
  for stacked diacritics. Six priority Thai translations exist; the rest fall
  back to English via Starlight's localized notice.
- **Synthetic placeholders only.** All examples use `PHI_xxx_001`, `PHI_xxx_002`,
  `PHI_xxx_003`. Never substitute real values, even in screenshots.
- **No emojis.** Project and user global rule.
- **Three audiences, one tree.** The landing routes each audience to its own
  doorway. Don't paper over the differences with one generic intro.

## Tone

Calm, technical, exact. Short sentences. No marketing voice. The pipeline does
something specific and audit-grade; the prose should do the same. No em
dashes. Use commas, colons, periods, parentheses. Reading age: assume a
clinical informaticist or a senior engineer, not a generalist consumer.

## Anti-references

- Consumer SaaS landing pages with hero gradients, big numbers, and
  testimonial walls.
- Crypto / AI hype aesthetics: neon-on-black, glowing accents, animated mesh
  backgrounds.
- Clinical-decision-support UI: full-screen red alerts, pulsing badges, "high
  severity" banners.
- Generic Starlight default with no theming choices made.
- "AI agent platform" templates: gradient text, animated terminal, side-stripe
  callouts on every card.

## Visual lanes to avoid

- Editorial / Vercel-clone: thin sans + huge serif + monochrome. Already
  saturated; signals "AI workflow tool".
- Terminal-native dark mode: monospace everywhere, scanlines, green-on-black.
  Signals "developer tool" but mismatches the clinical audience.
- Medical-blue-and-teal: signals "telehealth" or "consumer health". Wrong
  register; we are a hospital internal tool.

## Where the design lives

- Astro Starlight under `docs/`.
- Single accent `#b91c1c` (KCMH red) in light, `#ef4444` in dark, but expressed
  as OKLCH-tinted neutrals plus the accent role.
- Seven shipped components under `docs/src/components/` are the polish
  surface.
- Tailwind is allowed as an Astro integration scoped to `docs/`. CSS variables
  in `src/styles/custom.css` carry the theme tokens. Tailwind utilities carry
  layout and component composition.
