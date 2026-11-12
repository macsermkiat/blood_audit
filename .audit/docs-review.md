# Documentation review

The documentation needs an accuracy pass before a prose polish. The site describes an older application, repeats implementation history, and makes readers work through summaries and cautions before reaching instructions. Shortening sentences alone would leave the main problems intact.

This review covers the local checkout on 2026-10-19. I used the soundshuman humanize skill, scanned 63 Markdown and MDX files under docs, scored README separately, and inspected the public English and Thai guides, shared site components, product/design notes, and the existing clinical review deck. The deck was read without modification. I compared selected instructions and diagrams with current application code. I did not independently validate clinical policy, inspect patient exports, or check the deployed website.

## Findings in repair order

### 1. P1. The privacy guide says a development export is safe to commit

[The English de-identification guide](/Users/admin/Project_Chatbot_research/blood_audit/docs/src/content/docs/en/developers/deid.mdx:58) describes a bundle as partly real and partly synthetic, then says it contains no real patient identifiers and is safe to commit. The page supplies no verification evidence for that conclusion. The repository's [.gitignore](/Users/admin/Project_Chatbot_research/blood_audit/.gitignore:34) identifies raw and encrypted clinical data as material that must never be committed. The pilot run guide also treats its outputs as PHI.

Remove the safe-to-commit assertion. Describe synthetic test fixtures separately from clinical exports. A directory name or replaced admission number does not establish what the remaining free text contains. This finding concerns the instruction, not a demonstrated historical disclosure.

The same page says deid_redactor installs a traceback hook on import. The current CLI installs its hook explicitly and withholds messages and local values. The guide also omits signed privacy admission, the approved issuer, and the public verification key. It needs a factual rewrite rather than punctuation cleanup.

### 2. P1. The environment guide prints credentials and gives an incompatible DSN

[The verification command](/Users/admin/Project_Chatbot_research/blood_audit/docs/src/content/docs/en/operators/env-vars.mdx:95) selects ANTHROPIC_API_KEY from the process environment and prints its value. The following paragraph warns against logging that value, but the command has already exposed it to terminal capture. Check whether required variables are set without printing their contents.

[The DSN instructions](/Users/admin/Project_Chatbot_research/blood_audit/docs/src/content/docs/en/operators/env-vars.mdx:55) require postgresql+psycopg://, and [the shared checklist](/Users/admin/Project_Chatbot_research/blood_audit/docs/src/components/EnvVarChecklist.astro:23) repeats that example. Current runtime stores pass BBA_DB_URL to psycopg. An offline conninfo_to_dict probe accepted postgresql://synthetic@localhost/blood_audit and rejected the +psycopg variant with ProgrammingError. The suffix belongs to the SQLAlchemy adapter, not this runtime DSN.

Neither language's 27 site pages names BBA_APPLICATION_CONFIG, BBA_PRIVACY_POLICY, BBA_PRIVACY_ARTIFACTS, or BBA_DASHBOARD_AUTH_SECRET. Those settings are documented in the repository's newer production CLI guide. Operators cannot configure the current model and reviewer paths from the site alone.

### 3. P1. Clinical explanations disagree with the implemented boundaries and precedence

[The Hb guide](/Users/admin/Project_Chatbot_research/blood_audit/docs/src/content/docs/en/developers/three-tier-hb.mdx:20), [the shared tier matrix](/Users/admin/Project_Chatbot_research/blood_audit/docs/src/components/HbTierMatrix.astro:19), and [the policy summary](/Users/admin/Project_Chatbot_research/blood_audit/docs/src/content/docs/en/clinical/policy-summary.mdx:57) describe a strict Hb < 7.0 rule. [The current classifier](/Users/admin/Project_Chatbot_research/blood_audit/src/bba/deterministic_classifier/classifier.py:340) implements Hb <= 7.0 before MTP and unknown-context routing. The equality case changes the result, so this is a substantive mismatch.

[The review deck](/Users/admin/Project_Chatbot_research/blood_audit/clinical_audit_pitch_deck.html:1383) places unknown context before that universal low-Hb rule and shows a procedure within the next 72 hours as automatically appropriate. [The current site summary](/Users/admin/Project_Chatbot_research/blood_audit/docs/src/content/docs/en/clinical/policy-summary.mdx:68) instead describes that timing route as legacy mode and says it defers to model review. The deck also presents the older combined orthopedic/cardiac cohort, while the developer guide documents its replacement.

Reconcile the code, clinical diagrams, tables, and deck against one approved rule specification before clinicians use them for sign-off. Preserve policy values until that reconciliation is explicit. These are code/document consistency findings, not a judgment that a transfusion threshold is clinically correct.

### 4. P1. The published-site source describes unfinished commands that now work

[The landing page](/Users/admin/Project_Chatbot_research/blood_audit/docs/src/content/docs/en/index.mdx:30), [first-run guide](/Users/admin/Project_Chatbot_research/blood_audit/docs/src/content/docs/en/operators/first-audit-run.mdx:22), and [audit reference](/Users/admin/Project_Chatbot_research/blood_audit/docs/src/content/docs/en/operators/audit-command.mdx:9) say bba audit only validates input and writes markers. [The integration page](/Users/admin/Project_Chatbot_research/blood_audit/docs/src/content/docs/en/operators/integration-seams.mdx:9) says evaluate, serve-dashboard, and sentinel intentionally fail and tells readers not to fix them. The install guide repeats this advice.

Current application code writes AuditRows, and the CLI implements those commands. The newer [production CLI guide](/Users/admin/Project_Chatbot_research/blood_audit/docs/production-cli.md:1) describes that behavior. The prior implementation pass updated this guide and README but left the site pages behind. That synchronization gap needs correction.

This is spread across prose and shared components. Eleven pages in each language mention Phase 1.5; PipelineDiagram, AudienceTiles, and EnvVarChecklist also contain the older status. Updating only the homepage would leave contradictory instructions elsewhere.

### 5. P2. The testing page teaches the fixture behavior that was removed

[The testing guide](/Users/admin/Project_Chatbot_research/blood_audit/docs/src/content/docs/en/developers/testing.mdx:43) recommends wrapping container startup in a catch that converts failures to skips. Current tests/conftest.py distinguishes optional local infrastructure from required database execution. Setup and teardown defects must propagate. CI sets BBA_REQUIRE_POSTGRES_TESTS=1 and enforces 80 percent source coverage.

The page also includes dated pass/skip totals and a categorical claim that test typing errors cannot mask a source bug. Replace those assertions with current commands, the required gates, and the scope of the advisory typing check. Keep dated results in a verification record rather than a running how-to.

### 6. P2. The layout repeats the summary before readers reach the task

All 54 English and Thai MDX pages use AtAGlance, including short reference pages. The first-run page opens with an introduction, four summary cards, and a phase-scope notice before its first task. The cards make long paths wrap awkwardly into narrow columns. In the 600px Thai view, three cards occupy one row and the fourth occupies a second row with the same height, adding a large empty area.

The desktop landing page puts its audience links near the bottom, after implementation details and the full pipeline diagram. In the observed 1440 by 869 viewport, those links began about 3,059 CSS pixels down a 3,437px page. The main button serves operators, but developers and clinical reviewers must scroll through most of the page to find their dedicated routes.

Move the audience links next to the introduction. Use a short prerequisite list and a copyable command on task pages. Reserve summary cards for pages where the information helps a reader choose a path. Keep a concise scope statement and place detailed cautions beside the action they govern. The problem is the hierarchy and repetition; a new color palette would not resolve it.

### 7. P2. Internal terminology dominates the prose

Examples include 'can terminal the order', 'deployment-shaped glue', 'analysis leg', 'wired in Phase 1.5', and 'fail loud by design'. The phrases describe engineering work rather than what an operator or clinician needs to do. The integration page's 'Do not fix them' is particularly defensive. The clinical page's 'Anti-hallucination guarantees' promises more than a description of grounding checks establishes.

The dominant humanize patterns are diff-anchored writing, repeated announcements, excessive structure, and emphatic guarantees. Heavy bolding and status labels reinforce the problem. Technical reference pages should retain exact identifiers, but explain them in ordinary sentences. Thai pages should retain command and field names while translating the surrounding instructions instead of mixing English verbs into each sentence.

### 8. P2. The canonical glossary accumulates history and conflicting facts

[CONTEXT.md](/Users/admin/Project_Chatbot_research/blood_audit/docs/CONTEXT.md:34) is 3,628 lines and names ten canonical tables, including an obsolete background table. The current ingest schema names thirteen. Its drift section says unknown columns are always fatal, while normalization can project dropped columns before validation. Several passages tell agents how to work or narrate ticket history.

Separate current module contracts from dated design decisions. Keep the current contract once per module and link to it from the site. Archive plans with their dates and status intact rather than rewriting them to look current. Linking to an append-only glossary does not prevent its contents from becoming stale.

### 9. P3. README and the product notes need tighter editorial boundaries

README is 672 lines. Its HOSxP join examples occupy the section from line 208 through line 554, between the input description and architecture. The main entry page carries installation, historical pilot paths, long SQL, architecture, flags, and build history. A new reader has to decide which material is current and relevant.

Keep purpose, current scope, one supported quickstart, and links in README. Move SQL examples and build history into clearly named references. PRODUCT.md and DESIGN.md can retain concrete audience and layout decisions, but phrases such as 'Restraint reads as competence', 'Color is deployed as evidence', and 'Pills are evidence, not decoration' add slogans rather than testable guidance. Replace them with the specific visual decisions already present in those files.

## Ranked rewrite queue

This order reflects reader risk and editorial need. It is separate from the automated language score.

| Order | Files or group | Main work |
|---|---|---|
| 1 | English/Thai de-identification and environment guides; EnvVarChecklist | Correct the commit instruction, secret-check command, DSN, admission setup, and reviewer configuration. |
| 2 | Hb policy summaries, tier matrix, decision flow, clinical deck | Reconcile implemented boundaries and precedence with the approved clinical specification. |
| 3 | English/Thai landing, overview, install, first run, audit command, integration page; shared status components | Describe the current application and give each audience a direct route. |
| 4 | English/Thai testing pages | Document required checks and failure behavior without dated counts or broad guarantees. |
| 5 | README, production CLI guide, CONTEXT.md | Establish current entry points, split historical material, and reduce dense implementation prose. |
| 6 | PRODUCT.md, DESIGN.md, archived plans | Replace slogans in current guidance; preserve dated plans as history. |

## Automated language scan

sloplint is a screening aid, not evidence of authorship or factual quality. Its highest scores here mostly reflect dash punctuation. Thai word counts and English vocabulary rules make its Thai rankings unsuitable for comparison with English prose. A low score also misses serious factual problems: the de-identification page scored 11 and the integration page 22.

| File | Score | Main automated signal |
|---|---:|---|
| docs/CONTEXT.md | 59 | 324 dash matches; repeated technical phrasing |
| docs/transfusion-policy-application-plan.md | 54 | 77 dash matches; five emoji matches |
| docs/source_mapping_design.md | 46 | 26 dash matches |
| English developer testing page | 45 | Six dash matches |
| English contributing page | 41 | Four dash matches |
| English audit-store page | 40 | Seven dash matches |
| English install page | 39 | Three dash matches |
| English first-run page | 37 | Six dash matches |

README scored 18 and production-cli.md scored 7. Their lower scores do not remove the need for structural editing. Full scanner output is in docs-sloplint.json and readme-sloplint.json beside this report.

## Verification and limits

The Astro production build passed and generated 55 pages. A local link check inspected 56 built HTML files, including the static redirect, and found no missing internal page or anchor targets. The browser review covered the English landing, install and first-run pages, and the Thai first-run page at a narrower width. The build log is docs-build.log and the link result is docs-internal-links.txt.

I did not run the credential-printing command, use a live model, connect to a database, or change application code. The DSN probe parsed synthetic strings without a network connection. Existing documentation and the untracked review deck remain unchanged. This report and the scanner/build artifacts are local review files.

The humanize skill's repository-audit mode calls for a ranked review and says to rewrite only files the user selects. No bulk rewrite was applied. The useful next unit is the first group in the queue, followed by the current user journey in both languages. Clinical rule text needs factual reconciliation before stylistic editing.
