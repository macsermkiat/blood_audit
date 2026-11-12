# Documentation correction verification

The documentation rewrite is complete locally on `codex/documentation-cleanup`, based on merged application commit `b343d9a`. No commit, push, pull request, or deployment was performed for this rewrite.

## Changes

All 54 English and Thai site pages were revised against current source contracts. Operator guides distinguish ingest from committed audits, local execution from admitted model execution, and supported commands from dataset-specific pilot tools. Credential checks print only names and presence, runtime DSNs use `postgresql://`, and setup includes signed privacy admission, issuer key separation, PostgreSQL checkpoints, trusted reviewer identity, HTTPS origin, and immutable-run recovery.

Clinical and developer guides describe the implemented inclusive Hb boundaries, universal-rule precedence, ESRD 7.0, orthopedic surgery 8.0, and the strict platelet 100k model exception. They explain that the default-off platelet model route can leave expected work uncommitted and prevent CLI completion. Monthly reports and aggregate scorecards retain RBC scope; the queue can include committed unresolved cases from either component. These corrections describe code behavior without changing clinical policy or claiming clinical validation.

README is reduced from 672 to 89 lines. The current module reference is reduced from 3,628 to 127 lines. Full earlier prose is preserved in `docs/history`, with relative Markdown links rebased and explicit historical notices. SQL examples have a separate export reference. Historical plans retain their text and original paths with a notice directing readers to current contracts. Product and design notes now describe concrete reader tasks and current components.

Repeated AtAGlance blocks were removed from all 54 pages, and three unused summary/command/seam components were deleted after checking references. The landing page puts audience routes before processing detail. The Thai reference notice is shorter. The existing deck retains its styling, navigation, and unrelated content while threshold, cohort, precedence, review, and grounding descriptions are corrected. A slide-specific margin keeps the decision cards clear of their footer. Its before/after patch is `deck-docs-corrections.patch`; the original untracked deck was not replaced wholesale.

## Verification

| Check | Result |
| --- | --- |
| Production Astro build | 55 pages built successfully |
| Astro diagnostics | 12 files, zero errors, warnings, or hints |
| Internal navigation, stylesheet, and anchor checks | 56 generated HTML files, 2,614 local href checks, zero broken targets |
| Current Markdown and MDX repository links | 61 files, zero missing targets |
| English and repository prose scan | 34 current files, scores 0 to 20, below the humanize target of 25 |
| Rewritten prose punctuation | No em or en dashes outside code in those 61 files |
| Source consistency and safety review | Independent approval, all findings closed |
| Whitespace and scope | `git diff --check` passes; no application source, tests, migrations, dependencies, or workflows changed |

The link checker is `.audit/check_docs_rewrite.py`. It checks navigable anchors and stylesheets rather than canonical/alternate head metadata. A first broader pass flagged Starlight's generated 404 canonical and language-alternate metadata, which does not map to built normal pages. External web links were not checked. Historical records are intentionally excluded from current-guide prose and repository-link scoring. English lexical scores were not used to assess natural Thai.

The first build rejected a bare `<100` comparison in English and Thai platelet prose. Wrapping that comparison as code corrected the MDX syntax. Independent review then corrected misleading hashed-identifier descriptions, an ambiguous Thai required-database-test sentence, remaining deck safety-rule contradictions, and an overbroad grounding-failure claim. A later passing attempt can supersede an earlier failed attempt. The independent report records the corrections and final approval.

The application test suite was not repeated because this change does not alter application behavior. Existing source checks and clinical boundary tests were used as reference, not claimed as fresh execution. Operator snippets and configuration examples received bounded source/parser checks. The independent reviewer executed both environment-presence snippets with sentinel secrets and confirmed that values were absent from output. Its offline DSN probe required no connection.

## Rendered checks

The real desktop landing page was inspected at an actual 1440 by 900 viewport. Audience routes begin around 632 CSS pixels down the page and all audience links end around 794 pixels, inside the first viewport. Total page height is 1,665 pixels. The original audit observed the audience section around 3,059 pixels down a 3,437-pixel page, at a slightly different height. These are observations, not a controlled performance comparison.

The Thai first-run and configuration pages and English clinical policy page were checked at an actual 390-pixel width. Document width remained 390 pixels with no page-level horizontal overflow. The first-run page has no repeated summary cards. Screenshots are `docs-landing-desktop.png` and `docs-thai-narrow.png`.

The initial in-app viewport request silently retained 319 pixels, so it was not counted as desktop evidence. A documented CDP viewport override produced the checked dimensions. Its first permission check timed out; the one permitted retry succeeded without weakening a browser control. The normal screenshot wrapper scaled the narrow image incorrectly despite correct DOM dimensions; exact clipped CDP captures produced the readable saved screenshots.

The deck was served temporarily as a copied build artifact on the same loopback preview. Actual slide navigation was used after an anchor opened with the heading clipped. Measurements taken during smooth scrolling were discarded. Settled slide 8 has a top of 0, heading around 60 pixels, last card bottom 849, and footer top 870 at 1440 by 900. Slide 10's corrected text was also inspected after navigation settled. The saved deck screenshots are `docs-deck-decision.png` and `docs-deck-review.png`.

The temporary deck copy was removed. Browser metric overrides were cleared, the temporary tab closed, and the preview server stopped. Existing decoder and agent files still match their recorded pre-task hashes. No patient export, provider call, production identity integration, clinical validation, or application deployment was used.

## Review trail

The canonical decision trail is `docs-rewrite-decisions.tsv`. Its checkpoints were reconstructed from the matching transcript and artifacts after execution; their timestamps record when those checkpoints were logged. The earlier three-column planning note remains in `docs-rewrite.tsv` and is explicitly superseded rather than rewritten.

Minimize Reader Load led to removing repeated summaries, moving audience routes up, and separating current contracts from history. Prove It Works required the actual built-site link checks and observed desktop, narrow, and slide rendering. Humanize guided the prose; source comparisons determined facts. Cross-model trail review is recorded separately before handoff.

The GPT-5.6 trail review found no skipped or overclaimed verification. It requested stronger evidence pointers for the failed build and browser retry, and durable copies of the three temporary worker reports. Append-only evidence rows supply the exact matching transcript ranges, final build log, and workspace report copies. The original rows and planning note remain intact.

The subsequent design-hook follow-up is recorded in `deck-design-triage.md`. It corrects body leading and footer insets in the standalone deck, with a separate scoped code review and deck-only detector exceptions for verified padding false positives and the retained palette. `deck-design-final-metrics.json` and `deck-design-desktop.png` record the later rendered state. The original documentation verification above describes its own earlier checkpoint.
