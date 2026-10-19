# Clinical and reference documentation rewrite

Completed in `/tmp/bba-docs-clinical`. Patch is `/tmp/bba-docs-clinical.patch`.

Changed 16 owned files. Both languages now describe retrospective QI scope, actual classifier boundaries and routing, report exclusions and per-row provenance, real reviewer decisions, and the 13 canonical ingest tables. Removed repeated AtAGlance cards. The clinical banner no longer makes an unsupported regulatory classification. Historical changelog facts and dates remain, with explicit separation from current status.

Source comparisons used `deterministic_classifier/classifier.py`, `cohort_detector/rules.py` and `models.py`, `hb_lookup/lookup.py`, `audit_pipeline/pipeline.py` and `replay.py`, `platelet_classifier/classifier.py`, `platelet_guardrail/models.py`, `prompt_builder/system_prompt.py`, `feature_flags.py`, `application/config.py` and `source_snapshot.py`, `deid_redactor/admission.py`, `dashboard/auth.py`, `adjudication.py`, and `app.py`, and report builder, aggregate, CSV writer, models, and generator.

Corrections include universal Hb <=7 before MTP/UNKNOWN; inclusive cohort floors; ESRD-EPO7.0; orthopaedic surgery8.0 without cardiac-history requirement; retained cardiopulmonary label now heart-disease only; >=10review route after earlier rules; legacy upcoming procedure defers rather than clears. Distinguishes selected seven-day Hb from LLM's bounded24-hour minimum. Platelet10/20/50/80k inclusive,100k strict. Notes default-disabled platelet LLM can leave incomplete audits rejected by application; queue covers committed unresolved cases only.

Root and site report contracts now name five committee CSVs plus N physician CSVs, PDF, provenance.json; RBC-only aggregation; full-RBC per-row sidecar including excluded rows; mixed model/prompt summaries; actual deterministic marker and unapplied-redactor sentinels; explicit projection requirement for unresolved labels; committee-PDF physician visibility and delivery-access limits. No deterministic answer or clinical validation guarantees.

Static checks pass. `/tmp/check_clinical_docs.py` verified all16files for dash/card removal, MDX frontmatter, Astro delimiters, all13canonical names against schema AST, and all glossary module paths. It initially caught a missing Thai changelog description key, which was fixed before the final passing run. `git diff --check` passes.

Bundled humanize scores for English pages are policy0, what-this-is1, report-output19, glossary20, report-schema14, changelog9. Root report contract10. Thai scores are not used as a language-quality criterion. No em/en dashes or new emojis remain in touched files.

Integrated site build, internal-link and browser checks remain with root. No app behavior changed, no application tests repeated, no live provider calls, no data exports read, no dependencies installed, and no commit made.

Minimize Reader Load removed repeated summary cards and separated current contracts from historical notes. Prove It Works required direct source comparison and the deterministic registry/path checks before exporting the patch. The roots can independently review and validate the rendered MDX/Astro output.

Changed files are docs/report-schema.md, all six English and six Thai clinical/reference MDX pages, and AlgorithmDecisionFlow.astro, HbTierMatrix.astro, QiNotSamdBanner.astro.
