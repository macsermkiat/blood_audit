# Developer documentation rewrite

Rewrote all nine developer pages in English and Thai and updated PipelineDiagram and QuoteGroundingLayers. The patch is /tmp/bba-docs-developers.patch. No source behavior, dependency, patient-data file, or external application was changed. Nothing was committed.

## Corrected contracts

- De-identification distinguishes local approved redaction from request-producing code. It documents exact evidence and prompt binding, Ed25519 public-key verification, private-key issuer separation, operator-supplied backend, failed-admission local review, immutable completed results, and synthetic versus clinical fixtures. The unsupported partly-real bundle safe-to-commit claim is removed.
- Exception handling now describes the explicit CLI hook and withholding arbitrary exception messages and local values. It no longer claims import-time regex protection.
- Architecture, pipeline, and store pages describe current application wiring, source snapshots, committed AuditRows, exact manifests and cache identity, ordered attempts, real recovery verification, and uncertain submissions that require reconciliation. Runtime DSNs use bare postgresql://.
- Hb text follows classifier.py, detector.py, and rules.py. Global Hb <=7 is before assigned MTP and UNKNOWN. Cardiac surgery <=7.5, orthopedic <=8, heart disease <=8, and ESRD_EPO <=7 are explicit. The high-Hb >=10 route remains. Cohort assignment order is documented separately from classifier order. Legacy pre-op deferral uses Hb strictly above the threshold. Stored enum values are distinguished from member names.
- Platelet text removes mutually contradictory auto-clear claims. The default-on prophylaxis rule is conditional, fresh, and inclusive <=10. The model path is separately default-off. Unresolved flag-off platelet work remains uncommitted and prevents application completion. Prompt indication boundaries <=10/20/50/80 and strict <100 are preserved.
- Testing follows the shared fixture and CI, including BBA_REQUIRE_POSTGRES_TESTS=1, 80 percent coverage, specific optional-infrastructure skips, propagated startup/migration/teardown errors, advisory test typing limitations, and offline model tests. The old nonexistent tests/unit/cassettes directory reference is replaced with the actual temporary cassette builders in test_llm_client.py and test_application_cassette.py.
- Removed all 18 repeated AtAGlance blocks. Contributions guidance retains useful local conventions and removes obsolete fixed counts and guaranteed bot timing.

## Source comparisons

Read current application/audit.py, source_snapshot.py, production-cli.md, deid_redactor/admission.py and exports, cli/phi_scrubber.py, quote_grounder/verifier.py and layers.py, audit_store/models.py and store.py and cold_storage.py, audit_pipeline/pipeline.py and manifest.py and resume.py and replay.py, llm_client/request_identity.py, platelet_classifier/classifier.py, platelet_guardrail/models.py and guardrail.py, prompt_builder/system_prompt.py, cohort_detector/detector.py and rules.py and models.py, hb_lookup/lookup.py, feature_flags.py, tests/conftest.py, CI, and cassette builders.

## Checks

- git diff --check passes.
- All 23 explicit repository paths referenced in the rewritten pages exist.
- All 20 owned files have no em dash or en dash, AtAGlance reference, or Phase 1.5 text.
- Bundled humanize sloplint English scores range from 0 to 13. JSON is /tmp/developer-docs-en-sloplint.json. Thai was edited for natural technical instructions without treating English lexical scores as meaningful.
- Parent owns integrated Astro build, internal link/anchor checks, browser inspection, and independent final review. Those are not claimed as complete here.
- No application tests rerun for this documentation-only patch. No real clinical data or live provider calls used. Clinical values describe implementation and do not constitute independent clinical-policy approval.

Minimize Reader Load shaped the removal of repeated summary cards and the single Hb precedence table. Separate Before Serializing Shared State shaped the isolated worktree and owned-file patch. Both leaf skills were read. Technical-writing, unslop, and humanize shaped the prose and source checks.

## Changed files

Both language directories contain architecture.mdx, audit-store.mdx, contributing.mdx, deid.mdx, pipeline-stages.mdx, platelet-auditor.mdx, quote-grounding.mdx, testing.mdx, and three-tier-hb.mdx. Components are docs/src/components/PipelineDiagram.astro and docs/src/components/QuoteGroundingLayers.astro.
