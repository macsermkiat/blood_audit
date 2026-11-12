# Independent documentation review

Reviewed the integrated working tree on `codex/documentation-cleanup` against `b343d9a`. The reviewer did not author the documentation. This review owns only this report. Existing decoder edits, `CLAUDE.md`, and `docs/agents/` are outside scope.

## Findings

### High priority

1. Slide 10 still contradicts the corrected universal Hb rule.

   Files are `clinical_audit_pitch_deck.html:1443`, `:1453`, `:1457`, and `:1461`. The slide promises situations where the application refuses automatic classification. Its descriptions imply that hemodilution, isolated Hb below 8, and active haematologic malignancy always require review. The classifier returns `APPROPRIATE` at Hb <= 7 before those branches. The isolated-low flag also includes exactly 8.0. The corrected preceding slides do not make this independently readable slide accurate.

   Fix the slide's scope to remaining Hb > 7 after earlier exits. Describe higher-cohort flag branches at or below the cohort floor, use the inclusive isolated-low boundary, and qualify the remaining heme and high-Hb routes. Evidence is `src/bba/deterministic_classifier/classifier.py:340`, `:458`, `:466`, and `src/bba/hb_lookup/lookup.py:195`.

### Warnings

2. The report schema incorrectly called physician and ward values hashed.

   The initial reviewed rewrite retained this wording at `docs/report-schema.md:122` and `:142`. The supported CLI returns the `DCTREQ` physician code and the doctor's `Deptlct` text, or a sentinel. A reader could wrongly assume these artifacts pseudonymize physician identities. Root corrected both fields during review. Rechecked the descriptions against `src/bba/attribution/resolvers.py:47` and `:64`. Closed.

3. The Thai required-Postgres sentence is ambiguous.

   `docs/src/content/docs/th/developers/testing.mdx:20` says `การไม่มี dependency หรือการเชื่อม Docker ไม่ได้ทำให้ test ล้มเหลว`. It can be parsed as saying missing dependencies or Docker do not fail the tests. State the condition explicitly, such as `หากขาด dependency หรือเชื่อมต่อ Docker ไม่ได้ test จะล้มเหลว`. The English guide already states the intended required-mode failure.

### Suggestions

4. Remove a deleted component's remaining design reference.

   `DESIGN.md:115` still names `CliCommandCard` in the motion examples even though this rewrite removes that unused component. Remove its name from the example.

## Scope and evidence

Read the changed shared Astro components and bilingual operator, developer, clinical, and reference content. Checked the deck correction patch against the actual deck. Cross-checked the important runtime claims against application composition, classifiers, cohort detection, privacy admission, attribution, and report projection. This is code and documentation consistency review, not medical policy advice or clinical sign-off.

The English and Thai setup guides retain the public Ed25519 verification key and private issuer separation, exact evidence and run binding, local privacy rejection, provider-charge condition, trusted identity adapter, HTTPS proxy, signed cookie, CSRF and revision checks, immutable completed results, source snapshots, and uncertain-submission reconciliation. The platelet route remains explicitly default off, with incomplete audits rejected. Reports remain explicitly RBC-only with per-row provenance and separate attribution inputs. The inclusive Hb boundaries, ESRD 7.0, orthopedic 8.0, strict platelet 100k distinction, and legacy 72-hour review deferral agree with the implementation in the reviewed pages.

Executed the English and Thai environment-presence snippets in child processes with synthetic API key, password-bearing DSN, and authentication-secret values. Both printed only variable names and `set` or `unset`. None of the synthetic values appeared. The reviewer's first assertion mistakenly rejected `SECRET` within the variable name itself. The corrected check compares complete sentinel values and validates every output line's grammar. This was a reviewer assertion issue, not a documentation defect.

Direct psycopg parsing accepts the documented `postgresql://` example and rejects the SQLAlchemy `postgresql+psycopg://` form. No database connection was attempted.

Compared the historical README and CONTEXT snapshots to their base-revision originals after ignoring rebased Markdown targets. Their prose is unchanged apart from the explicit historical notice. Historical and changelog descriptions are not treated as current setup instructions. SQL examples remain exporter references rather than commands executed by the application.

The deleted Astro components have no remaining imports. Retained component changes introduce no external input execution or runtime network behavior. Frontmatter banner HTML remains the pre-existing trusted repository-content boundary. No dependency manifests or lockfiles changed, so this rewrite integrates no new library license or dependency-security exposure. `git diff --check` passes.

Root owns site build, Astro checking, browser verification, and built-site link checks. Those checks are not claimed as independently executed here. This review makes no claim about live providers, real clinical data, deployed authentication, external link availability, or clinical validity. Application tests were not rerun for this static documentation review.

## Verdict

Blocked on finding 1 until the contradictory slide is corrected. Findings 3 and 4 should be corrected in this documentation pass. Finding 2 is closed.

### Follow-up verification

Rechecked root's subsequent corrections. The Thai test guide now explicitly says missing dependencies or an unreachable Docker daemon fail required-mode tests. DESIGN no longer names the removed command-card component. Findings 3 and 4 are closed. Finding 1 remains open at this checkpoint.

### Deck and layout follow-up

Rechecked the actual slide 10 and refreshed correction patch. The slide now limits review checks to remaining Hb above 7 after earlier exits, states the inclusive isolated-low boundary, and qualifies unknown-context and active-heme review. Finding 1 is closed.

Slide 11 now describes the actual six citation predicates and separates signed privacy admission from grounding. Its footer explicitly avoids claiming clinical truth or redaction quality. These descriptions agree with `quote_grounder/verifier.py` and `layers.py`.

One warning remains in slide 11's callout at `clinical_audit_pitch_deck.html:1486`. A single grounding failure does not necessarily determine the stored classification because `audit_pipeline/replay.py` selects the latest passing attempt. Replace the blanket failure wording with the condition that no attempt passes. This is the same condition already used by the current site guide.

Also reviewed the added `docs/src/styles/custom.css:97` hero rule. The selector scopes vertical padding to hero pages and changes no interactive behavior, data flow, or theme tokens. It introduces no apparent security or maintainability issue. Root owns final rendered viewport verification and the rebuilt-site checks.

The high-priority blocker is resolved. The remaining callout warning requires the small factual correction above before final documentation approval.

### Final verdict

Rechecked the corrected callout at `clinical_audit_pitch_deck.html:1484`. It now conditions the review result on no attempt passing grounding and explicitly allows a later passing attempt after an earlier rejection. This agrees with the winning-attempt rule. The line references in the preceding checkpoint refer to that earlier file state.

Approve the reviewed documentation and scoped CSS changes. All findings from this independent review are closed. No critical or high-priority content or component issue remains identified. Final site build, link checks, and viewport evidence remain root-owned verification, with their results and any limitations reported separately. This approval does not certify clinical policy or hospital deployment readiness.

### Final render correction

Read the slide-8-only rule at `clinical_audit_pitch_deck.html:586`. It adds `margin-bottom: 1.5rem` to that slide's decision tree. The selector is narrowly scoped, changes no content or executable behavior, and introduces no review finding. The DESIGN motion paragraph preserves the reduced-motion requirement, timing, easing, visible focus, and restriction on animated layout properties. Approval remains unchanged. Root owns verification that the final rendered decision cards clear the footer at the measured viewport and remain readable in narrow views. `git diff --check` still passes.
