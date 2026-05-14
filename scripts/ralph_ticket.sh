#!/usr/bin/env bash
# Generates the three-step Claude Code workflow for a single ticket:
#   1. /tdd            — scaffold tests + interface from the issue body (RED phase)
#   2. /ralph-loop     — iterate red→green→refactor until the promise fires
#   3. /codex:review   — independent code review on the diff before closing the issue
#
# Usage:  ./scripts/ralph_ticket.sh <issue_number> <module_path>
# Example: ./scripts/ralph_ticket.sh 3 ingest
#          ./scripts/ralph_ticket.sh 6 vitals_extractor

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <issue_number> <module_path>"
  echo "Example: $0 3 ingest"
  exit 1
fi

N="$1"
MODULE="$2"
REPO="macsermkiat/blood_audit"

cat <<EOF

==============================================================
  Ticket #${N} (bba.${MODULE}) — three-step workflow
==============================================================

Run these three slash commands in your Claude Code session,
sequentially, gating on success of each.

--------------------------------------------------------------
STEP 1 — /tdd (RED phase: scaffold interface + failing tests)
--------------------------------------------------------------

/tdd Implement issue #${N} (bba.${MODULE}) from ${REPO}.

Steps:
1. Read the ticket:  gh issue view ${N} --repo ${REPO}
2. Read referenced rules:  ~/.claude/rules/testing.md, ~/.claude/rules/coding-style.md
3. Scaffold the module interface in src/bba/${MODULE}/ (or src/bba/${MODULE}.py if single-file)
   - Type-hinted public API only. NO implementation bodies.
   - Pydantic v2 models for inputs/outputs.
4. Generate the FULL test plan in tests/unit/test_${MODULE}.py:
   - One test per acceptance criterion in the issue body
   - Property tests (hypothesis) for deep modules
   - Adversarial fixtures where the issue calls them out (quote_grounder, vitals_extractor, etc.)
5. Run:  uv run pytest tests/unit/test_${MODULE}.py -v
   - Confirm EVERY test FAILS (RED). If any passes, the test is wrong.
6. Commit:  "test(${MODULE}): scaffold failing tests for issue #${N}"

Do NOT write implementation in this step. Stop after RED is confirmed.

--------------------------------------------------------------
STEP 2 — /ralph-loop (GREEN → REFACTOR phase: drive to promise)
--------------------------------------------------------------

/ralph-loop "Implement issue #${N} (bba.${MODULE}) from ${REPO}.

Each iteration:
1. Run:  uv run pytest tests/unit/test_${MODULE}.py -v
2. For the first failing test, write the MINIMUM code in src/bba/${MODULE}/ to make it pass.
3. Re-run pytest. Confirm that one test now passes and others still fail for the right reason.
4. Refactor only after the test is green. Commit each red→green→refactor cycle separately:
   - feat(${MODULE}): minimal impl for <test name>
   - refactor(${MODULE}): <what was cleaned up>
5. Run:  uv run ruff check src/bba && uv run mypy --strict src/bba
6. Run coverage:  uv run pytest --cov=bba.${MODULE} --cov-report=term-missing
7. If coverage < 80% on the module, add tests for the uncovered branches FIRST, then loop back to step 2.

Output <promise>ISSUE-${N}-COMPLETE</promise> ONLY when ALL of:
- every test in tests/unit/test_${MODULE}.py passes
- coverage ≥ 80% on src/bba/${MODULE}/
- ruff check clean
- mypy --strict clean
- at least one hypothesis property test exists (for deep modules)
- no TODO / FIXME / pass-only function bodies in the implementation

Otherwise continue iterating." --completion-promise "ISSUE-${N}-COMPLETE" --max-iterations 40

--------------------------------------------------------------
STEP 3 — /codex:review (independent review on the diff)
--------------------------------------------------------------

After the promise fires, BEFORE closing the issue, run:

/codex:review Review the implementation of issue #${N} (bba.${MODULE}) in ${REPO}.

Context:
- PRD: gh issue view 1 --repo ${REPO}
- Ticket: gh issue view ${N} --repo ${REPO}
- Diff: git log --oneline main..HEAD -- src/bba/${MODULE} tests/unit/test_${MODULE}.py

Independent review focus:
- Does the implementation match the ticket's acceptance criteria, not just the tests?
- Tests that test behavior vs tests that test implementation?
- Property tests genuinely cover the invariant, or do they just exercise the happy path?
- Edge cases in the ticket body that don't have corresponding tests?
- Any silent error swallowing, naive datetime, hardcoded values?
- Coverage gaps in critical branches even if overall % is >= 80%?

Return: PASS / NEEDS-CHANGES with specific line-anchored feedback.

--------------------------------------------------------------
STEP 4 — Manual: close the issue
--------------------------------------------------------------

If /codex:review returns PASS:
  gh issue close ${N} --repo ${REPO} --comment "Closed by ralph-loop workflow. PR: <link if applicable>"

If NEEDS-CHANGES:
  Address the feedback (a smaller /ralph-loop iteration usually suffices), re-run /codex:review.

==============================================================
EOF
