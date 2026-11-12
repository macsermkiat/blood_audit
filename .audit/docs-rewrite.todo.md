# Documentation corrections

- [x] Use the completed documentation audit as the scope and record the dirty baseline.
- [x] Correct English and Thai operator setup, command references, and privacy instructions.
- [x] Correct developer guidance against current application and test contracts.
- [x] Align clinical explanations and diagrams with implemented rules without changing policy.
- [x] Simplify site hierarchy, shared summaries, README, and product/design notes.
- [x] Separate historical context and plans from current reference material.
- [x] Correct the existing deck in place while preserving unrelated user content.
- [x] Review all edits independently for factual accuracy, security, and readable prose.
- [x] Build the site, check internal links and anchors, scan prose, and inspect desktop and narrow layouts.
- [x] Summarize completed changes and any limits with a reviewable local diff.

The user approved the rewrite with 'go ahead' after the review and the application merge. Scope includes the ranked groups in .audit/docs-review.md. Code samples may change when needed to correct unsafe or obsolete instructions. Application code and clinical policy stay unchanged. Existing untracked agent documents and the decoder modification are excluded. The existing deck is included only for the identified clinical consistency corrections and a concise current-status note.

Independent groups run in separate worktrees. Operator and developer rewrites can proceed concurrently with clinical/reference corrections. Root integrates those patches sequentially and handles entry points, layout, long-form repository docs, and final verification. No live provider call, patient export, application deployment, commit, or push is needed for the documentation rewrite.
