# Application review

- [x] 1. Route through the **how** skill. For motivation questions, also route through the **why** skill.
- [x] 2. Throughput checkpoint stays one line: `throughput checkpoint: n/a, read-only investigation`.
- [x] 3. Produce the `how`-shaped output (Overview / Key Concepts / How It Works / Where Things Live / Gotchas), or a recommendation with a tradeoffs table if the request is a decision between alternatives.
- [x] 4. Apply the **unslop** skill to the reply.

## Swarm

- [x] Frame
- [x] Fan out
- [x] Aggregate
- [x] Report

## Review criteria

- [x] Trace decision flow and data integrity.
- [x] Inspect privacy and application boundaries.
- [x] Inspect reviewer experience and delivery checks.
- [x] Run local checks without live clinical data or paid APIs.
- [x] Reconcile findings and record verification limits.
- [x] Complete the no-comments report-only prerequisite.

Investigation only. No application edits, remote issues, commits, or PRs.

The how workflow used two source explorers and one synthesis agent. The no-comments prerequisite used a separate report-only reviewer with zero edits. The why skill was not applicable because no historical motivation question was asked. The review table and ordered recommendations adapt the investigation output to the user's application-quality question.

The two saved reproduction command blocks were extracted from the final document and executed. Both exited zero and reproduced the recorded defects. The temporary UI server exited zero after inspection. Full limits and excluded tests are recorded in the report.
