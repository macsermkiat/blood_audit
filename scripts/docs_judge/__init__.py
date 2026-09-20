"""docs_judge: judge the project's docs for AI-writing tells and readability.

Deterministic work (splitting MDX into prose blocks, counting mechanical
tells) lives in code. Semantic judgments (puffery, superficial analysis,
readability for a named audience, one verdict per block) go to TypeSafe's
Jev model as typed questions. Nothing here rewrites text; the report is the
input to a separate humanize pass.
"""
