# Declared USETYPE surgical-intent plan

**Status:** LIVE (default-ON since 2026-07-15) — `DECLARED_USETYPE_ENABLED`
defaults to `True` and both pilot legs default to it; set
`BBA_PILOT_DECLARED_USETYPE=0` to force a run off. Go-live evidence: a
representative 3,000-order preflight (12 flips, 0 unexpected, delta-Hb bucket
empty), a flag-on LLM-leg comparison (1 intended verdict flip, 0 collateral),
and clinician sign-off on the `hb_ge_10 → defer` bucket.
**Scope:** Interpret the clinician's order-time declared use as a surgical-intent
signal without changing routing until the later wiring tickets land.

---

## 1. Signal facts

`BDVSTDT.USETYPE` records the clinician's declared use at order time: `1` is
ward, `2` is surgery, `3` is type and screen, and `4` is Day Care. Code `5` and
all other unknown values are inert. Validation against the 2025 export covered
39,749 REQNOs: no order mixed USETYPE values, while USETYPE `2` was 91.2%
operation-adjacent and only 0.2% had no operation row.

## 2. Locked constraints

Declared intent never auto-clears an order, and flag-off output must be
byte-identical. Detail rows collapse per order using `(HN, REQNO)`, never bare
`REQNO`. Unknown codes drive no routing and never render to the LLM. The
`DECLARED_USETYPE_ENABLED` flag is default-ON since go-live (2026-07-15);
forcing the feature off with `BBA_PILOT_DECLARED_USETYPE=0` reproduces the
pre-feature output byte-for-byte.

## 3. Mapping and collapse semantics

`bba.declared_use` is the single source of truth for code labels, the surgical
group, and per-order collapse. Exactly one distinct non-blank detail-line code
wins; empty or blank-only values produce no signal, and mixed codes log a
warning and produce no signal without raising.

## 4. Data flow and rollout

The five-PR stack proceeds as: ingest, vocabulary, and flag; classifier and
dispatch; bundle and prompts; pilot threading and preflight; then the go-live
default flip after the preflight and clinician sign-off (shipped 2026-07-15).
The deterministic and LLM pilot legs default to the library flag
`DECLARED_USETYPE_ENABLED` (now ON); `BBA_PILOT_DECLARED_USETYPE=1` forces on
and anything else (e.g. `0`) forces off.

## 5. Go-live evidence gate

Run `scripts/pilot/preflight_declared_usetype.py` against the pilot work bundle.
It prints the collapsed distribution, mixed-key and cross-HN collision checks,
the real-input deterministic flip matrix, and the incremental no-operation-row
population, then writes
`$BBA_PILOT_WORK_DIR/preflight_declared_usetype.json` by default. Any mixed
`(HN, REQNO)` or unexpected flip is a HOLD; an empty audited or
declared-surgical population is also a HOLD. The artifact is read-only and does
not enable either feature flag.

This artifact was one of four go-live gates, all met on 2026-07-15: a clean
representative flip matrix (3,000-order sample: 12 flips, 0 unexpected), a
flag-on LLM-leg comparison (1 intended verdict flip, 0 collateral), an
attribution/dashboard delta summary, and clinician sign-off on the high-Hb
(`hb_ge_10` to defer) bucket (the delta-Hb `bypass_delta_hb` bucket produced no
cases). `DECLARED_USETYPE_ENABLED` is default-ON as a result; re-run the
preflight when validating a new cohort or if a `bypass_delta_hb` flip ever
appears.

## 6. Risks

USETYPE is now a hard ingest requirement, so archived exports without it fail
`validate_header` loudly. The resulting schema-fingerprint and `run_id` bump is
the single non-flag-gated day-one consequence; all routing behavior remains
unchanged.
