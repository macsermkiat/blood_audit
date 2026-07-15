# Declared USETYPE surgical-intent plan

**Status:** IN PROGRESS — specification #147, delivered as a stacked PR series;
PR-1 adds ingest, vocabulary, and the default-off feature flag.
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
`REQNO`. Unknown codes drive no routing, never render to the LLM, and the
`DECLARED_USETYPE_ENABLED` flag remains default-off.

## 3. Mapping and collapse semantics

`bba.declared_use` is the single source of truth for code labels, the surgical
group, and per-order collapse. Exactly one distinct non-blank detail-line code
wins; empty or blank-only values produce no signal, and mixed codes log a
warning and produce no signal without raising.

## 4. Data flow and rollout

The five-PR stack proceeds as: ingest, vocabulary, and flag; classifier and
dispatch; bundle and prompts; pilot threading and preflight; then the go-live
default flip after preflight and clinician sign-off. The deterministic and LLM
pilot legs now thread declared use behind `BBA_PILOT_DECLARED_USETYPE`, which is
default-off; ticket #152 retains the go-live gate and default-flip decision.

## 5. Risks

USETYPE is now a hard ingest requirement, so archived exports without it fail
`validate_header` loudly. The resulting schema-fingerprint and `run_id` bump is
the single non-flag-gated day-one consequence; all routing behavior remains
unchanged.
