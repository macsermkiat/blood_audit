"""System-prompt template + task-mode switch.

Two task modes, one ``cohort_threshold`` injection slot per template:

* ``HB_7_10_REVIEW`` — gray-zone case. The LLM is asked to read the
  ±24-h note window for Tier-1 indications (active bleeding,
  hemodynamic instability, ACS, peri-operative, symptomatic anemia,
  neuro-target) and Tier-2 supportive context.
* ``HB_GT_10_OVERRIDE`` — Hb > 10 case. The LLM is asked to look for
  Tier-1 override conditions (MTP, active bleed, hemodynamic
  instability, ACS, peri-operative, symptomatic anemia, neuro-target)
  that would justify ordering despite the deterministic engine's
  ``POTENTIALLY_INAPPROPRIATE`` pre-classification.

The ``cohort_threshold`` lands as a hard numeric input — never inferred
by the LLM (PRD §"Cohort detection is deterministic, not LLM-judged").

RED-phase scaffold: :func:`system_prompt_for` raises
:class:`NotImplementedError`. The template strings are absent on purpose
— the GREEN phase pins the exact wording, which is committee-approved
content and must not be invented during scaffolding.
"""

from __future__ import annotations

from bba.prompt_builder.models import TaskMode


def system_prompt_for(*, task_mode: TaskMode, cohort_threshold: float) -> str:
    """Return the task-mode-specific system prompt with ``cohort_threshold`` injected.

    Output contract:

    * Plain text (no XML envelope at this layer — the user payload uses
      ``<evidence>`` envelopes, but the system prompt is free text the
      LLM treats as its operator's guidance).
    * Contains the exact ``cohort_threshold`` value as a numeric string
      with a single decimal place (``"7.0"`` / ``"7.5"`` / ``"8.0"``).
    * Contains a mode-discriminating phrase so a downstream test can
      distinguish ``HB_7_10_REVIEW`` output from ``HB_GT_10_OVERRIDE``
      output without parsing.
    * NFC-normalized (the canonical hash depends on byte stability).

    Raises :class:`bba.prompt_builder.UnknownTaskModeError` if ``task_mode``
    is outside :data:`TASK_MODES`. Raises
    :class:`bba.prompt_builder.UnsupportedCohortThresholdError` if
    ``cohort_threshold`` is outside :data:`ALLOWED_COHORT_THRESHOLDS`.
    """
    raise NotImplementedError("RED-phase scaffold; see issue #21")
