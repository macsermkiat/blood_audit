"""Shared regex fragments for named blood components."""

BLOOD_COMPONENT = r"(?:LPRC|PRBC|PRC|FFP|platelets?|SDP|cryo(?:precipitate)?)"

RBC_COMPONENT = r"(?:LPRC|PRBC|PRC)"
"""Red-cell products only. The administration scan's affirmative cues key on
this narrower set: for an RBC order audit, a charted FFP/platelet/cryo
administration does not confirm that the reserved red cells were given."""
