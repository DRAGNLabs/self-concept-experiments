"""Colours, markers and arm labels shared by the behaviour-measure charts."""

from selfconcept.deceiver_axis.behaviour_measures.episodes import PRESETS, ArmName

INK, MUTED, GRID = "#2b2b2b", "#6b6a63", "#d9d8d2"
ARM_LABEL: dict[ArmName, str] = {"unsteered": "unsteered", "gi_v2": "gi_v2 −", "gfar": "gfar −"}
COLOR_BY_ARM: dict[ArmName, str] = {"unsteered": "#8a8984", "gi_v2": "#2a78d6", "gfar": "#eb6834"}
COLOR_BY_MODEL = dict(zip([preset["label"] for preset in PRESETS], ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
MARKER_BY_MODEL = dict(zip([preset["label"] for preset in PRESETS], ["o", "s", "^", "D"]))
