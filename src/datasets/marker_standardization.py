"""
Shared marker standardization options for Eurobench trajectory conversion.
"""

MARKER_ALIAS_MAP: dict[str, str] = {
    "lpsis": "LPSI",
    "rpsis": "RPSI",
    "l_toe": "LTOE",
    "r_toe": "RTOE",
    "ltoe": "LTOE",
    "rtoe": "RTOE",
}

DROP_MARKER_PATTERNS: list[str] = [
    r"^\*\d+$",
]

FORBIDDEN_MARKER_PATTERNS: list[str] = [
    r"^\*\d+$",
]

REQUIRED_MARKERS_MINIMAL: list[str] = [
    "LASI",
    "RASI",
]

TRAJECTORY_MARKER_STANDARDIZATION: dict = {
    "marker_alias_map": MARKER_ALIAS_MAP,
    "drop_marker_patterns": DROP_MARKER_PATTERNS,
    "required_markers": REQUIRED_MARKERS_MINIMAL,
    "forbidden_marker_patterns": FORBIDDEN_MARKER_PATTERNS,
    "strict_marker_validation": False,
}
