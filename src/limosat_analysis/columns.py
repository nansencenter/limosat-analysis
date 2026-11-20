"""Names and small helpers shared by the Parquet and NetCDF readers."""

from __future__ import annotations

import pandas as pd

SUPPORTED_MAJOR_VERSION = 2

# Columns every point read returns unless ``columns`` says otherwise: identity, time, position, displacement and speed from
# the previous observation, and the is_primary marker. Pass columns="all" for every column, or a list of names.
CORE_COLUMNS = ("trajectory_id", "point_index", "time", "x_m", "y_m", "dx_m", "dy_m", "speed_m_per_day", "is_primary")
CORE_DISPLACEMENT_COLUMNS = ("trajectory_id", "point_index", "start_time", "end_time", "elapsed_seconds", "x0_m", "y0_m", "x1_m",
                             "y1_m", "dx_m", "dy_m", "speed_m_per_day", "is_primary")
DISPLACEMENT_COLUMNS = (
    "dx_m", "dy_m", "speed_m_per_day", "interpolated", "correlation", "direct_match_share", "quality_flags",
)


def utc(value):
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("Time bounds must not be NaT")
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def bounds(start, stop):
    start, stop = utc(start), utc(stop)
    if start is not None and stop is not None and start >= stop:
        raise ValueError("Expected start < stop (stop is exclusive)")
    return start, stop


def limit(max_rows):
    if max_rows is not None and (isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1):
        raise ValueError("max_rows must be a positive integer")


def resolve_columns(columns, available, core) -> list[str]:
    """``None`` -> the core columns that exist, ``"all"`` -> every column, otherwise the requested names (checked)."""
    available = list(available)
    if columns is None:
        return [c for c in core if c in available]
    if isinstance(columns, str):
        if columns == "all":
            return available
        columns = [columns]
    missing = [c for c in columns if c not in available]
    if missing:
        raise ValueError(f"Unavailable columns: {sorted(missing)}; available: {available}")
    return list(dict.fromkeys(columns))
