"""
Departure-time scheduling helpers shared across demand generation paths.
"""
from __future__ import annotations

import numpy as np


def sequential_departure_times(
    start_time: float,
    end_time: float,
    count: int,
) -> np.ndarray:
    """
    Create equally spaced departures over (start_time, end_time], including end_time.

    Example:
        start=0, end=120, count=4 -> [30, 60, 90, 120]

    If the bin duration is zero or negative, departures fall back to end_time.
    """
    if count <= 0:
        return np.array([], dtype=float)

    start = float(start_time)
    end = float(end_time)

    if end <= start:
        return np.full(count, end, dtype=float)

    step = (end - start) / float(count)
    departures = start + step * np.arange(1, count + 1, dtype=float)
    departures[-1] = end  # avoid floating drift on the last value
    return departures


def format_departure_time(value: float, decimals: int = 6) -> str:
    """Format a departure time for XML while keeping useful precision."""
    text = f"{float(value):.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text
