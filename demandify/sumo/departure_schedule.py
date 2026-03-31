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
    Create deterministic, evenly spaced departures strictly inside a bin.

    Example:
        start=0, end=120, count=4 -> [24, 48, 72, 96]

    If the bin duration is zero or negative, departures fall back to end_time.
    """
    if count <= 0:
        return np.array([], dtype=float)

    start = float(start_time)
    end = float(end_time)

    if end <= start:
        return np.full(count, end, dtype=float)

    section = (end - start) / float(count + 1)
    return start + section * np.arange(1, count + 1, dtype=float)


def format_departure_time(value: float, decimals: int = 6) -> str:
    """Format a departure time for XML while keeping useful precision."""
    text = f"{float(value):.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text
