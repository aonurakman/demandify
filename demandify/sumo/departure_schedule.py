"""
Departure-time scheduling helpers shared across demand generation paths.
"""
from __future__ import annotations

from typing import Optional
import numpy as np


GOLDEN_RATIO_CONJUGATE: float = 0.6180339887498949


def sequential_departure_times(
    start_time: float,
    end_time: float,
    count: int,
    phase_offset: Optional[float] = None,
) -> np.ndarray:
    """
    Create deterministic, evenly spaced departures strictly inside a bin.

    Args:
        start_time: Start of the bin in seconds.
        end_time: End of the bin in seconds.
        count: Number of vehicle departures.
        phase_offset: Optional phase offset in [0, 1) to stagger departure grids
            across different OD pairs, avoiding synchronized departure spikes.
            If None, defaults to the legacy centered midpoint spacing:
            start + ((end - start) / (count + 1)) * [1..count].

    Example:
        start=0, end=120, count=4, phase_offset=None -> [24, 48, 72, 96]
        start=0, end=120, count=1, phase_offset=0.25 -> [30]

    If the bin duration is zero or negative, departures fall back to end_time.
    """
    if count <= 0:
        return np.array([], dtype=float)

    start = float(start_time)
    end = float(end_time)

    if end <= start:
        return np.full(count, end, dtype=float)

    if phase_offset is None:
        section = (end - start) / float(count + 1)
        return start + section * np.arange(1, count + 1, dtype=float)

    phi = float(phase_offset) % 1.0
    step = (end - start) / float(count)
    return start + (np.arange(count, dtype=float) + phi) * step


def format_departure_time(value: float, decimals: int = 6) -> str:
    """Format a departure time for XML while keeping useful precision."""
    text = f"{float(value):.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text

