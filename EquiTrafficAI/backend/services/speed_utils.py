"""Shared traffic speed conversion helpers."""

import math

DEFAULT_SPEED_MEAN_MPH=54.40
DEFAULT_SPEED_STD_MPH=19.40
MIN_SPEED_MPH=10.0
MAX_SPEED_MPH=75.0


def standardized_speed_to_mph(
    value: float,
    fallback: float,
    mean_mph: float=DEFAULT_SPEED_MEAN_MPH,
    std_mph: float=DEFAULT_SPEED_STD_MPH,
) -> float:
    """Convert a standardized speed value to a bounded physical speed."""
    try:
        fallback_speed=float(fallback)
    except (TypeError, ValueError):
        fallback_speed=MIN_SPEED_MPH

    try:
        speed=float(value)
    except (TypeError, ValueError):
        return max(MIN_SPEED_MPH, fallback_speed)

    if not math.isfinite(speed):
        return max(MIN_SPEED_MPH, fallback_speed)
    if -5.0 < speed < 5.0:
        speed=mean_mph + speed * std_mph
    return max(MIN_SPEED_MPH, min(MAX_SPEED_MPH, speed))
