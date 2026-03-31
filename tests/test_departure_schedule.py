import pytest

from demandify.sumo.departure_schedule import sequential_departure_times


def test_sequential_departures_match_expected_example():
    departures = sequential_departure_times(0, 120, 4)
    assert departures.tolist() == pytest.approx([24.0, 48.0, 72.0, 96.0])
    assert departures[0] > 0.0
    assert departures[-1] < 120.0


def test_single_departure_is_centered_in_the_bin():
    departures = sequential_departure_times(60, 120, 1)
    assert departures.tolist() == pytest.approx([90.0])


def test_zero_or_negative_bin_duration_falls_back_to_end_time():
    assert sequential_departure_times(10, 10, 3).tolist() == pytest.approx([10.0, 10.0, 10.0])
    assert sequential_departure_times(20, 10, 2).tolist() == pytest.approx([10.0, 10.0])


def test_non_positive_count_returns_empty_schedule():
    assert sequential_departure_times(0, 10, 0).size == 0
    assert sequential_departure_times(0, 10, -5).size == 0
