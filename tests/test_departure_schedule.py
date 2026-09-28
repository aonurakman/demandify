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


def test_sequential_departures_with_phase_offset():
    # Single vehicle with phase offset 0.1 in [0, 1200] departs at 120s
    deps = sequential_departure_times(0, 1200, 1, phase_offset=0.1)
    assert deps.tolist() == pytest.approx([120.0])

    # 4 vehicles with phase offset 0.5 in [0, 120]: step is 30s, departs at 15, 45, 75, 105
    deps4 = sequential_departure_times(0, 120, 4, phase_offset=0.5)
    assert deps4.tolist() == pytest.approx([15.0, 45.0, 75.0, 105.0])


def test_staggered_od_departures_eliminate_warmup_void():
    from demandify.sumo.departure_schedule import GOLDEN_RATIO_CONJUGATE

    # Simulate 50 OD pairs with 1 vehicle each in a 1200s window
    all_departures = []
    for od_idx in range(50):
        phi = ((od_idx + 1) * GOLDEN_RATIO_CONJUGATE) % 1.0
        deps = sequential_departure_times(0, 1200, 1, phase_offset=phi)
        all_departures.extend(deps.tolist())

    # Warmup (first 300s) has traffic!
    early_deps = [d for d in all_departures if d < 300.0]
    assert len(early_deps) >= 10, f"Expected at least 10 early departures, got {len(early_deps)}"

    # Departures span all quadrants of the window
    assert min(all_departures) < 60.0
    assert max(all_departures) > 1140.0

