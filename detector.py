"""Anomaly detection functions for the NAB machine temperature data.

No Streamlit code lives here, so these functions can be run and tested alone.

Why we don't report accuracy: anomalies are rare, so accuracy is meaningless
(same lesson as my SMS spam classifier). About 90% of these rows are outside
the labeled windows, so a detector that never flags anything would score about
90% and still miss every failure. We use windows caught, false alarms,
precision and recall instead.
"""

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
CSV_PATH = DATA_DIR / "machine_temperature_system_failure.csv"
LABELS_PATH = DATA_DIR / "combined_windows.json"
LABELS_KEY = "realKnownCause/machine_temperature_system_failure.csv"

READINGS_PER_HOUR = 12  # one reading every 5 minutes


def load_windows():
    """Return the labeled anomaly windows as a list of (start, end) timestamps."""
    with open(LABELS_PATH) as f:
        all_windows = json.load(f)
    return [(pd.Timestamp(start), pd.Timestamp(end)) for start, end in all_windows[LABELS_KEY]]


def load_data():
    """Return the sensor data with parsed timestamps and an `in_window` column."""
    df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])

    # On 2014-01-07 the clock jumps back 55 minutes, so 12 timestamps appear
    # twice. Rolling stats work row by row and need time order. A stable sort
    # fixes the order and keeps both readings, since we can't tell which is right.
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)

    df["in_window"] = False
    for start, end in load_windows():
        df.loc[df["timestamp"].between(start, end), "in_window"] = True
    return df


def zscore_detector(df, window, threshold):
    """Flag readings far from the recent average. Returns a True/False Series.

    window: how many past readings form the baseline (288 = 24 hours).
    threshold: how many standard deviations away counts as an anomaly.

    Threshold trade-off: lower threshold = earlier warnings but more false alarms.
    """
    # Rolling stats only look backward, so the detector could run on a live stream.
    # shift(1) leaves the current reading out of its own baseline. Otherwise a
    # sudden spike would raise the average it is compared to and hide itself.
    past_values = df["value"].shift(1).rolling(window)
    mean = past_values.mean()
    std = past_values.std()

    # A perfectly flat baseline has std 0, and we can't divide by that.
    z = (df["value"] - mean) / std.where(std > 0)

    # The first `window` rows have no baseline yet, so z is NaN there. NaN > threshold
    # is False, so those rows are never flagged.
    return z.abs() > threshold


def evaluate(df, flags):
    """Score flags against the labeled windows. Returns a dictionary of results.

    The labels are only used here, to grade the detector.

    NAB's docs list three anomalies (planned shutdown, subtle change, catastrophic
    failure) but the labels have four windows and don't say which is which. So this
    mapping is our own reading of the data, not something NAB states:
      - last window = the failure. It has the longest stretch below 60 degrees.
        The failure moment is its midpoint, which is NAB's exact labeled time.
      - second-to-last window = the subtle change that led to the failure.
      - earlier windows = about a month before the failure, so we don't count
        flags in them as warnings of it.
    """
    windows = load_windows()
    failure_start, failure_end = windows[-1]
    failure_time = failure_start + (failure_end - failure_start) / 2

    window_results = []
    warning_hours = []
    for number, (start, end) in enumerate(windows, start=1):
        flagged_times = df.loc[flags & df["timestamp"].between(start, end), "timestamp"]
        caught = len(flagged_times) > 0
        first_flag = flagged_times.min() if caught else None

        hours_before_failure = None
        is_related_to_failure = number >= len(windows) - 1
        if caught and is_related_to_failure:
            hours_before_failure = (failure_time - first_flag).total_seconds() / 3600
            warning_hours.append(hours_before_failure)

        window_results.append(
            {
                "window": number,
                "start": start,
                "end": end,
                "caught": caught,
                "flag_count": len(flagged_times),
                "first_flag": first_flag,
                "hours_before_failure": hours_before_failure,
            }
        )

    # Point-level counts: every 5-minute reading is graded on its own.
    true_positives = int((flags & df["in_window"]).sum())
    false_positives = flags & ~df["in_window"]
    false_alarm_points = int(false_positives.sum())
    total_flags = true_positives + false_alarm_points
    total_in_windows = int(df["in_window"].sum())

    # One bad afternoon can flag dozens of readings in a row, so also count each
    # unbroken run of false flags as one event. That is closer to how many times
    # an operator would actually get paged.
    run_starts = false_positives & ~false_positives.shift(1, fill_value=False)

    return {
        "windows": window_results,
        "windows_caught": sum(w["caught"] for w in window_results),
        "total_windows": len(windows),
        "false_alarm_points": false_alarm_points,
        "false_alarm_events": int(run_starts.sum()),
        "precision": true_positives / total_flags if total_flags else 0.0,
        "recall": true_positives / total_in_windows if total_in_windows else 0.0,
        "failure_time": failure_time,
        "earliest_warning_hours": max(warning_hours) if warning_hours else None,
    }


if __name__ == "__main__":
    WINDOW = 24 * READINGS_PER_HOUR  # 24 hours of past readings
    THRESHOLD = 3.0

    data = load_data()
    flags = zscore_detector(data, WINDOW, THRESHOLD)
    results = evaluate(data, flags)

    print(f"Z-score detector: window={WINDOW} readings (24 h), threshold={THRESHOLD}")
    print(f"Rows: {len(data)}, flagged: {int(flags.sum())}")
    print(f"Failure moment: {results['failure_time']}")
    print()
    for w in results["windows"]:
        hours = w["hours_before_failure"]
        warning = f"{hours:.1f} h before failure" if hours is not None else "n/a"
        print(
            f"Window {w['window']} ({w['start']} to {w['end']}): caught={w['caught']}, "
            f"flags={w['flag_count']}, first flag={w['first_flag']}, warning={warning}"
        )
    print()
    print(f"Windows caught: {results['windows_caught']} of {results['total_windows']}")
    print(
        f"False alarms: {results['false_alarm_points']} readings "
        f"in {results['false_alarm_events']} separate events"
    )
    print(f"Precision: {results['precision']:.3f}  Recall: {results['recall']:.3f}")
    earliest = results["earliest_warning_hours"]
    print(f"Earliest warning before failure: {earliest:.1f} h" if earliest is not None else "No warning")
