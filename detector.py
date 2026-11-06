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
from sklearn.ensemble import IsolationForest

DATA_DIR = Path(__file__).parent / "data"
CSV_PATH = DATA_DIR / "machine_temperature_system_failure.csv"
LABELS_PATH = DATA_DIR / "combined_windows.json"
LABELS_KEY = "realKnownCause/machine_temperature_system_failure.csv"

READINGS_PER_HOUR = 12  # one reading every 5 minutes

DEFAULT_WINDOW = 24 * READINGS_PER_HOUR  # 24 hours of past readings
DEFAULT_THRESHOLD = 4.0
DEFAULT_CONTAMINATION = 0.01  # flag the strangest 1% of readings
WARNING_LOOKBACK_HOURS = 48  # how far before the failure a flag counts as a warning


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


def zscore_detector(df, window=DEFAULT_WINDOW, threshold=DEFAULT_THRESHOLD):
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


def make_features(df, window=DEFAULT_WINDOW):
    """Build the table of numbers Isolation Forest looks at, one row per reading.

    value: the reading itself.
    rolling_mean / rolling_std: the average and spread of the past `window` readings.
    diff: the change since the previous reading, which shows sudden jumps.

    These only look backward, so they could be calculated on a live stream.
    The first `window` rows have no baseline yet, so they contain NaN.
    """
    past_values = df["value"].shift(1).rolling(window)
    return pd.DataFrame(
        {
            "value": df["value"],
            "rolling_mean": past_values.mean(),
            "rolling_std": past_values.std(),
            "diff": df["value"].diff(),
        }
    )


def isolation_forest_detector(df, contamination=DEFAULT_CONTAMINATION, window=DEFAULT_WINDOW):
    """Flag the readings that are easiest to separate from the rest.

    contamination: the share of readings to flag, so 0.01 always flags about 1%.
    window: how many past readings the rolling features use.

    Isolation Forest is unsupervised: it never sees the labels; labels are only
    used to evaluate. It builds random trees that keep splitting the data, and
    unusual readings end up on their own after only a few splits.

    One limit: we train on the whole file, so the model has seen the future. The
    features could run on a live stream, but this training setup could not. A real
    system would train on past data only.
    """
    # IsolationForest can't handle NaN, so leave out the rows with no baseline yet.
    features = make_features(df, window).dropna()

    # random_state makes the random trees the same on every run.
    model = IsolationForest(contamination=contamination, random_state=42)
    predictions = model.fit_predict(features)  # -1 means anomaly, 1 means normal

    # Rows we left out stay False, the same as in the z-score detector.
    flags = pd.Series(False, index=df.index)
    flags.loc[features.index] = predictions == -1
    return flags


def evaluate(df, flags):
    """Score flags against the labeled windows. Returns a dictionary of results.

    The labels are only used here, to grade the detector.

    NAB's docs list three anomalies (planned shutdown, subtle change, catastrophic
    failure) but the labels have four windows and don't say which is which. We treat
    the last window as the failure, because it has the longest stretch below 60
    degrees. That is our own reading of the data, not something NAB states. The
    failure moment is the window's midpoint, which is NAB's exact labeled time.

    Earliest warning = the first flag in the 48 hours before the failure moment,
    whether or not it is inside a labeled window. The temperature starts dropping
    before the failure window opens, so looking only inside the window would miss
    that warning. Flags from further back are too far away to link to the failure.
    A flag in this period can still be noise, and if it is outside a window it also
    counts as a false alarm, because the labels are NAB's and we don't change them.
    """
    windows = load_windows()
    failure_start, failure_end = windows[-1]
    failure_time = failure_start + (failure_end - failure_start) / 2

    window_results = []
    for number, (start, end) in enumerate(windows, start=1):
        flagged_times = df.loc[flags & df["timestamp"].between(start, end), "timestamp"]
        caught = len(flagged_times) > 0
        window_results.append(
            {
                "window": number,
                "start": start,
                "end": end,
                "caught": caught,
                "flag_count": len(flagged_times),
                "first_flag": flagged_times.min() if caught else None,
            }
        )

    lookback_start = failure_time - pd.Timedelta(hours=WARNING_LOOKBACK_HOURS)
    warning_flags = df.loc[flags & df["timestamp"].between(lookback_start, failure_time), "timestamp"]
    earliest_warning_hours = None
    if len(warning_flags) > 0:
        earliest_warning_hours = (failure_time - warning_flags.min()).total_seconds() / 3600

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
        "earliest_warning_hours": earliest_warning_hours,
    }


def print_report(name, df, flags):
    """Print a detector's results in plain text."""
    results = evaluate(df, flags)
    print(f"{name}")
    print(f"Rows: {len(df)}, flagged: {int(flags.sum())}")
    print(f"Failure moment: {results['failure_time']}")
    for w in results["windows"]:
        print(
            f"  Window {w['window']} ({w['start']} to {w['end']}): caught={w['caught']}, "
            f"flags={w['flag_count']}, first flag={w['first_flag']}"
        )
    print(f"Windows caught: {results['windows_caught']} of {results['total_windows']}")
    print(
        f"False alarms: {results['false_alarm_points']} readings "
        f"in {results['false_alarm_events']} separate events"
    )
    print(f"Precision: {results['precision']:.3f}  Recall: {results['recall']:.3f}")
    earliest = results["earliest_warning_hours"]
    if earliest is None:
        print(f"Earliest warning: no flag in the {WARNING_LOOKBACK_HOURS} h before the failure")
    else:
        print(f"Earliest warning: {earliest:.1f} h before the failure")


if __name__ == "__main__":
    data = load_data()
    print_report(
        f"Z-score detector: window={DEFAULT_WINDOW} readings (24 h), threshold={DEFAULT_THRESHOLD}",
        data,
        zscore_detector(data),
    )
    print()
    print_report(
        f"Isolation Forest: window={DEFAULT_WINDOW} readings (24 h), contamination={DEFAULT_CONTAMINATION}",
        data,
        isolation_forest_detector(data),
    )
