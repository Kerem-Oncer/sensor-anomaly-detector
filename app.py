"""Streamlit page that shows the anomaly detectors on the sensor data.

All the detection logic lives in detector.py. This file only handles the page:
the controls, the chart and the numbers.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from detector import (
    DEFAULT_CONTAMINATION,
    DEFAULT_THRESHOLD,
    DEFAULT_WINDOW,
    WARNING_LOOKBACK_HOURS,
    evaluate,
    isolation_forest_detector,
    load_data,
    load_windows,
    zscore_detector,
)

ISOLATION_FOREST = "Isolation Forest"
Z_SCORE = "Z-score"

# Chart colors. Red is only used for flagged readings, so it always means the same thing.
LINE_COLOR = "#2a78d6"
FLAG_COLOR = "#d03b3b"
WINDOW_COLOR = "#898781"


@st.cache_data
def load_nab_data():
    """Read the NAB file once, then reuse it each time the page reruns."""
    return load_data()


@st.cache_data
def run_detector(df, detector_name, window, threshold, contamination):
    """Run the chosen detector. Streamlit reruns this file whenever a control
    changes, so caching means settings we've already tried come back instantly."""
    if detector_name == ISOLATION_FOREST:
        return isolation_forest_detector(df, contamination, window)
    return zscore_detector(df, window, threshold)


def read_uploaded_csv(uploaded_file):
    """Turn an uploaded CSV into the same shape as load_data(), or return None."""
    try:
        df = pd.read_csv(uploaded_file)
    except ValueError:
        return None
    if "timestamp" not in df.columns or "value" not in df.columns:
        return None

    df = df[["timestamp", "value"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().sort_values("timestamp", kind="stable").reset_index(drop=True)

    # An uploaded file has no labels, so nothing is inside a window.
    df["in_window"] = False
    return df


def make_chart(df, flags, windows, failure_time):
    """Draw the temperature line, shaded windows, flagged readings and failure moment."""
    fig = go.Figure()

    # Scattergl draws with the graphics card, which keeps 22,000 points smooth.
    fig.add_trace(
        go.Scattergl(
            x=df["timestamp"],
            y=df["value"],
            mode="lines",
            name="Temperature",
            line={"color": LINE_COLOR, "width": 1.5},
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=df.loc[flags, "timestamp"],
            y=df.loc[flags, "value"],
            mode="markers",
            name="Flagged reading",
            marker={"color": FLAG_COLOR, "size": 8},
        )
    )

    for number, (start, end) in enumerate(windows):
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=WINDOW_COLOR,
            opacity=0.25,
            line_width=0,
            layer="below",
            name="Labeled window",
            showlegend=number == 0,  # one legend entry is enough for all the windows
        )

    if failure_time is not None:
        fig.add_shape(
            type="line",
            x0=failure_time,
            x1=failure_time,
            y0=0,
            y1=1,
            yref="paper",
            line={"color": WINDOW_COLOR, "width": 1, "dash": "dash"},
        )
        fig.add_annotation(
            x=failure_time,
            y=1,
            yref="paper",
            yanchor="bottom",
            text="Failure",
            showarrow=False,
        )

    fig.update_layout(
        height=480,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.05, "x": 0},
        yaxis_title="Temperature",
    )
    return fig


st.set_page_config(page_title="Sensor Anomaly Detector", layout="wide")
st.title("Sensor Anomaly Detector")
st.write(
    "Finding the warning signs of a machine failure in real industrial temperature "
    "data, with one reading every 5 minutes."
)

# Sidebar: the detector and its settings. Each detector only shows the sliders it uses.
st.sidebar.header("Settings")
detector_name = st.sidebar.selectbox("Detector", [ISOLATION_FOREST, Z_SCORE])
window = st.sidebar.slider(
    "Baseline window (readings)",
    min_value=12,
    max_value=864,
    value=DEFAULT_WINDOW,
    step=12,
    help="How many past readings form the baseline. For 5-minute data, 12 = 1 hour and 288 = 24 hours.",
)

threshold = DEFAULT_THRESHOLD
contamination = DEFAULT_CONTAMINATION
if detector_name == Z_SCORE:
    threshold = st.sidebar.slider(
        "Threshold (standard deviations)",
        min_value=1.0,
        max_value=8.0,
        value=DEFAULT_THRESHOLD,
        step=0.5,
        help="Lower = earlier warnings but more false alarms.",
    )
else:
    contamination = st.sidebar.slider(
        "Contamination (share of readings to flag)",
        min_value=0.005,
        max_value=0.10,
        value=DEFAULT_CONTAMINATION,
        step=0.005,
        format="%.3f",
        help="Higher = earlier warnings but more false alarms.",
    )

uploaded_file = st.sidebar.file_uploader("Or use your own CSV (timestamp, value columns)", type="csv")

# Pick the data. An uploaded file has no labels, so it can't be scored.
if uploaded_file is None:
    data = load_nab_data()
else:
    data = read_uploaded_csv(uploaded_file)
    if data is None or data.empty:
        st.error("Couldn't read that file. It needs a CSV with timestamp and value columns.")
        st.stop()

if len(data) <= window:
    st.error(f"This file has {len(data)} readings, but the baseline window needs more than {window}.")
    st.stop()

flags = run_detector(data, detector_name, window, threshold, contamination)

# Metrics row. Anomalies are rare, so accuracy would be meaningless here: flagging
# nothing would score about 90%. These numbers show what actually matters.
if uploaded_file is None:
    results = evaluate(data, flags)
    earliest = results["earliest_warning_hours"]

    column1, column2, column3 = st.columns(3)
    column1.metric("Windows caught", f"{results['windows_caught']} of {results['total_windows']}")
    column2.metric("False alarms", f"{results['false_alarm_points']} readings")
    column3.metric(
        "Earliest warning before failure",
        f"{earliest:.1f} h" if earliest is not None else "None",
    )
    st.caption(
        f"False alarms came in {results['false_alarm_events']} separate events. "
        f"Precision {results['precision']:.3f}, recall {results['recall']:.3f}. "
        f"Earliest warning is the first flag in the {WARNING_LOOKBACK_HOURS} hours before the failure."
    )
    chart = make_chart(data, flags, load_windows(), results["failure_time"])
else:
    column1, column2 = st.columns(2)
    column1.metric("Flagged readings", int(flags.sum()))
    column2.metric("Share flagged", f"{flags.mean():.1%}")
    st.caption("Your file has no labels, so these flags can't be scored as right or wrong.")
    chart = make_chart(data, flags, [], None)

st.plotly_chart(chart, width="stretch")

with st.expander("See the flagged readings as a table"):
    st.dataframe(data.loc[flags, ["timestamp", "value"]], hide_index=True, width="stretch")

with st.expander("How it works"):
    st.markdown(
        """
**The data.** A temperature sensor on an internal part of a large industrial machine,
from the Numenta Anomaly Benchmark (NAB). The gray bands are the periods NAB labeled as
anomalies. The dashed line is the failure. NAB doesn't say which band is which, so
treating the last band as the failure is our own reading of the data.

**Z-score.** Each reading is compared to the average of the readings before it. If it's
too many standard deviations away, it's flagged. It only looks backward, so it could run
on a live stream. A slow decline can fool it, because the baseline drifts down with the
temperature.

**Isolation Forest.** A model that builds random trees to split the data, and flags the
readings that get separated fastest. It never sees the labels, which are only used to
grade it. Here it trains on the whole file, including the failure, so unlike the z-score
detector it couldn't run live as it is. Its warning time also shifts with the random
seed, so read it as a rough figure.

**How to read the numbers.** Lower thresholds and higher contamination give earlier
warnings but more false alarms. Flags in the hours before a labeled window count as false
alarms, even when they're a real warning, because we keep NAB's labels as they are.
"""
    )
