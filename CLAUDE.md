# Sensor Anomaly Detector

## What this is
A small Python project that finds anomalies in real industrial sensor data and shows them in a web app.
Built by Kerem (junior, Computer Systems Engineering, ASU) as a learning project. Kerem can read code
but is new to ML deployment, so: keep code simple, readable, and commented. Explain each step briefly
in plain English after you finish it. No clever abstractions.

## Hard constraints
- Must be finished and deployed in about 3 hours. Prefer the simplest thing that works.
- Python 3.13. Libraries: pandas, numpy, scikit-learn, plotly, streamlit. Nothing else unless asked.
- Deploy target: Streamlit Community Cloud (free, from a public GitHub repo). So the app entry point
  is `app.py` at the repo root and all deps are in `requirements.txt`.
- No API keys, no databases, no Docker.

## Data
NAB (Numenta Anomaly Benchmark), realKnownCause/machine_temperature_system_failure.csv
- Temperature sensor on an internal component of a large industrial machine, one reading every 5 minutes,
  ~22,700 rows, columns: timestamp, value.
- Known anomalies: a planned shutdown, a subtle early change, and a catastrophic failure.
- Data URL: https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/machine_temperature_system_failure.csv
- Label windows: https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_windows.json
  (use the "realKnownCause/machine_temperature_system_failure.csv" key)
- Save both into `data/`. NAB is AGPL-3.0; credit it in the README.

## Approach
1. `detector.py`: pure functions, no Streamlit code.
   - `load_data()` -> DataFrame with parsed timestamps + a boolean `in_window` column from the labels.
   - `zscore_detector(df, window, threshold)`: rolling mean/std baseline; flag |z| > threshold.
   - `make_features(df, window)`: value, rolling mean, rolling std, first difference.
   - `isolation_forest_detector(df, contamination)`: scikit-learn IsolationForest on those features.
   - `evaluate(df, flags)`: for each labeled window, was anything flagged inside it, and how many hours
     before the failure was the first flag; count false alarms outside windows; precision/recall at the
     point level.
2. `app.py` (Streamlit):
   - Plotly line chart of temperature, shaded labeled windows, red markers on flagged points.
   - Sidebar: pick detector, sliders for window / threshold / contamination.
   - Metrics row: windows caught, false alarms, earliest warning before failure.
   - Short "How it works" expander in plain English.
   - Optional: upload your own CSV with timestamp,value columns.
3. `README.md`: what it does, live app link placeholder, screenshot placeholder, how to run locally,
   results, limitations, data credit.

## Key ideas to keep visible in code comments
- Anomalies are rare, so accuracy is meaningless (same lesson as my SMS spam classifier). Use
  windows caught, false alarms, precision and recall.
- Rolling stats only look backward, so the detector could run on a live stream.
- Isolation Forest is unsupervised: it never sees the labels; labels are only used to evaluate.
- Threshold trade-off: lower threshold = earlier warnings but more false alarms.

## Working style
- Do one step at a time, run it, show me the output, then stop for my OK.
- After each step, commit with a clear message.
