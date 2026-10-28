# Network Anomaly Detector

A data pipeline that turns a stream of per-device telemetry (latency,
bandwidth, packet loss, error counts) into flagged incidents with a
proposed root cause and remediation, using each tool for what it is
actually good at: PySpark for distributed, event-stream-scale processing,
scikit-learn for the actual outlier model, and a small deterministic agent
for the investigation step.

## Pipeline

1. **Generate** (`app/data_generation.py`) — seeded, deliberately realistic
   telemetry for many devices over many days, entirely through Spark
   DataFrame transformations (not a Python loop): a diurnal traffic
   pattern per device, plus injected fault windows (latency spikes,
   packet loss bursts, bandwidth drops, and compound device faults) with
   a ground-truth label kept alongside for evaluation.
2. **Feature engineer** (`app/features.py`) — a rolling, per-device
   modified z-score (median and median absolute deviation, not mean and
   standard deviation — see *Design notes*) over a trailing window,
   computed with Spark window functions.
3. **Detect** (`app/anomaly_detection.py`) — a cheap, fully distributed
   z-score pass narrows the whole stream down to a small candidate set,
   then an Isolation Forest, fit on a sample of confirmed-normal rows,
   scores those candidates against what normal actually looks like.
4. **Confirm** (`app/hypothesis_testing.py`) — a two-sample statistical
   test (Mann-Whitney U) checks that a metric's values during flagged
   windows are actually, measurably different from normal, rather than
   trusting the detector's label alone.
5. **Investigate** (`app/root_cause_agent.py`) — for each detected
   anomaly, a small multi-step agent checks whether other devices are
   anomalous in the same window (an isolated fault vs. a shared network
   issue), matches the metric signature against a set of known fault
   patterns, and proposes a remediation action — an auditable, rule-based
   tool-calling pipeline rather than an opaque model.

Run the whole thing with `python3 -m app.pipeline`.

## Three real bugs found and fixed while building this

**A rolling z-score that goes numb to the anomaly it just caught.** The
first version used ordinary mean and standard deviation over the trailing
window. A single huge spike inflates the window's standard deviation for
the next `window_minutes` rows, so the z-score for every later point in
the *same* sustained anomaly collapses toward zero almost immediately —
recall on a 20+ minute injected fault measured at about 10% in testing,
because only its first one or two minutes were ever flagged. Switched to
a robust modified z-score (median and median absolute deviation, Iglewicz
and Hoaglin's method), which is far less sensitive to the one outlier it
is trying to detect.

**An Isolation Forest fit on the wrong population.** The anomaly detector
originally fit the Isolation Forest directly on the z-score candidate
set. But once the z-score stage has already narrowed the data down, that
candidate set is *mostly real anomalies* — so the model looked for
outliers among rows that were themselves outliers, treated the anomalies
as the normal majority, and rejected most of the genuine detections.
Fixed by fitting the model on a separate sample of confirmed-normal rows
instead, then scoring candidates against that baseline.

**A startup artifact that looked like a network-wide outage.** Running the
full pipeline end to end (not just the unit tests) surfaced a third issue
the smaller, isolated tests had not: nearly every device in the dataset
showed up as a "co-occurring anomaly" at minute index 1. A device's first
couple of minutes in the stream have almost no rolling history (1-2 prior
points), so the median and MAD computed from them are nearly meaningless
— that noise was enough to cross the detection threshold for almost the
whole fleet simultaneously, a false "network-wide incident" that was
really just an artifact of insufficient warm-up data. Fixed by requiring
a minimum amount of real history before trusting the statistic at all.

## An honest, documented limitation

Recall plateaus around 0.35-0.4 on this synthetic data no matter how the
z-score threshold is tuned (verified with a threshold sweep — see
`anomaly_detection.Z_SCORE_THRESHOLD`). This is not a bug: once a
sustained anomaly lasts longer than roughly the rolling window size (30
minutes here), the window itself becomes dominated by the anomaly's own
values, so the back half of a long fault statistically looks like the new
normal to any *window-relative* detector. A production system would pair
this with an absolute-threshold check or a longer reference window
alongside the short one, rather than relying on one window size alone.

## Running it locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python3 -m app.pipeline
```

## Testing

```bash
pytest -v
```

28 tests, including unit tests for each pipeline stage, a direct
regression test for the mean/std bug above, and an end-to-end integration
test whose quality bounds come from the measured threshold sweep, not
guesswork.
