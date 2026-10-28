"""Rolling, per-device time series features computed with Spark window
functions -- the event-stream-mining step that turns a raw telemetry
stream into per-minute deviation signals a detector can threshold on.

Uses a robust modified z-score (median and median absolute deviation,
Iglewicz and Hoaglin's method) rather than mean and standard deviation.
A plain mean/std rolling z-score is not robust to the very outlier it is
trying to detect: once one extreme value enters the trailing window, it
inflates the std for the next `window_minutes` rows and the z-score for
every subsequent point in the SAME sustained anomaly collapses toward
zero almost immediately, even though the anomaly is still ongoing. This
showed up as a real bug during testing -- recall on a multi-minute
injected anomaly dropped to about 10% because only its first one or two
minutes were ever flagged. Median and MAD are far less sensitive to a
single outlier sitting in the window, so a sustained anomaly keeps being
flagged for its whole duration instead of just its onset."""
from __future__ import annotations

import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window

METRICS = ["latency_ms", "bandwidth_mbps", "packet_loss_pct", "error_count"]

# Iglewicz and Hoaglin's consistency constant, so the modified z-score is
# on roughly the same scale as a standard z-score under normal data.
_MAD_SCALE = 0.6745

# A sparse/discrete metric like error_count is 0 in the overwhelming
# majority of normal minutes, so its rolling MAD is exactly 0 most of the
# time by construction (more than half the window ties at the same
# value). Treating that as "zero spread, so even the smallest deviation is
# infinitely anomalous" was a real bug: an ordinary 0-to-1 blip got the
# same extreme score as a genuine fault. Flooring the denominator avoids
# a divide-by-near-zero blowup while still easily catching real faults,
# whose injected magnitude is always many times this floor.
_MIN_MAD = 1.0

# A device's first few minutes in the stream have almost no rolling
# history (1-2 prior points), so the median/MAD computed from them is
# nearly meaningless -- this was a second real bug: at minute_idx 0-2,
# nearly every device in the pipeline run spuriously registered as a
# candidate purely from this startup noise, not any real fault (visible
# as a report with ~190 "co-occurring" devices at minute_idx=1). Require
# a minimum amount of real history before trusting the statistic at all.
_MIN_HISTORY = 10


def _robust_zscore(value: float, window_values: list[float]) -> float:
    if len(window_values) < _MIN_HISTORY:
        return 0.0
    arr = np.array(window_values, dtype=float)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    safe_mad = max(mad, _MIN_MAD)
    return _MAD_SCALE * (value - median) / safe_mad


_robust_zscore_udf = F.udf(_robust_zscore, DoubleType())


def add_rolling_features(df: DataFrame, window_minutes: int = 30) -> DataFrame:
    """For each metric, adds a rolling median, rolling MAD, and a modified
    z-score for how far the current value deviates from that recent
    robust baseline (excluding the current row, so a spike cannot inflate
    its own baseline)."""
    device_window = (
        Window.partitionBy("device_id")
        .orderBy("minute_idx")
        .rowsBetween(-window_minutes, -1)
    )

    result = df
    for metric in METRICS:
        window_values_col = f"__{metric}_window_values"
        z_col = f"{metric}_zscore"

        result = result.withColumn(
            window_values_col, F.collect_list(F.col(metric).cast("double")).over(device_window)
        )
        result = result.withColumn(
            z_col, _robust_zscore_udf(F.col(metric).cast("double"), F.col(window_values_col))
        )
        result = result.drop(window_values_col)

    return result
