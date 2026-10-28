from app.features import add_rolling_features


def _make_series(spark, latencies):
    rows = [
        (0, i, float(lat), 500.0, 0.1, 0)
        for i, lat in enumerate(latencies)
    ]
    return spark.createDataFrame(
        rows, schema=["device_id", "minute_idx", "latency_ms", "bandwidth_mbps", "packet_loss_pct", "error_count"]
    )


def test_zscore_is_near_zero_without_enough_history(spark):
    latencies = [10.0, 20.0, 30.0]
    df = _make_series(spark, latencies)
    result = add_rolling_features(df, window_minutes=30)

    first_row = result.filter("minute_idx = 0").collect()[0]
    assert first_row["latency_ms_zscore"] == 0.0


def test_stable_series_has_near_zero_zscore(spark):
    latencies = [10.0, 11.0, 9.0, 10.5, 9.5] * 10
    df = _make_series(spark, latencies)
    result = add_rolling_features(df, window_minutes=30)

    last_row = result.filter("minute_idx = 49").collect()[0]
    assert abs(last_row["latency_ms_zscore"]) < 3.0


def test_single_point_spike_is_flagged(spark):
    latencies = [10.0] * 34 + [100.0]
    df = _make_series(spark, latencies)
    result = add_rolling_features(df, window_minutes=30)

    spike_row = result.filter("minute_idx = 34").collect()[0]
    assert spike_row["latency_ms_zscore"] > 3.0


def test_sustained_anomaly_stays_flagged_past_its_first_minute(spark):
    """Regression test: a plain mean/std rolling z-score gets numbed by
    its own outlier after just one or two minutes once that outlier sits
    inside the trailing window, because a single extreme value inflates
    the std dramatically. The robust median/MAD z-score should keep
    flagging a sustained anomaly for most of its duration instead of
    just its very first minute."""
    normal = [10.0 + (i % 3) for i in range(60)]
    sustained_anomaly = [100.0] * 20
    latencies = normal + sustained_anomaly

    df = _make_series(spark, latencies)
    result = add_rolling_features(df, window_minutes=30)

    anomaly_rows = result.filter("minute_idx >= 60").collect()
    flagged = sum(1 for row in anomaly_rows if row["latency_ms_zscore"] > 3.0)
    assert flagged >= 15  # most of the 20-minute anomaly, not just the first minute or two
