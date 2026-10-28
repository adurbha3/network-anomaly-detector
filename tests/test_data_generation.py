from app.data_generation import generate_telemetry


def test_row_count_matches_devices_times_minutes(spark):
    df = generate_telemetry(spark, n_devices=3, n_days=1, n_anomalies=5, seed=0)
    assert df.count() == 3 * 1440


def test_expected_columns_present(spark):
    df = generate_telemetry(spark, n_devices=2, n_days=1, n_anomalies=0, seed=0)
    expected = {
        "device_id", "minute_idx", "latency_ms", "bandwidth_mbps",
        "packet_loss_pct", "error_count", "anomaly_type", "is_anomaly",
    }
    assert expected.issubset(set(df.columns))


def test_reproducible_with_same_seed(spark):
    a = generate_telemetry(spark, n_devices=3, n_days=1, n_anomalies=5, seed=42)
    b = generate_telemetry(spark, n_devices=3, n_days=1, n_anomalies=5, seed=42)
    a_sum = a.selectExpr("sum(latency_ms) as s").collect()[0]["s"]
    b_sum = b.selectExpr("sum(latency_ms) as s").collect()[0]["s"]
    assert a_sum == b_sum


def test_no_anomalies_when_none_requested(spark):
    df = generate_telemetry(spark, n_devices=2, n_days=1, n_anomalies=0, seed=0)
    assert df.filter("is_anomaly").count() == 0


def test_injected_anomalies_are_labeled_consistently(spark):
    df = generate_telemetry(spark, n_devices=10, n_days=2, n_anomalies=20, seed=1)
    anomaly_count = df.filter("is_anomaly").count()
    typed_count = df.filter("anomaly_type IS NOT NULL").count()
    assert anomaly_count > 0
    assert anomaly_count == typed_count
