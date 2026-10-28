"""Generates synthetic, seeded network telemetry data at scale using
PySpark's own DataFrame API (not pandas/numpy row loops): per-device
diurnal traffic patterns plus injected fault windows (latency spikes,
packet loss bursts, bandwidth drops, and compound device faults) with a
ground-truth anomaly label kept alongside for evaluation."""
from __future__ import annotations

import math
import random

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructField, StructType, StringType, IntegerType, DoubleType

ANOMALY_TYPES = ["latency_spike", "packet_loss_burst", "bandwidth_drop", "device_fault"]

_DEVICE_SCHEMA = StructType([
    StructField("device_id", IntegerType(), False),
    StructField("base_latency_ms", DoubleType(), False),
    StructField("base_bandwidth_mbps", DoubleType(), False),
    StructField("base_packet_loss_pct", DoubleType(), False),
    StructField("phase_offset", DoubleType(), False),
])

_WINDOW_SCHEMA = StructType([
    StructField("device_id", IntegerType(), False),
    StructField("anomaly_type", StringType(), False),
    StructField("start_minute", IntegerType(), False),
    StructField("end_minute", IntegerType(), False),
])


def _generate_devices(spark: SparkSession, n_devices: int, seed: int) -> DataFrame:
    rng = random.Random(seed)
    rows = [
        (
            device_id,
            rng.uniform(15.0, 60.0),
            rng.uniform(200.0, 1000.0),
            rng.uniform(0.0, 0.5),
            rng.uniform(0.0, 2 * math.pi),
        )
        for device_id in range(n_devices)
    ]
    return spark.createDataFrame(rows, schema=_DEVICE_SCHEMA)


def _generate_anomaly_windows(
    n_devices: int, n_minutes: int, n_anomalies: int, seed: int
) -> list[tuple[int, str, int, int]]:
    rng = random.Random(seed + 1)
    windows = []
    for _ in range(n_anomalies):
        device_id = rng.randrange(n_devices)
        anomaly_type = rng.choice(ANOMALY_TYPES)
        duration = rng.randint(10, 60)
        start_minute = rng.randrange(0, max(1, n_minutes - duration))
        windows.append((device_id, anomaly_type, start_minute, start_minute + duration))
    return windows


def generate_telemetry(
    spark: SparkSession,
    n_devices: int = 200,
    n_days: int = 14,
    n_anomalies: int = 150,
    seed: int = 0,
) -> DataFrame:
    """Returns one row per (device, minute) with telemetry metrics, an
    injected anomaly_type (or None), and a boolean is_anomaly ground truth
    label -- all computed through Spark DataFrame transformations."""
    n_minutes = n_days * 1440

    devices = _generate_devices(spark, n_devices, seed)
    minutes = spark.range(n_minutes).withColumnRenamed("id", "minute_idx")
    base = devices.crossJoin(minutes)

    minute_of_day = F.col("minute_idx") % 1440
    diurnal = F.sin(2 * math.pi * minute_of_day / 1440 + F.col("phase_offset"))

    base = base.withColumn("minute_of_day", minute_of_day).withColumn(
        "latency_ms",
        F.col("base_latency_ms") + 8.0 * diurnal + F.randn(seed) * 2.0,
    ).withColumn(
        "bandwidth_mbps",
        F.col("base_bandwidth_mbps") - 100.0 * diurnal + F.randn(seed + 1) * 15.0,
    ).withColumn(
        "packet_loss_pct",
        F.greatest(F.lit(0.0), F.col("base_packet_loss_pct") + F.randn(seed + 2) * 0.1),
    ).withColumn(
        "error_count",
        F.greatest(F.lit(0), F.round(F.abs(F.randn(seed + 3)) * 0.3).cast(IntegerType())),
    )

    windows = _generate_anomaly_windows(n_devices, n_minutes, n_anomalies, seed)
    if windows:
        windows_df = spark.createDataFrame(windows, schema=_WINDOW_SCHEMA)
        base = base.join(
            F.broadcast(windows_df),
            on=(
                (base.device_id == windows_df.device_id)
                & (base.minute_idx >= windows_df.start_minute)
                & (base.minute_idx < windows_df.end_minute)
            ),
            how="left",
        ).select(base["*"], windows_df.anomaly_type)
    else:
        base = base.withColumn("anomaly_type", F.lit(None).cast(StringType()))

    base = base.withColumn(
        "latency_ms",
        F.when(F.col("anomaly_type") == "latency_spike", F.col("latency_ms") * 4.5)
        .when(F.col("anomaly_type") == "device_fault", F.col("latency_ms") * 2.5)
        .otherwise(F.col("latency_ms")),
    ).withColumn(
        "packet_loss_pct",
        F.when(F.col("anomaly_type") == "packet_loss_burst", F.col("packet_loss_pct") + 35.0)
        .when(F.col("anomaly_type") == "device_fault", F.col("packet_loss_pct") + 15.0)
        .otherwise(F.col("packet_loss_pct")),
    ).withColumn(
        "bandwidth_mbps",
        F.when(F.col("anomaly_type") == "bandwidth_drop", F.col("bandwidth_mbps") * 0.2)
        .otherwise(F.col("bandwidth_mbps")),
    ).withColumn(
        "error_count",
        F.when(F.col("anomaly_type") == "device_fault", F.col("error_count") + 20)
        .otherwise(F.col("error_count")),
    ).withColumn("is_anomaly", F.col("anomaly_type").isNotNull())

    return base.select(
        "device_id", "minute_idx", "latency_ms", "bandwidth_mbps",
        "packet_loss_pct", "error_count", "anomaly_type", "is_anomaly",
    )
