import os
from pathlib import Path

import joblib
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, first
import pyspark.sql.functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
)

# -------------------------------------------------------------------
# 1. Spark Session
# -------------------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("BearingToolWearStreamingPrediction_Synapse")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# -------------------------------------------------------------------
# 2. Load scikit-learn models
# -------------------------------------------------------------------
this_file = Path(__file__).resolve()
repo_root = this_file.parents[1]

bearing_model_path = repo_root / "models" / "bearing_rul_sklearn.pkl"
toolwear_model_path = repo_root / "models" / "toolwear_clf_sklearn.pkl"

print(f"[STREAM] Loading bearing model from {bearing_model_path}")
bearing_bundle = joblib.load(bearing_model_path)
bearing_model = bearing_bundle["model"]
bearing_feature_cols = bearing_bundle["feature_cols"]

print(f"[STREAM] Loading toolwear model from {toolwear_model_path}")
toolwear_bundle = joblib.load(toolwear_model_path)
toolwear_model = toolwear_bundle["model"]
toolwear_feature_cols = toolwear_bundle["feature_cols"]
toolwear_classes = toolwear_bundle["classes"]

# -------------------------------------------------------------------
# 3. JDBC connection info for Azure Synapse
# -------------------------------------------------------------------
jdbc_url = os.getenv("AZ_SYNAPSE_JDBC_URL")
jdbc_user = os.getenv("AZ_SYNAPSE_USER")
jdbc_password = os.getenv("AZ_SYNAPSE_PASSWORD")

bearing_table = os.getenv("AZ_SYNAPSE_BEARING_TABLE", "dbo.BearingPredictions")
toolwear_table = os.getenv("AZ_SYNAPSE_TOOLWEAR_TABLE", "dbo.ToolwearPredictions")

if not jdbc_url or not jdbc_user or not jdbc_password:
    raise RuntimeError(
        "AZ_SYNAPSE_JDBC_URL, AZ_SYNAPSE_USER, AZ_SYNAPSE_PASSWORD "
        "must be set in the environment."
    )

jdbc_props = {
    "user": jdbc_user,
    "password": jdbc_password,
    "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver",
}

# -------------------------------------------------------------------
# 4. Define schemas for JSON payloads (must match your producer)
# -------------------------------------------------------------------

# Bearing messages from producer:
# {
#   "dataset": "bearing",
#   "test_set": "1st_test" | "2nd_test" | "3rd_test",
#   "timestamp": "2004-02-12 10:32:39",
#   "file": "2004.02.12.10.32.39",
#   "b1": float,
#   "b2": float,
#   "b3": float,
#   "b4": float
# }
bearing_schema = StructType([
    StructField("dataset", StringType(), True),
    StructField("test_set", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("file", StringType(), True),
    StructField("b1", DoubleType(), True),
    StructField("b2", DoubleType(), True),
    StructField("b3", DoubleType(), True),
    StructField("b4", DoubleType(), True),
])

# Tool wear messages from producer:
# {
#   "dataset": "toolwear",
#   "experiment": "experiment_01",
#   "timestamp": <time>,
#   "payload": { many sensor columns },
#   "metadata": {
#       "feedrate": ...,
#       "clamp_pressure": ...,
#       "tool_condition": "...",
#       ...
#   }
# }
payload_schema = StructType([
    StructField("time", DoubleType(), True),
    StructField("X1_ActualPosition", DoubleType(), True),
    StructField("Y1_ActualPosition", DoubleType(), True),
    StructField("Z1_ActualPosition", DoubleType(), True),
    StructField("X1_DCBusVoltage", DoubleType(), True),
    StructField("Y1_DCBusVoltage", DoubleType(), True),
    StructField("S1_DCBusVoltage", DoubleType(), True),
    StructField("X1_OutputCurrent", DoubleType(), True),
    StructField("Y1_OutputCurrent", DoubleType(), True),
    StructField("S1_OutputCurrent", DoubleType(), True),
    StructField("X1_OutputVoltage", DoubleType(), True),
    StructField("Y1_OutputVoltage", DoubleType(), True),
    StructField("S1_OutputVoltage", DoubleType(), True),
    StructField("M1_CURRENT_FEEDRATE", DoubleType(), True),
    StructField("S1_ActualVelocity", DoubleType(), True),
    StructField("S1_CurrentFeedback", DoubleType(), True),
    StructField("S1_CommandVelocity", DoubleType(), True),
    StructField("M1_sequence_number", DoubleType(), True),
    StructField("S1_OutputPower", DoubleType(), True),
])

metadata_schema = StructType([
    StructField("feedrate", DoubleType(), True),
    StructField("clamp_pressure", DoubleType(), True),
    StructField("tool_condition", StringType(), True),
    StructField("machining_finalized", StringType(), True),
    StructField("passed_visual_inspection", StringType(), True),
])

toolwear_schema = StructType([
    StructField("dataset", StringType(), True),
    StructField("experiment", StringType(), True),
    StructField("timestamp", DoubleType(), True),
    StructField("payload", payload_schema, True),
    StructField("metadata", metadata_schema, True),
])

# -------------------------------------------------------------------
# 5. Read from Kafka topics
# -------------------------------------------------------------------
kafka_bootstrap = "localhost:9092"  # from HOST perspective

raw_df = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", kafka_bootstrap)
    .option("subscribe", "bearing_raw,toolwear_raw")
    .option("startingOffsets", "earliest")
    .load()
)

json_df = raw_df.selectExpr(
    "CAST(topic AS STRING) as topic",
    "CAST(value AS STRING) as value"
)

# -------------------------------------------------------------------
# 6. Parse JSON into structured dataframes
# -------------------------------------------------------------------

bearing_stream = (
    json_df
    .filter(col("topic") == "bearing_raw")
    .select(from_json(col("value"), bearing_schema).alias("data"))
    .select("data.*")
)

toolwear_stream = (
    json_df
    .filter(col("topic") == "toolwear_raw")
    .select(from_json(col("value"), toolwear_schema).alias("data"))
    .select("data.*")
)

# -------------------------------------------------------------------
# 7. Bearing: aggregate per file to compute features (max, p2p, rms)
# -------------------------------------------------------------------

bearing_features = (
    bearing_stream
    .groupBy("test_set", "file")
    .agg(
        first("timestamp").alias("timestamp"),

        # b1
        F.max(F.abs(col("b1"))).alias("b1_max"),
        (F.max(F.abs(col("b1"))) + F.abs(F.min(col("b1")))).alias("b1_p2p"),
        F.sqrt(F.avg(col("b1") * col("b1"))).alias("b1_rms"),

        # b2
        F.max(F.abs(col("b2"))).alias("b2_max"),
        (F.max(F.abs(col("b2"))) + F.abs(F.min(col("b2")))).alias("b2_p2p"),
        F.sqrt(F.avg(col("b2") * col("b2"))).alias("b2_rms"),

        # b3
        F.max(F.abs(col("b3"))).alias("b3_max"),
        (F.max(F.abs(col("b3"))) + F.abs(F.min(col("b3")))).alias("b3_p2p"),
        F.sqrt(F.avg(col("b3") * col("b3"))).alias("b3_rms"),

        # b4
        F.max(F.abs(col("b4"))).alias("b4_max"),
        (F.max(F.abs(col("b4"))) + F.abs(F.min(col("b4")))).alias("b4_p2p"),
        F.sqrt(F.avg(col("b4") * col("b4"))).alias("b4_rms"),
    )
)

# -------------------------------------------------------------------
# 8. Tool wear: flatten into feature columns used in training
# -------------------------------------------------------------------

toolwear_features = toolwear_stream.select(
    col("experiment"),
    col("timestamp"),
    col("dataset"),

    col("metadata.clamp_pressure").alias("clamp_pressure"),
    col("metadata.feedrate").alias("feedrate"),
    col("metadata.tool_condition").alias("tool_condition_true"),

    col("payload.X1_ActualPosition").alias("X1_ActualPosition"),
    col("payload.Y1_ActualPosition").alias("Y1_ActualPosition"),
    col("payload.Z1_ActualPosition").alias("Z1_ActualPosition"),
    col("payload.X1_DCBusVoltage").alias("X1_DCBusVoltage"),
    col("payload.Y1_DCBusVoltage").alias("Y1_DCBusVoltage"),
    col("payload.S1_DCBusVoltage").alias("S1_DCBusVoltage"),
    col("payload.X1_OutputCurrent").alias("X1_OutputCurrent"),
    col("payload.Y1_OutputCurrent").alias("Y1_OutputCurrent"),
    col("payload.S1_OutputCurrent").alias("S1_OutputCurrent"),
    col("payload.X1_OutputVoltage").alias("X1_OutputVoltage"),
    col("payload.Y1_OutputVoltage").alias("Y1_OutputVoltage"),
    col("payload.S1_OutputVoltage").alias("S1_OutputVoltage"),
    col("payload.M1_CURRENT_FEEDRATE").alias("M1_CURRENT_FEEDRATE"),
    col("payload.S1_ActualVelocity").alias("S1_ActualVelocity"),
    col("payload.S1_CurrentFeedback").alias("S1_CurrentFeedback"),
    col("payload.S1_CommandVelocity").alias("S1_CommandVelocity"),
    col("payload.M1_sequence_number").alias("M1_sequence_number"),
    col("payload.S1_OutputPower").alias("S1_OutputPower"),
)

# -------------------------------------------------------------------
# 9. foreachBatch: run scikit-learn models and write to Synapse
# -------------------------------------------------------------------

def predict_bearing_batch(df, batch_id: int):
    if df.rdd.isEmpty():
        return

    pdf = df.toPandas()
    if pdf.empty:
        return

    # Ensure all expected feature columns exist (fill missing with 0.0)
    for col_name in bearing_feature_cols:
        if col_name not in pdf.columns:
            pdf[col_name] = 0.0

    X = pdf[bearing_feature_cols].values
    pdf["rul_prediction"] = bearing_model.predict(X)

    # Select only what we want in the DB
    pdf_out = pdf[["test_set", "file", "timestamp", "rul_prediction"]].copy()
    pdf_out["batch_id"] = batch_id

    sdf_out = spark.createDataFrame(pdf_out)

    (
        sdf_out.write
        .mode("append")
        .jdbc(jdbc_url, bearing_table, properties=jdbc_props)
    )
    print(f"[STREAM] Wrote bearing predictions for batch {batch_id} to {bearing_table}")


def predict_toolwear_batch(df, batch_id: int):
    if df.rdd.isEmpty():
        return

    pdf = df.toPandas()
    if pdf.empty:
        return

    # Ensure all expected feature columns exist (fill missing with 0.0)
    for col_name in toolwear_feature_cols:
        if col_name not in pdf.columns:
            pdf[col_name] = 0.0

    X = pdf[toolwear_feature_cols].values
    pred_int = toolwear_model.predict(X)

    label_map = {i: name for i, name in enumerate(toolwear_classes)}
    pdf["tool_condition_prediction"] = [label_map[i] for i in pred_int]

    # Select only what we want in the DB
    pdf_out = pdf[["experiment", "timestamp", "tool_condition_prediction", "tool_condition_true"]].copy()
    pdf_out["batch_id"] = batch_id

    sdf_out = spark.createDataFrame(pdf_out)

    (
        sdf_out.write
        .mode("append")
        .jdbc(jdbc_url, toolwear_table, properties=jdbc_props)
    )
    print(f"[STREAM] Wrote toolwear predictions for batch {batch_id} to {toolwear_table}")


bearing_query = (
    bearing_features
    .writeStream
    .outputMode("update")
    .foreachBatch(predict_bearing_batch)
    .start()
)

toolwear_query = (
    toolwear_features
    .writeStream
    .outputMode("append")
    .foreachBatch(predict_toolwear_batch)
    .start()
)

spark.streams.awaitAnyTermination()
