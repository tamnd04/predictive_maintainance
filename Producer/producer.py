import os
import time
import json
import pandas as pd
from kafka import KafkaProducer

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# 1) NASA Bearing Dataset (Kaggle: vinayak123tyagi/bearing-dataset)
# Root folder *inside the container* (see docker-compose volume mappings)
# Expected structure:
#   /app/NASA_Bearing_Data/1st_test/1st_test
#   /app/NASA_Bearing_Data/2nd_test/2nd_test
#   /app/NASA_Bearing_Data/3rd_test/3rd_test
BEARING_ROOT_DIR = "/app/NASA_Bearing_Data"

# 2) CNC Mill Tool Wear dataset (Kaggle: shasun/tool-wear-detection-in-cnc-mill)
# Expected structure:
#   /app/CNC_Mill_Tool_Wear/train.csv
#   /app/CNC_Mill_Tool_Wear/experiment_01.csv
#   ...
TOOLWEAR_ROOT_DIR = "/app/CNC_Mill_Tool_Wear"

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")

# Topics
BEARING_TOPIC = os.getenv("BEARING_TOPIC", "bearing_raw")
TOOLWEAR_TOPIC = os.getenv("TOOLWEAR_TOPIC", "toolwear_raw")

# Streaming speed controls (seconds)
# (0 = max speed; 30 = 30 sec wait between bearing files)
BEARING_FILE_INTERVAL = float(os.getenv("BEARING_FILE_INTERVAL", "30"))
TOOLWEAR_ROW_INTERVAL = float(os.getenv("TOOLWEAR_ROW_INTERVAL", "0.0"))

# Which datasets to stream: "bearing", "toolwear", or "both"
DATASETS_TO_STREAM = os.getenv("DATASETS_TO_STREAM", "both").lower()


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def create_kafka_producer() -> KafkaProducer:
    """Create a KafkaProducer with JSON value serializer and some sensible defaults."""
    producer = None
    print(f"Connecting to Kafka at {KAFKA_BOOTSTRAP_SERVERS} ...")
    while producer is None:
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                request_timeout_ms=120000,
                max_block_ms=120000,
                batch_size=32768,
                linger_ms=50,
                delivery_timeout_ms=130000,
            )
            print("Connected to Kafka!")
        except Exception as e:
            print("Kafka not ready yet, retrying in 5s:", e)
            time.sleep(5)

    # Give Kafka time to finish waking up / creating topics
    print("Warming up producer for 10 seconds...")
    time.sleep(10)
    return producer


def get_sorted_files(directory: str):
    """Return list of filenames sorted by time (filenames are timestamps)."""
    if not os.path.isdir(directory):
        print(f"[BEARING] WARNING: Directory does not exist: {directory}")
        return []

    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    files.sort()  # 'YYYY.MM.DD.HH.MM.SS' sorts correctly as strings
    print(f"[BEARING] Found {len(files)} files in {directory}")
    if files:
        print(f"[BEARING] Sample files: {files[:5]}")
    return files


def filename_to_timestamp(filename: str) -> str:
    """
    Convert NASA bearing filename into a human-readable timestamp string.
    Example:
        '2004.02.12.10.32.39' -> '2004-02-12 10:32:39'
    """
    name = os.path.splitext(filename)[0]
    parts = name.split(".")
    if len(parts) == 6:
        date_part = "-".join(parts[:3])
        time_part = ":".join(parts[3:])
        return f"{date_part} {time_part}"
    return name


# ---------------------------------------------------------------------------
# NASA BEARING DATASET STREAMING (1st_test, 2nd_test, 3rd_test)
# ---------------------------------------------------------------------------

def stream_bearing_dataset(producer: KafkaProducer):
    """
    Stream NASA Bearing Kaggle dataset files to Kafka.

    Reads 1st_test "like the reference file":
      - DATA_DIR = /app/NASA_Bearing_Data/1st_test/1st_test
      - files are tab-separated with 8 columns:
          b1a, b1b, b2a, b2b, b3a, b3b, b4a, b4b
      - we send b1..b4 from the 'a' channels (b1a, b2a, b3a, b4a)

    Additionally reads:
      - /app/NASA_Bearing_Data/2nd_test/2nd_test  (4 columns: bearings 1..4)
      - /app/NASA_Bearing_Data/3rd_test/3rd_test  (4 columns)
      and maps them to b1a..b4a so the outgoing messages stay consistent:
          "b1", "b2", "b3", "b4"
    """

    tests = [
        ("1st_test", os.path.join(BEARING_ROOT_DIR, "1st_test", "1st_test")),
        ("2nd_test", os.path.join(BEARING_ROOT_DIR, "2nd_test", "2nd_test")),
        ("3rd_test", os.path.join(BEARING_ROOT_DIR, "3rd_test", "3rd_test")),
    ]

    total_files = 0

    for test_name, data_dir in tests:
        print(f"[BEARING] Looking for files in {data_dir} (test set: {test_name})")
        files = get_sorted_files(data_dir)
        if not files:
            continue

        for filename in files:
            file_path = os.path.join(data_dir, filename)
            timestamp_str = filename_to_timestamp(filename)

            try:
                df = pd.read_csv(file_path, sep="\t", header=None)
            except Exception as e:
                print(f"[BEARING] Skipping bad file {file_path}: {e}")
                continue

            # 1st_test: 8 columns (b1a,b1b,b2a,b2b,b3a,b3b,b4a,b4b)
            # 2nd/3rd_test: 4 columns (one axis per bearing)
            if test_name == "1st_test":
                if df.shape[1] != 8:
                    print(f"[BEARING] Unexpected number of columns ({df.shape[1]}) in {file_path}, expected 8, skipping.")
                    continue
                df.columns = ["b1a", "b1b", "b2a", "b2b", "b3a", "b3b", "b4a", "b4b"]
            else:
                if df.shape[1] != 4:
                    print(f"[BEARING] Unexpected number of columns ({df.shape[1]}) in {file_path}, expected 4, skipping.")
                    continue
                df.columns = ["b1a", "b2a", "b3a", "b4a"]

            total_files += 1
            print(f"[BEARING] Streaming file {filename} from {data_dir} "
                  f"({len(df)} rows, {df.shape[1]} cols)")

            # Loop through rows and send each as a message
            for _, row in df.iterrows():
                message = {
                    "dataset": "bearing",
                    "test_set": test_name,
                    "timestamp": timestamp_str,
                    "file": filename,
                    # use "a" channels for all tests, just like the reference producer
                    "b1": float(row["b1a"]),
                    "b2": float(row["b2a"]),
                    "b3": float(row["b3a"]),
                    "b4": float(row["b4a"]),
                }
                producer.send(BEARING_TOPIC, message)

            # Flush ensures all messages from this file are sent before we sleep
            producer.flush()
            time.sleep(BEARING_FILE_INTERVAL)

    if total_files == 0:
        print(
            f"[BEARING] No files found under {BEARING_ROOT_DIR} for 1st/2nd/3rd_test. "
            f"Check that the Kaggle bearing-dataset is unzipped under "
            f"1st_test/1st_test, 2nd_test/2nd_test, 3rd_test/3rd_test."
        )
    else:
        print(f"[BEARING] Finished streaming {total_files} files across test sets 1–3.")


# ---------------------------------------------------------------------------
# CNC MILL TOOL WEAR DATASET STREAMING (unchanged)
# ---------------------------------------------------------------------------

def stream_toolwear_dataset(producer: KafkaProducer):
    """
    Stream CNC Mill Tool Wear dataset to Kafka.

    The dataset contains:
      - 18 per-experiment CSVs with time-series sensor data
      - train.csv with metadata (tool condition, etc.)

    For simplicity we:
      1) Look for all CSV files except 'train.csv' in TOOLWEAR_ROOT_DIR
      2) Stream each row as a message and attach:
          - dataset = "toolwear"
          - experiment = file name (without extension)
      3) Optionally, if train.csv is present and has a matching 'exp_number'
         or similar column, we merge metadata before sending.
    """
    if not os.path.isdir(TOOLWEAR_ROOT_DIR):
        print(f"[TOOLWEAR] Directory does not exist: {TOOLWEAR_ROOT_DIR}")
        return

    # Try to load train.csv metadata if present
    metadata_path = os.path.join(TOOLWEAR_ROOT_DIR, "train.csv")
    metadata_df = None
    if os.path.isfile(metadata_path):
        try:
            metadata_df = pd.read_csv(metadata_path)
            print(f"[TOOLWEAR] Loaded metadata from {metadata_path} with {len(metadata_df)} rows")
        except Exception as e:
            print(f"[TOOLWEAR] Could not read metadata train.csv: {e}")

    # Experiment CSVs (exclude train.csv itself)
    experiment_files = [
        f for f in os.listdir(TOOLWEAR_ROOT_DIR)
        if f.lower().endswith(".csv") and f.lower() != "train.csv"
    ]
    experiment_files.sort()

    if not experiment_files:
        print(
            f"[TOOLWEAR] No experiment CSV files found in {TOOLWEAR_ROOT_DIR}. "
            f"Make sure the Kaggle CNC Mill Tool Wear CSVs are unzipped there."
        )
        return

    print(f"[TOOLWEAR] Found {len(experiment_files)} experiment files")

    for exp_file in experiment_files:
        exp_path = os.path.join(TOOLWEAR_ROOT_DIR, exp_file)
        experiment_id = os.path.splitext(exp_file)[0]

        try:
            df = pd.read_csv(exp_path)
        except Exception as e:
            print(f"[TOOLWEAR] Skipping bad file {exp_path}: {e}")
            continue

        print(f"[TOOLWEAR] Streaming experiment {experiment_id} ({len(df)} rows)")

        # If metadata is available, try to find matching row(s) for this experiment
        meta_row = None
        if metadata_df is not None:
            # Heuristic: look for a column containing experiment id (adapt if your column names differ)
            possible_keys = ["exp_number", "experiment", "Experiment", "file_id"]
            key_col = None
            for col in metadata_df.columns:
                if col in possible_keys:
                    key_col = col
                    break
            if key_col is not None:
                meta_matches = metadata_df[metadata_df[key_col].astype(str) == experiment_id]
                if not meta_matches.empty:
                    meta_row = meta_matches.iloc[0].to_dict()

        for _, row in df.iterrows():
            payload = row.to_dict()

            # Ensure all numpy types are converted to native Python types
            for k, v in list(payload.items()):
                if hasattr(v, "item"):
                    try:
                        payload[k] = v.item()
                    except Exception:
                        pass

            message = {
                "dataset": "toolwear",
                "experiment": experiment_id,
                "timestamp": payload.get("time", None),  # adjust if column named differently
                "payload": payload,
            }

            if meta_row is not None:
                message["metadata"] = meta_row

            producer.send(TOOLWEAR_TOPIC, message)

            if TOOLWEAR_ROW_INTERVAL > 0:
                time.sleep(TOOLWEAR_ROW_INTERVAL)

        producer.flush()

    print("[TOOLWEAR] Finished streaming all experiments.")


# ---------------------------------------------------------------------------
# MAIN ENTRYPOINT
# ---------------------------------------------------------------------------

def run_producer():
    producer = create_kafka_producer()

    # Decide which datasets to stream based on env var
    stream_bearing = DATASETS_TO_STREAM in ("bearing", "both")
    stream_toolwear = DATASETS_TO_STREAM in ("toolwear", "both")

    if not (stream_bearing or stream_toolwear):
        print(f"No valid DATASETS_TO_STREAM specified: {DATASETS_TO_STREAM}")
        return

    if stream_bearing:
        stream_bearing_dataset(producer)

    if stream_toolwear:
        stream_toolwear_dataset(producer)

    print("All requested datasets have been streamed. Exiting producer.")


if __name__ == "__main__":
    run_producer()
