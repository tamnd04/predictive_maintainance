import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


def build_cnc_toolwear_dataframe(root: Path) -> pd.DataFrame:
    """
    Build a single pandas DataFrame with:
      - time-series sensor data from experiment_*.csv
      - metadata (including tool_condition) from train.csv
    Metadata for each row is broadcast from the matching experiment's row in train.csv,
    assuming that:
      - experiment files sorted by name correspond to rows in train.csv by order.
    """
    root = Path(root)

    # Find all experiment CSVs (exclude train.csv)
    exp_files = sorted(
        [
            p
            for p in root.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".csv"
            and p.name.lower() != "train.csv"
        ],
        key=lambda p: p.name,
    )

    if not exp_files:
        raise RuntimeError(
            f"No experiment CSVs found under {root}. "
            "Expected files like experiment_01.csv, experiment_02.csv, ..."
        )

    train_path = root / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train.csv at {train_path}")

    meta = pd.read_csv(train_path)

    if len(exp_files) != len(meta):
        print(
            "[TOOLWEAR] WARNING: number of experiment files and train.csv rows differ "
            f"({len(exp_files)} vs {len(meta)}). "
            "Will only use the minimum of the two."
        )

    n = min(len(exp_files), len(meta))
    rows = []

    for i in range(n):
        exp_path = exp_files[i]
        exp_df = pd.read_csv(exp_path)

        meta_row = meta.iloc[i]

        # Attach metadata columns to every row in this experiment
        for col, val in meta_row.items():
            exp_df[col] = val

        exp_df["experiment"] = exp_path.stem  # e.g. "experiment_01"
        rows.append(exp_df)

    final_data = pd.concat(rows, ignore_index=True)

    # Basic cleaning
    final_data = final_data.dropna().reset_index(drop=True)
    return final_data


def main():
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[1]

    # Where you unzipped the CNC Mill tool wear Kaggle dataset on the HOST
    default_root = repo_root / "cnc-mill-tool-wear"
    if not default_root.exists():
        # fallback to alternative capitalization
        default_root = repo_root / "CNC_Mill_Tool_Wear"

    cnc_root = Path(os.getenv("CNC_TOOLWEAR_ROOT", str(default_root)))

    if not cnc_root.exists():
        raise FileNotFoundError(
            f"CNC tool wear root not found. Expected at {cnc_root}. "
            f"Set CNC_TOOLWEAR_ROOT env var or adjust the path in train_toolwear_model.py."
        )

    print(f"[TOOLWEAR] Building preprocessed DataFrame from {cnc_root}")
    final_data = build_cnc_toolwear_dataframe(cnc_root)
    print(f"[TOOLWEAR] Final data shape (after cleaning): {final_data.shape}")

    # -----------------------------
    # Select features & label
    # -----------------------------
    if "tool_condition" not in final_data.columns:
        raise RuntimeError(
            "Expected 'tool_condition' column in train.csv metadata, "
            "but it was not found."
        )

    # Encode tool_condition as integers, keep mapping
    label_series = final_data["tool_condition"].astype("category")
    y = label_series.cat.codes.values
    classes = list(label_series.cat.categories)

    # Candidate feature columns (must exist in both training & streaming)
    candidate_feature_cols = [
        "clamp_pressure",
        "feedrate",
        "X1_ActualPosition",
        "Y1_ActualPosition",
        "Z1_ActualPosition",
        "X1_DCBusVoltage",
        "Y1_DCBusVoltage",
        "S1_DCBusVoltage",
        "X1_OutputCurrent",
        "Y1_OutputCurrent",
        "S1_OutputCurrent",
        "X1_OutputVoltage",
        "Y1_OutputVoltage",
        "S1_OutputVoltage",
        "M1_CURRENT_FEEDRATE",
        "S1_ActualVelocity",
        "S1_CurrentFeedback",
        "S1_CommandVelocity",
        "M1_sequence_number",
        "S1_OutputPower",
    ]

    feature_cols = [c for c in candidate_feature_cols if c in final_data.columns]

    if not feature_cols:
        raise RuntimeError(
            "[TOOLWEAR] No overlap between candidate_feature_cols and data columns. "
            "Check the column names in your CNC dataset."
        )

    X = final_data[feature_cols].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = GradientBoostingClassifier(random_state=42)

    print("[TOOLWEAR] Training GradientBoostingClassifier (scikit-learn)...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[TOOLWEAR] Test accuracy (scikit-learn): {acc:.3f}")
    print("[TOOLWEAR] Classification report:")
    print(classification_report(y_test, y_pred, target_names=classes))

    # Save model under repo_root/models/toolwear_clf_sklearn.pkl
    models_dir = repo_root / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "toolwear_clf_sklearn.pkl"

    bundle = {
        "model": clf,
        "feature_cols": feature_cols,
        "classes": classes,  # index -> label name
    }
    joblib.dump(bundle, model_path)
    print(f"[TOOLWEAR] Saved scikit-learn model to {model_path}")


if __name__ == "__main__":
    main()
