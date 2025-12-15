# main.py
"""
PCOS Machine Learning Project (Educational Only)

This small project:
- Loads a public PCOS dataset from a CSV file
- Performs simple data cleaning and preprocessing
- Trains a machine learning classifier to predict PCOS (yes/no)
- Evaluates the model with accuracy and a classification report
- Shows basic feature importance

IMPORTANT:
This project is ONLY for learning and educational purposes.
It must NOT be used for medical diagnosis, treatment decisions, or any real-world clinical use.
Always consult a qualified medical professional for health-related decisions.
"""

import os
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def find_label_column(df: pd.DataFrame) -> str:
    """
    Try to automatically find the PCOS label column.
    Many public PCOS datasets use names like:
    - 'PCOS (Y/N)'
    - 'PCOS..Y.N.'
    - 'PCOS'
    This helper searches for a column name that contains 'PCOS'.
    """
    candidates: List[str] = []
    for col in df.columns:
        col_upper = col.upper()
        if "PCOS" in col_upper:
            candidates.append(col)

    if not candidates:
        raise ValueError(
            "Could not find a label column containing 'PCOS' in the column names."
        )

    print(f"Detected label column: {candidates[0]}")
    return candidates[0]


def clean_and_split_data(df: pd.DataFrame, label_col: str):
    """
    Enhanced data cleaning for the PCOS dataset.

    Improvements:
    - Encodes "Yes"/"No" into 1/0
    - Converts other string numbers into float
    - One-hot encodes categorical features
    - Keeps only informative features
    """

    # 1. Drop ID-like columns
    id_like_keywords = ["SL", "PATIENT", "FILE", "ID", "NO"]
    cols_to_drop = [col for col in df.columns if any(k in col.upper() for k in id_like_keywords)]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # 2. Handle label column
    y_raw = df[label_col]
    df = df.drop(columns=[label_col])

    # Map Y/N or Yes/No to 1/0
    if y_raw.dtype == object:
        mapping = {"Y": 1, "N": 0, "YES": 1, "NO": 0, "Yes": 1, "No": 0}
        y = y_raw.map(lambda v: mapping.get(str(v).strip(), np.nan))
        y = y.astype(float).fillna(y.mode()[0]).astype(int)
    else:
        y = y_raw.astype(int)

    # 3. Convert numeric-like strings (e.g., "36.0") into floats
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # 4. Encode "Yes"/"No" in feature columns
    yesno_map = {"Y": 1, "N": 0, "YES": 1, "NO": 0, "Yes": 1, "No": 0}
    for col in df.columns:
        if df[col].dtype == object:
            unique_vals = df[col].dropna().unique()
            if set(unique_vals).issubset(set(yesno_map.keys())):
                df[col] = df[col].map(yesno_map)

    # 5. One-hot encode remaining categorical columns
    df = pd.get_dummies(df, drop_first=True)

    # 6. Fill missing values
    df = df.fillna(df.median(numeric_only=True))

    # 7. Train/test split
    X = df.values
    feature_names = list(df.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_names


def train_and_evaluate_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
) -> None:
    """
    Train a Random Forest classifier and evaluate it.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=150, random_state=42, n_jobs=-1
    )
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f}\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\nTop 10 features:")
    for rank, idx in enumerate(indices[:10], start=1):
        print(f"{rank}. {feature_names[idx]}: {importances[idx]:.4f}")


def main():
    csv_path = os.path.join("data", "PCOS_infertility.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found at {csv_path}. Place PCOS_infertility.csv inside /data folder."
        )

    print("Loading dataset...")
    df = pd.read_csv(csv_path)

    print(f"Dataset shape: {df.shape}")
    print("Columns:", df.columns)

    label_col = find_label_column(df)
    X_train, X_test, y_train, y_test, feature_names = clean_and_split_data(df, label_col)

    print("\nTraining model...")
    train_and_evaluate_model(X_train, X_test, y_train, y_test, feature_names)


if __name__ == "__main__":
    main()
