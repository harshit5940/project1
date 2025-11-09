import os
import io
import requests
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

IBM_TELCO_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DEFAULT_DATA_PATH = os.path.join("data", "Telco-Customer-Churn.csv")


def ensure_data(local_path: str = DEFAULT_DATA_PATH, url: str = IBM_TELCO_URL) -> str:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path):
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
    return local_path


def load_data(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    path = ensure_data(path)
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Fix TotalCharges numeric conversion
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Strip spaces in column names
    df.columns = [c.strip() for c in df.columns]

    # Normalize target
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Tenure group feature
    if "tenure" in df.columns:
        bins = [0, 12, 24, 48, 72, np.inf]
        labels = ["0-12", "12-24", "24-48", "48-72", "72+"]
        df["TenureGroup"] = pd.cut(df["tenure"], bins=bins, labels=labels, right=False)

    # Charges per month feature
    if all(col in df.columns for col in ["TotalCharges", "tenure"]):
        df["ChargesPerMonth"] = df["TotalCharges"] / df["tenure"].replace({0: np.nan})
        df["ChargesPerMonth"].fillna(df["ChargesPerMonth"].median(), inplace=True)

    return df


def split_features_target(df: pd.DataFrame, target: str = "Churn") -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    # Remove known non-feature identifiers if present
    for col in ["customerID"]:
        if col in numeric_cols:
            numeric_cols.remove(col)
        if col in categorical_cols:
            categorical_cols.remove(col)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )
    return preprocessor


def train_models(df: pd.DataFrame, target: str = "Churn", random_state: int = 42) -> Dict[str, Dict]:
    df = clean_data(df)
    df = df.drop(columns=["customerID"], errors="ignore")
    X, y = split_features_target(df, target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    pre = build_preprocessor(X_train)

    # Logistic Regression
    log_reg = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=200)),
    ])
    log_reg.fit(X_train, y_train)
    y_pred_lr = log_reg.predict(X_test)
    y_proba_lr = log_reg.predict_proba(X_test)[:, 1]

    # Random Forest
    rf = Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(n_estimators=300, max_depth=None, random_state=random_state)),
    ])
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]

    results = {
        "log_reg": {
            "model": log_reg,
            "report": classification_report(y_test, y_pred_lr, output_dict=True),
            "auc": roc_auc_score(y_test, y_proba_lr),
        },
        "random_forest": {
            "model": rf,
            "report": classification_report(y_test, y_pred_rf, output_dict=True),
            "auc": roc_auc_score(y_test, y_proba_rf),
        },
    }
    return results


def kpis(df: pd.DataFrame) -> Dict[str, float]:
    df = clean_data(df)
    churn_rate = df["Churn"].mean() * 100.0
    retention_rate = 100.0 - churn_rate
    avg_tenure = float(df.get("tenure", pd.Series(dtype=float)).mean()) if "tenure" in df.columns else np.nan

    # High-risk segment heuristic: short tenure and month-to-month
    high_risk_mask = (
        (df.get("tenure", pd.Series(0)).astype(float) < 12)
        & (df.get("Contract", pd.Series("")) == "Month-to-month")
    )
    high_risk_share = high_risk_mask.mean() * 100.0

    # Simple CLV estimate: MonthlyCharges * avg tenure * margin (assume 60%)
    margin = 0.6
    if "MonthlyCharges" in df.columns and "tenure" in df.columns:
        clv_estimate = float((df["MonthlyCharges"] * df["tenure"] * margin).mean())
    else:
        clv_estimate = np.nan

    return {
        "churn_rate": float(churn_rate),
        "retention_rate": float(retention_rate),
        "avg_tenure": float(avg_tenure) if not np.isnan(avg_tenure) else np.nan,
        "high_risk_share": float(high_risk_share),
        "clv_estimate": float(clv_estimate) if not np.isnan(clv_estimate) else np.nan,
    }


def feature_importance_from_rf(pipeline: Pipeline, feature_names: pd.Index) -> pd.DataFrame:
    # Extract feature names post-transform
    pre: ColumnTransformer = pipeline.named_steps["pre"]
    num_features = pre.transformers_[0][2]
    cat_encoder: OneHotEncoder = pre.transformers_[1][1]
    cat_features = cat_encoder.get_feature_names_out(pre.transformers_[1][2])
    full_features = np.array(list(num_features) + list(cat_features))

    clf: RandomForestClassifier = pipeline.named_steps["clf"]
    importances = clf.feature_importances_
    imp_df = pd.DataFrame({"feature": full_features, "importance": importances}).sort_values(
        "importance", ascending=False
    )
    return imp_df
