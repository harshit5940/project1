from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
from src.churn.utils import load_data, clean_data, kpis, train_models

app = FastAPI(title="Churn API", version="1.0.0")

# Enable CORS for frontend (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache data and models at startup
@app.on_event("startup")
def startup_cache():
    global DF, MODELS
    DF = clean_data(load_data())
    MODELS = train_models(DF)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.get("/kpis")
def get_kpis() -> Dict[str, float]:
    return kpis(DF)


@app.get("/aggregates/contract")
def churn_by_contract() -> Any:
    df = DF
    if "Contract" not in df.columns:
        return []
    agg = (
        df.groupby("Contract")["Churn"].mean().mul(100).reset_index()
        .rename(columns={"Churn": "churn_rate"})
    )
    return agg.to_dict(orient="records")


@app.get("/aggregates/payment_method")
def churn_by_payment_method() -> Any:
    df = DF
    if "PaymentMethod" not in df.columns:
        return []
    agg = (
        df.groupby("PaymentMethod")["Churn"].mean().mul(100).reset_index()
        .rename(columns={"Churn": "churn_rate"})
    )
    return agg.to_dict(orient="records")


class PredictRequest(BaseModel):
    # Accepts a subset of columns; missing columns will be filled with dataset medians/modes
    payload: Dict[str, Any]
    model: str = "random_forest"  # or "log_reg"


@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    model_key = req.model if req.model in MODELS else "random_forest"
    model = MODELS[model_key]["model"]

    # Build a single-row DataFrame aligning to DF columns
    input_df = pd.DataFrame([req.payload])

    # Align columns to training data
    base_cols = DF.drop(columns=["Churn"], errors="ignore").columns
    for col in base_cols:
        if col not in input_df.columns:
            # fill with typical values
            if pd.api.types.is_numeric_dtype(DF[col]):
                input_df[col] = float(DF[col].median())
            else:
                input_df[col] = DF[col].mode().iloc[0]

    # Remove extraneous columns not used in training
    input_df = input_df[base_cols]

    proba = float(model.predict_proba(input_df)[:, 1][0])
    pred = int(proba >= 0.5)

    return {"model": model_key, "churn_probability": proba, "prediction": pred}


# For local dev: uvicorn backend.main:app --reload --port 8000
