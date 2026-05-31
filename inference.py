from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
CLF_MODEL_PATH = PROJECT_ROOT / "catboost_classifier_best.joblib"
REG_MODEL_PATH = PROJECT_ROOT / "xgb_regressor_best.joblib"


def load_model(
    clf_path: Path = CLF_MODEL_PATH,
    reg_path: Path = REG_MODEL_PATH,
) -> Dict[str, Any]:
    """
    Load the trained classification (CatBoost) and regression (XGBoost) pipelines.

    Returns:
        {'clf': pipeline, 'reg': pipeline}
    """
    if not clf_path.exists():
        raise FileNotFoundError(f"Classifier model not found: {clf_path}")
    if not reg_path.exists():
        raise FileNotFoundError(f"Regressor model not found: {reg_path}")

    return {
        'clf': joblib.load(clf_path),
        'reg': joblib.load(reg_path),
    }


def predict_flight_decision(
    feature_row: Dict[str, Any],
    model: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Run inference for a single feature row and return a map-like response.

    - BUY/WAIT decision : CatBoost classifier (predict_proba threshold 0.5)
    - Drop amount       : XGBoost regressor (target_log_ratio → KRW via current price)

    Returns:
    {
        "decision": "BUY" or "WAIT",
        "predicted_drop_amount": float,
        "predicted_future_min_price": float
    }
    """
    if model is None:
        model = load_model()

    if "current_cheapest_price" not in feature_row:
        raise ValueError("feature_row must include 'current_cheapest_price'")

    input_df = pd.DataFrame([feature_row])

    # BUY/WAIT from classifier
    wait_prob = float(model['clf'].predict_proba(input_df)[:, 1][0])
    decision = "WAIT" if wait_prob >= 0.5 else "BUY"

    # Drop amount from regressor (target_log_ratio)
    log_ratio = float(model['reg'].predict(input_df)[0])
    current_price = float(feature_row["current_cheapest_price"])

    # log_ratio = log(future_price / current_price)
    # → future_price = current_price * exp(log_ratio)
    import math
    predicted_future_min_price = current_price * math.exp(log_ratio)
    predicted_drop_amount = max(0.0, current_price - predicted_future_min_price)

    return {
        "decision": decision,
        "predicted_drop_amount": predicted_drop_amount,
        "predicted_future_min_price": predicted_future_min_price,
    }


if __name__ == "__main__":
    sample_input = {
        "route_id": "ICN-NRT",
        "searched_day_of_week": "MON",
        "days_to_departure": 35,
        "is_weekend_search": 0,
        "is_long_haul": 0,
        "offer_count": 42,
        "nonstop_ratio": 0.26,
        "cheapest_nonstop_price": 355000,
        "cheapest_offer_has_layover": 0,
        "current_cheapest_price": 318000,
        "curr_gap_to_typical_min": -5000,
        "curr_gap_to_typical_max": -15000,
        "hist_recent_std": 12000.0,
        "hist_recent_slope": -3500.0,
        "curr_vs_hist_mean": 0.94,
        "price_change_1": -7000.0,
        "rolling_std_3": 6500.0,
        "price_vs_rolling_mean_3": -4333.0,
    }

    result = predict_flight_decision(sample_input)
    print(result)
