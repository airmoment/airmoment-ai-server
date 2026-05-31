from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "xgb_regressor_best.joblib"

# Prototype decision rule
WAIT_THRESHOLD = 15000.0


def load_model(model_path: Path = MODEL_PATH):
    """
    Load the trained regression pipeline.
    This joblib is expected to contain both preprocessing and model.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def predict_flight_decision(
    feature_row: Dict[str, Any],
    model=None,
    wait_threshold: float = WAIT_THRESHOLD,
) -> Dict[str, Any]:
    """
    Run inference for a single feature row and return a map-like response.

    Required:
    - feature_row must contain the exact model input fields expected by the trained pipeline.
    - feature_row must include current_cheapest_price so predicted_future_min_price can be calculated.

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

    predicted_drop_amount = float(model.predict(input_df)[0])

    if predicted_drop_amount < 0:
        predicted_drop_amount = 0.0

    current_price = float(feature_row["current_cheapest_price"])
    predicted_future_min_price = current_price - predicted_drop_amount

    decision = "WAIT" if predicted_drop_amount >= wait_threshold else "BUY"

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
