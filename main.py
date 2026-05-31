import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List

from decision_rule import decide
from predict import load_forecaster

app = FastAPI()

# ─────────────────────────────────────────────
# 1) 기존 XGBoost 모델 (Buy / Wait 판단)
# ─────────────────────────────────────────────
model = joblib.load("xgb_regressor_best_0421.joblib")

FEATURE_COLUMNS = [
    "route_id", "departure_airport_code", "arrival_airport_code", "outbound_date",
    "searched_day_of_week", "days_to_departure",
    "is_weekend_search", "is_long_haul", "offer_count", "nonstop_ratio",
    "cheapest_nonstop_price", "cheapest_offer_has_layover", "current_cheapest_price",
    "curr_gap_to_typical_min", "curr_gap_to_typical_max",
    "hist_recent_std", "hist_recent_slope", "curr_vs_hist_mean",
    "price_change_1", "rolling_std_3", "price_vs_rolling_mean_3",
]


class FlightFeatureRequest(BaseModel):
    route_id: str
    departure_airport_code: str
    arrival_airport_code: str
    outbound_date: str
    searched_day_of_week: str
    days_to_departure: int
    is_weekend_search: bool
    is_long_haul: bool
    offer_count: int
    nonstop_ratio: float
    cheapest_nonstop_price: Optional[int] = None
    cheapest_offer_has_layover: bool
    current_cheapest_price: int
    curr_gap_to_typical_min: Optional[int] = None
    curr_gap_to_typical_max: Optional[int] = None
    hist_recent_std: Optional[float] = None
    hist_recent_slope: Optional[float] = None
    curr_vs_hist_mean: Optional[float] = None
    price_change_1: Optional[float] = None
    rolling_std_3: Optional[float] = None
    price_vs_rolling_mean_3: Optional[float] = None


@app.post("/predict")
def predict(request: FlightFeatureRequest):
    df = pd.DataFrame([request.model_dump()], columns=FEATURE_COLUMNS)
    predicted_drop = float(model.predict(df)[0])
    decision = decide(predicted_drop)
    return {"predictedDrop": predicted_drop, "decision": decision}


# ─────────────────────────────────────────────
# 2) ConformalForecaster (다구간 가격 예측)
# ─────────────────────────────────────────────
forecaster = load_forecaster("airmoment_forecast.joblib")


class ForecastRequest(BaseModel):
    # 필수
    route_id: str
    current_cheapest_price: int

    # 항공편 기본 정보
    days_to_departure: Optional[int] = None
    outbound_month: Optional[int] = Field(default=None, ge=1, le=12)
    searched_day_of_week: Optional[str] = None       # "MON" ~ "SUN"
    outbound_day_of_week: Optional[str] = None
    is_weekend_search: Optional[int] = None          # 0/1
    is_peak_season: Optional[int] = None
    is_holiday_near: Optional[int] = None
    is_long_haul: Optional[int] = None
    offer_count: Optional[int] = None
    nonstop_ratio: Optional[float] = None
    cheapest_nonstop_price: Optional[int] = None
    cheapest_offer_has_layover: Optional[int] = None

    # 가격 수준
    price_level: Optional[str] = None                # "low" / "typical" / "high"

    # 히스토리 기반
    curr_gap_to_typical_min: Optional[int] = None
    curr_gap_to_typical_max: Optional[int] = None
    hist_recent_std: Optional[float] = None
    hist_recent_slope: Optional[float] = None
    curr_vs_hist_mean: Optional[float] = None

    # 단기 시계열
    lag_1_price: Optional[int] = None
    price_change_1: Optional[int] = None
    rolling_std_3: Optional[float] = None
    price_vs_rolling_mean_3: Optional[int] = None

    # 외부 요인 (7d / 14d 모델만 사용)
    oil_price_usd: Optional[float] = None
    oil_change_7d: Optional[float] = None
    arr_fx_change_7d: Optional[float] = None


class ForecastResponse(BaseModel):
    current_price: int
    x: List[int]
    q10: List[int]
    q25: List[int]
    q50: List[int]
    q75: List[int]
    q90: List[int]


@app.post("/forecastPrice", response_model=ForecastResponse)
def forecast_price(request: ForecastRequest):
    try:
        return forecaster.forecast(request.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"forecast failed: {e}")
