import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Literal

from inference import load_model, predict_flight_decision
from predict import load_forecaster
from explain import explain_forecast

app = FastAPI()

# 서버 시작 시 모델 1회 로드
_xgb_model  = load_model()
_forecaster = load_forecaster("airmoment_forecast.joblib")


# ---------------------------------------------------------------------------
# /predict  —  buy/wait 판단 (XGBoost)
# ---------------------------------------------------------------------------

class FlightFeatureRequest(BaseModel):
    # 노선 / 검색 기본 정보
    route_id: str
    searched_day_of_week: str
    days_to_departure: int
    is_weekend_search: bool
    is_long_haul: bool

    # 출발편 정보 (outbound_date로부터 생성)
    outbound_month: int = Field(ge=1, le=12)
    outbound_day_of_week: str
    is_peak_season: bool
    is_holiday_near: bool

    # 실시간 검색 결과
    offer_count: int
    nonstop_ratio: float
    cheapest_nonstop_price: Optional[int] = None
    cheapest_offer_has_layover: bool
    current_cheapest_price: int
    curr_gap_to_typical_min: Optional[int] = None
    curr_gap_to_typical_max: Optional[int] = None

    # 과거 관측값 기반
    hist_recent_std: Optional[float] = None
    hist_recent_slope: Optional[float] = None
    curr_vs_hist_mean: Optional[float] = None
    price_change_1: Optional[float] = None
    rolling_std_3: Optional[float] = None
    price_vs_rolling_mean_3: Optional[float] = None


@app.post("/predict")
def predict(request: FlightFeatureRequest):
    """
    항공권 buy/wait 판단.

    Response:
        decision               : "BUY" | "WAIT"
        predicted_drop_amount  :
            WAIT → 기다리면 최대 절감 가능 금액 (current - conformal q10)
            BUY  → 지금 안 사면 최대 추가 부담 금액 (conformal q90 - current)
        predicted_future_min_price :
            WAIT → conformal q10 (낙관적 미래 최저가)
            BUY  → conformal q50 (예상 미래 가격)
    """
    result = predict_flight_decision(
        feature_row=request.model_dump(),
        model=_xgb_model,
        forecaster=_forecaster,
    )
    return result


# ---------------------------------------------------------------------------
# /forecastPrice  —  다구간 가격 추이 예측 (Conformal)
# ---------------------------------------------------------------------------

class ForecastRequest(BaseModel):
    # 필수
    route_id: str
    current_cheapest_price: int

    # 항공편 기본 정보
    days_to_departure: Optional[int] = None
    outbound_month: Optional[int] = Field(default=None, ge=1, le=12)
    searched_day_of_week: Optional[str] = None
    outbound_day_of_week: Optional[str] = None
    is_weekend_search: Optional[int] = None
    is_peak_season: Optional[int] = None
    is_holiday_near: Optional[int] = None
    is_long_haul: Optional[int] = None
    offer_count: Optional[int] = None
    nonstop_ratio: Optional[float] = None
    cheapest_nonstop_price: Optional[int] = None
    cheapest_offer_has_layover: Optional[int] = None

    # 가격 수준
    price_level: Optional[str] = None

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
    """
    다구간 항공권 가격 추이 예측 (Split Conformal).

    Response:
        current_price : 현재 최저가 (₩)
        x             : 시간축 [0, 1, 3, 7, 14] (일)
        q10~q90       : 각 시점별 예측 구간 (길이 5 배열, ₩ 정수)
    """
    try:
        return _forecaster.forecast(request.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"forecast failed: {e}")


# ---------------------------------------------------------------------------
# /explain  —  예측 근거 설명 (SHAP)
# ---------------------------------------------------------------------------

class ExplainRequest(BaseModel):

    # 필수
    route_id: str
    current_cheapest_price: int

    # 항공편 기본 정보
    days_to_departure: Optional[int] = None
    outbound_month: Optional[int] = Field(default=None, ge=1, le=12)
    searched_day_of_week: Optional[str] = None
    outbound_day_of_week: Optional[str] = None
    is_weekend_search: Optional[int] = None
    is_peak_season: Optional[int] = None
    is_holiday_near: Optional[int] = None
    is_long_haul: Optional[int] = None
    offer_count: Optional[int] = None
    nonstop_ratio: Optional[float] = None
    cheapest_nonstop_price: Optional[int] = None
    cheapest_offer_has_layover: Optional[int] = None

    # 가격 수준
    price_level: Optional[str] = None

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


class ExplainResponse(BaseModel):
    direction: Literal["up", "down"]
    direction_amount: int
    reasons: List[str]


@app.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest):
    """
    BUY/WAIT 판단 근거 설명 (SHAP 기반 한국어 문장).

    /predict와 동일한 로직(predict_flight_decision)으로 먼저 BUY/WAIT을 결정하고,
    그 결정과 align된 SHAP 근거 문장을 반환한다.

    Response:
        direction        : 'down'(WAIT, 가격 하락 예상) | 'up'(BUY, 가격 상승/유지 예상)
        direction_amount : 절감/추가부담 예상액 (KRW)
        reasons          : 근거 문장 목록
    """
    features = request.model_dump()
    try:
        decision = predict_flight_decision(features, _xgb_model, forecaster=_forecaster)
        return explain_forecast(
            features,
            clf=_xgb_model['clf'],
            forecaster=_forecaster,
            is_wait=(decision['decision'] == 'WAIT'),
            drop_amount=decision['predicted_drop_amount'],
            top_n=3,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"explain failed: {e}")
