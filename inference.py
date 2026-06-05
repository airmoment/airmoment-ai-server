from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer


PROJECT_ROOT = Path(__file__).resolve().parent
CLF_MODEL_PATH = PROJECT_ROOT / "catboost_classifier_best.joblib"
REG_MODEL_PATH = PROJECT_ROOT / "xgb_regressor_best.joblib"


# ---------------------------------------------------------------------------
# NativeCatBoostClassifier — 학습/추론 공통 클래스
# 이 파일에 정의해야 joblib 역직렬화 시 클래스를 찾을 수 있음
# ---------------------------------------------------------------------------

class NativeCatBoostClassifier:
    """CatBoost 네이티브 범주형 처리 래퍼.
    수치형은 median impute, 범주형은 문자열 그대로 CatBoost에 전달.
    sklearn Pipeline 없이 fit/predict_proba 인터페이스 제공.
    """
    def __init__(self, cat_cols: List[str], **catboost_params):
        self.cat_cols = cat_cols
        self.catboost_params = catboost_params
        self.num_imputer_ = None
        self.model_ = None
        self.num_cols_: List[str] = []
        self.feat_cols_: List[str] = []

    def _prepare(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        num_cols = [c for c in X.columns if c not in self.cat_cols]
        if fit:
            self.num_cols_ = num_cols
            self.num_imputer_ = SimpleImputer(strategy='median').fit(X[num_cols])
        # 학습 시 있었던 컬럼이 추론 시 없으면 NaN으로 채움
        X_num = X.reindex(columns=self.num_cols_)
        num_part = pd.DataFrame(
            self.num_imputer_.transform(X_num),
            columns=self.num_cols_,
        ).reset_index(drop=True)
        present_cats = [c for c in self.cat_cols if c in X.columns]
        cat_part = (
            X.reindex(columns=self.cat_cols)
            .fillna('__missing__')
            .astype(str)
            .reset_index(drop=True)
        )
        out = pd.concat([num_part, cat_part], axis=1)
        if fit:
            self.feat_cols_ = out.columns.tolist()
        return out

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        X_prep = self._prepare(X, fit=True)
        cat_indices = [X_prep.columns.tolist().index(c)
                       for c in self.cat_cols if c in X_prep.columns]
        self.model_ = CatBoostClassifier(cat_features=cat_indices, **self.catboost_params)
        self.model_.fit(X_prep, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(self._prepare(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self._prepare(X))

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model_.get_feature_importance()


# ---------------------------------------------------------------------------
# 모델 로드 / 추론
# ---------------------------------------------------------------------------

def load_model(
    clf_path: Path = CLF_MODEL_PATH,
    reg_path: Path = REG_MODEL_PATH,
) -> Dict[str, Any]:
    """
    Load the trained classification (CatBoost) and regression (XGBoost) pipelines.

    Returns:
        {'clf': NativeCatBoostClassifier, 'reg': pipeline}
    """
    if not clf_path.exists():
        raise FileNotFoundError(f"Classifier model not found: {clf_path}")
    if not reg_path.exists():
        raise FileNotFoundError(f"Regressor model not found: {reg_path}")

    return {
        'clf': joblib.load(clf_path),
        'reg': joblib.load(reg_path),
    }


def _horizon_index(days_to_departure: int) -> int:
    """days_to_departure 기준으로 conformal forecast horizon 인덱스 반환.
    forecast['q10'] = [현재, +1d, +3d, +7d, +14d] → 인덱스 0~4
    """
    if days_to_departure > 30:
        return 4   # +14d
    elif days_to_departure > 14:
        return 3   # +7d
    elif days_to_departure > 7:
        return 2   # +3d
    else:
        return 1   # +1d


def predict_flight_decision(
    feature_row: Dict[str, Any],
    model: Dict[str, Any] | None = None,
    forecaster: Any | None = None,
) -> Dict[str, Any]:
    """
    Run inference for a single feature row and return a map-like response.

    - BUY/WAIT decision       : CatBoost classifier
    - predicted_drop_amount   : conformal q10/q90 기반
        WAIT → current - q10[horizon]  (기다리면 최대 이만큼 아낄 수 있음)
        BUY  → q90[horizon] - current  (지금 안 사면 최대 이만큼 더 낼 수 있음)
    - predicted_future_price  : WAIT → q10, BUY → q50

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

    current_price = float(feature_row["current_cheapest_price"])
    days = int(feature_row.get("days_to_departure", 999))
    input_df = pd.DataFrame([feature_row])

    # 학습 시 사용한 컬럼 중 누락된 것은 NaN으로 채움
    clf = model['clf']
    reg = model['reg']
    reg_preprocessor = reg.named_steps['preprocessor']
    reg_expected = (
        list(reg_preprocessor.transformers[0][2]) +
        list(reg_preprocessor.transformers[1][2])
    )
    for col in set(clf.num_cols_ + clf.cat_cols + reg_expected):
        if col not in input_df.columns:
            input_df[col] = np.nan

    # BUY/WAIT from classifier
    wait_prob = float(model['clf'].predict_proba(input_df)[:, 1][0])
    decision = "WAIT" if wait_prob >= 0.5 else "BUY"

    # Override: 분류기가 WAIT이더라도 예측 절감액이 3% 미만이면 BUY
    MIN_WAIT_RATIO = 0.03

    # 절감액 계산: conformal 우선, 없으면 XGBoost fallback
    # 결정 보정: 회귀 예측 drop < 3%이면 WAIT 철회
    log_ratio = float(model['reg'].predict(input_df)[0])
    drop_ratio_pred = math.expm1(log_ratio)
    reg_drop_amount = max(0.0, current_price * drop_ratio_pred)
    if decision == "WAIT" and (reg_drop_amount / current_price) < MIN_WAIT_RATIO:
        decision = "BUY"

    # 절감액 계산: conformal 기반 (결정은 이미 위에서 확정)
    if forecaster is not None:
        try:
            fc  = forecaster.forecast(feature_row)
            idx = _horizon_index(days)
            q10 = float(fc['q10'][idx])
            q50 = float(fc['q50'][idx])
            q90 = float(fc['q90'][idx])

            if decision == "WAIT":
                # 기다리면 최대 이만큼 아낄 수 있음 (낙관적 하한 기준)
                predicted_drop_amount      = max(0.0, current_price - q10)
                predicted_future_min_price = q10
                # conformal도 유의미한 하락을 예측하지 않으면 BUY로 전환
                if predicted_drop_amount / current_price < MIN_WAIT_RATIO:
                    decision                   = "BUY"
                    predicted_drop_amount      = max(0.0, q90 - current_price)
                    predicted_future_min_price = q50
            else:
                # 지금 안 사면 최대 이만큼 더 낼 수 있음 (비관적 상한 기준)
                predicted_drop_amount      = max(0.0, q90 - current_price)
                predicted_future_min_price = q50

        except Exception:
            forecaster = None

    if forecaster is None:
        # fallback: XGBoost 기반
        if decision == "WAIT":
            predicted_drop_amount      = reg_drop_amount
            predicted_future_min_price = current_price - reg_drop_amount
        else:
            predicted_drop_amount      = 0.0
            predicted_future_min_price = current_price

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
