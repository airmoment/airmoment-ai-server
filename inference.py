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

    # 학습 시 사용한 컬럼 중 누락된 것은 NaN으로 채움 (optional 피처 처리)
    clf = model['clf']
    reg = model['reg']
    reg_preprocessor = reg.named_steps['preprocessor']
    reg_expected = (
        list(reg_preprocessor.transformers[0][2]) +  # numeric cols
        list(reg_preprocessor.transformers[1][2])    # categorical cols
    )
    for col in set(clf.num_cols_ + clf.cat_cols + reg_expected):
        if col not in input_df.columns:
            input_df[col] = np.nan

    # BUY/WAIT from classifier
    wait_prob = float(model['clf'].predict_proba(input_df)[:, 1][0])
    decision = "WAIT" if wait_prob >= 0.5 else "BUY"

    # Drop amount from regressor (target_log_ratio)
    log_ratio = float(model['reg'].predict(input_df)[0])
    current_price = float(feature_row["current_cheapest_price"])

    # log_ratio = log(future_price / current_price)
    # → future_price = current_price * exp(log_ratio)
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
