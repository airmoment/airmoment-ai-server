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
    forecast['q10'] = [현재(0), +1d(1), +3d(2), +7d(3), +14d(4)]

    D > 30  → +14d (D>60, D31~60 공통)
    D 15~30 → +7d
    D 8~14  → +3d
    D ≤ 7   → +1d
    """
    if days_to_departure > 30:
        return 4   # +14d
    elif days_to_departure > 14:
        return 3   # +7d
    elif days_to_departure > 7:
        return 2   # +3d
    else:
        return 1   # +1d


MIN_WAIT_RATIO = 0.03          # target_wait 레이블 정의(≥3% 하락)와 일치
CLF_THRESHOLD_STD  = 0.5      # D > 30 구간 (학습 범위 내)
CLF_THRESHOLD_CONS = 0.6      # D ≤ 30 구간 (OOD, 보수적 WAIT)


def predict_flight_decision(
    feature_row: Dict[str, Any],
    model: Dict[str, Any] | None = None,
    forecaster: Any | None = None,
) -> Dict[str, Any]:
    """
    Run inference for a single feature row.

    Decision logic (구간별 단일 조건, override 없음):

      D > 60  : clf ≥ 0.5  AND  reg_drop ≥ 3%
                → WAIT 절감액 = reg 기반 / BUY 절감액 = conformal q90

      D 31~60 : clf ≥ 0.5  AND  conf_drop[+14d] ≥ 3%
                → WAIT 절감액 = current - q10[+14d]

      D 15~30 : conf_drop[+7d] ≥ 3%  AND  clf ≥ 0.6
                → WAIT 절감액 = current - q10[+7d]

      D 8~14  : conf_drop[+3d] ≥ 3%  AND  clf ≥ 0.6
                → WAIT 절감액 = current - q10[+3d]

      D ≤ 7   : conf_drop[+1d] ≥ 3%  AND  clf ≥ 0.6
                → WAIT 절감액 = current - q10[+1d]

    절감액 의미:
      WAIT → 기다리면 최대 이만큼 아낄 수 있음 (낙관적 하한 q10 기준)
      BUY  → 지금 안 사면 최대 이만큼 더 낼 수 있음 (비관적 상한 q90 기준)

    Returns:
      { "decision", "predicted_drop_amount", "predicted_future_min_price" }
    """
    if model is None:
        model = load_model()

    if "current_cheapest_price" not in feature_row:
        raise ValueError("feature_row must include 'current_cheapest_price'")

    current_price = float(feature_row["current_cheapest_price"])
    days          = int(feature_row.get("days_to_departure", 999))

    # ── 입력 준비 ──────────────────────────────────────────────────────────
    clf = model['clf']
    reg = model['reg']
    reg_expected = (
        list(reg.named_steps['preprocessor'].transformers[0][2]) +
        list(reg.named_steps['preprocessor'].transformers[1][2])
    )
    input_df = pd.DataFrame([feature_row])
    for col in set(clf.num_cols_ + clf.cat_cols + reg_expected):
        if col not in input_df.columns:
            input_df[col] = np.nan

    # ── 공통 신호 ──────────────────────────────────────────────────────────
    wait_prob = float(clf.predict_proba(input_df)[:, 1][0])
    reg_drop  = max(0.0, math.expm1(float(reg.predict(input_df)[0])))

    # ── Conformal 신호 ────────────────────────────────────────────────────
    fc_ok = False
    if forecaster is not None:
        try:
            fc  = forecaster.forecast(feature_row)
            idx = _horizon_index(days)
            q10 = float(fc['q10'][idx])
            q50 = float(fc['q50'][idx])
            q90 = float(fc['q90'][idx])
            conf_drop = (current_price - q50) / current_price
            fc_ok = True
        except Exception:
            pass

    # ── 구간별 결정 (단일 조건, override 없음) ────────────────────────────
    if days > 60:
        # clf + reg: conformal 14d 창이 남은 기간 대비 너무 짧아 신뢰도 낮음
        wait = (wait_prob >= CLF_THRESHOLD_STD) and (reg_drop >= MIN_WAIT_RATIO)
        if wait:
            amount = reg_drop * current_price
            future = current_price - amount
        else:
            amount = max(0.0, q90 - current_price) if fc_ok else 0.0
            future = max(q50, current_price) if fc_ok else current_price

    elif days > 30:
        # clf AND conformal +14d: 두 모델 동시 합의
        if fc_ok:
            wait = (wait_prob >= CLF_THRESHOLD_STD) and (conf_drop >= MIN_WAIT_RATIO)
        else:
            wait = (wait_prob >= CLF_THRESHOLD_STD) and (reg_drop >= MIN_WAIT_RATIO)
        if wait:
            amount = max(0.0, current_price - q10) if fc_ok else reg_drop * current_price
            future = q10 if fc_ok else current_price - reg_drop * current_price
        else:
            amount = max(0.0, q90 - current_price) if fc_ok else 0.0
            future = max(q50, current_price) if fc_ok else current_price

    else:
        # conformal 주도 + clf 보조 (보수적 threshold 0.6)
        if fc_ok:
            wait = (conf_drop >= MIN_WAIT_RATIO) and (wait_prob >= CLF_THRESHOLD_CONS)
        else:
            wait = (wait_prob >= CLF_THRESHOLD_CONS) and (reg_drop >= MIN_WAIT_RATIO)
        if wait:
            amount = max(0.0, current_price - q10) if fc_ok else reg_drop * current_price
            future = q10 if fc_ok else current_price - reg_drop * current_price
        else:
            amount = max(0.0, q90 - current_price) if fc_ok else 0.0
            future = max(q50, current_price) if fc_ok else current_price

    return {
        "decision":                  "WAIT" if wait else "BUY",
        "predicted_drop_amount":     amount,
        "predicted_future_min_price": future,
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
