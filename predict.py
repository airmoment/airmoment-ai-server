"""predict.py

Split Conformal inference for multi-horizon flight price forecasting.
Loads a single bundle file: airmoment_model.joblib

API:
    from predict import load_forecaster, forecast

    forecaster = load_forecaster('models/airmoment_model.joblib')
    result = forecaster.forecast(features)

Output dict:
    {
        'current_price': 510000,
        'x':   [0, 1, 3,  7,  14],
        'q10': [510000, ...],
        'q25': [510000, ...],
        'q50': [510000, ...],
        'q75': [510000, ...],
        'q90': [510000, ...],
    }
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


HORIZONS = [1, 3, 7, 14]

NUMERIC_COLS = [
    'days_to_departure', 'is_weekend_search', 'outbound_month',
    'is_peak_season', 'is_holiday_near', 'is_long_haul',
    'offer_count', 'nonstop_ratio', 'cheapest_nonstop_price',
    'cheapest_offer_has_layover', 'current_cheapest_price',
    'curr_gap_to_typical_min', 'curr_gap_to_typical_max',
    'hist_recent_std', 'hist_recent_slope', 'curr_vs_hist_mean',
    'lag_1_price', 'price_change_1', 'rolling_std_3', 'price_vs_rolling_mean_3',
    'oil_price_usd', 'oil_change_7d', 'arr_fx_change_7d',
]


# ---------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------

class ConformalForecaster:

    def __init__(self, model_path: str | Path) -> None:
        bundle = joblib.load(model_path)
        self.models:       dict[int, object]              = bundle['models']
        self.conf_corr:    dict                           = bundle['conf']
        self.encoders:     dict[str, dict[str, int]]      = bundle['encoders']
        self.feature_cols: dict[int, list[str]]           = bundle['feature_cols']
        print(f'[ConformalForecaster] {len(self.models)}/{len(HORIZONS)} models loaded')

    def _encode(self, features: dict, feat_cols: list[str]) -> np.ndarray:
        row = {col: features.get(col, np.nan) for col in NUMERIC_COLS}
        for cat, mapping in self.encoders.items():
            raw = str(features.get(cat, '__missing__') or '__missing__')
            row[f'{cat}_enc'] = float(mapping.get(raw, mapping.get('__missing__', 0)))
        return pd.DataFrame([row])[feat_cols].astype(float).to_numpy()

    def forecast(self, features: dict) -> dict:
        result_forecasts = []

        for h in HORIZONS:
            model = self.models.get(h)
            if model is None:
                continue

            feat_cols = self.feature_cols[h]
            X   = self._encode(features, feat_cols)
            q50 = float(model.predict(X)[0])

            corr   = self.conf_corr.get(f'{h}d', {})
            qhat80 = corr.get('80pct', 0.0)
            qhat50 = corr.get('50pct', 0.0)

            vals = sorted([q50 - qhat80, q50 - qhat50, q50, q50 + qhat50, q50 + qhat80])
            q10, q25, q50_out, q75, q90 = (int(round(v)) for v in vals)

            result_forecasts.append((h, q10, q25, q50_out, q75, q90))

        current_price = int(round(features.get('current_cheapest_price', 0)))
        return {
            'current_price': current_price,
            'x':   [0]             + [h                        for h, *_          in result_forecasts],
            'q10': [current_price] + [q10 for _, q10, *_       in result_forecasts],
            'q25': [current_price] + [q25 for _, _, q25, *_    in result_forecasts],
            'q50': [current_price] + [q50 for _, _, _, q50, _, _ in result_forecasts],
            'q75': [current_price] + [q75 for _, _, _, _, q75, _ in result_forecasts],
            'q90': [current_price] + [q90 for _, _, _, _, _, q90 in result_forecasts],
        }


# ---------------------------------------------------------------------------
# forecast_with_reasons ← 백엔드 단일 호출용
# ---------------------------------------------------------------------------

def forecast_with_reasons(
    features: dict,
    forecaster: ConformalForecaster,
    model: dict,
    top_n: int = 3,
) -> dict:
    """예측값(conformal bands) + CatBoost/LightGBM SHAP 판단 근거를 한 번에 반환.

    Parameters
    ----------
    model : inference.load_model() 반환값 {'clf': ..., 'reg': ...}

    Returns
    -------
    {
        'current_price': 510000,
        'x':   [0, 1, 3, 7, 14],
        'q10': [...], 'q25': [...], 'q50': [...], 'q75': [...], 'q90': [...],
        'explanation': {
            'direction':        'down',
            'direction_amount': 45000,
            'reasons':          ['문장1', '문장2', '문장3']
        }
    }
    """
    from explain import explain_forecast
    from inference import predict_flight_decision

    base     = forecaster.forecast(features)
    decision = predict_flight_decision(features, model, forecaster=forecaster)

    exp = explain_forecast(
        features,
        clf=model['clf'],
        forecaster=forecaster,
        is_wait=(decision['decision'] == 'WAIT'),
        drop_amount=decision['predicted_drop_amount'],
        top_n=top_n,
    )

    return {**base, 'explanation': exp}


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_forecaster: ConformalForecaster | None = None


def load_forecaster(model_path: str | Path) -> ConformalForecaster:
    global _default_forecaster
    _default_forecaster = ConformalForecaster(model_path)
    return _default_forecaster


def forecast(features: dict, model_path: str | Path | None = None) -> dict:
    global _default_forecaster
    if _default_forecaster is None:
        if model_path is None:
            model_path = Path(__file__).resolve().parent / 'data_0526' / 'models' / 'airmoment_model.joblib'
        _default_forecaster = ConformalForecaster(model_path)
    return _default_forecaster.forecast(features)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    MODEL_PATH = Path(__file__).resolve().parent / 'data_0526' / 'models' / 'airmoment_model.joblib'

    if not MODEL_PATH.exists():
        print('No model found. Run train_cqr_model.py first.')
    else:
        example = {
            'route_id': 'ICN-CDG', 'days_to_departure': 21,
            'searched_day_of_week': 'MON', 'outbound_month': 7,
            'outbound_day_of_week': 'FRI', 'is_weekend_search': 0,
            'is_peak_season': 1, 'is_holiday_near': 0, 'is_long_haul': 1,
            'offer_count': 8, 'nonstop_ratio': 0.25,
            'cheapest_nonstop_price': 890000, 'cheapest_offer_has_layover': 1,
            'current_cheapest_price': 510000, 'curr_gap_to_typical_min': 75000,
            'curr_gap_to_typical_max': -230000, 'price_level': 'typical',
            'hist_recent_std': 8500, 'hist_recent_slope': -1200,
            'curr_vs_hist_mean': 0.97, 'lag_1_price': 522000,
            'price_change_1': -12000, 'rolling_std_3': 9800,
            'price_vs_rolling_mean_3': -6000, 'oil_price_usd': 78.5,
            'oil_change_7d': -1.2, 'arr_fx_change_7d': 0.003,
        }

        forecaster = load_forecaster(MODEL_PATH)
        result = forecaster.forecast(example)

        print(f"\nCurrent: ₩{result['current_price']:,}")
        for i, h in enumerate(result['x'][1:], 1):
            print(
                f"  +{h:>2d}d │ "
                f"[₩{result['q10'][i]:,} — ₩{result['q25'][i]:,} — ₩{result['q50'][i]:,}"
                f" — ₩{result['q75'][i]:,} — ₩{result['q90'][i]:,}]"
            )
