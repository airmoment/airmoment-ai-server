"""predict.py

Loads trained quantile models and generates probabilistic price forecasts
for a single search event.

API-facing function:
    forecast(features: dict, model_dir: str | Path) -> dict

Output JSON structure:
    {
        "current_price": 510000,
        "forecasts": [
            {
                "horizon_days": 1,
                "q10": 485000, "q25": 498000, "q50": 511000,
                "q75": 528000, "q90": 547000
            },
            { "horizon_days": 3, ... },
            { "horizon_days": 7, ... },
            { "horizon_days": 14, ... }
        ]
    }

q50  → most likely price (center line on chart)
q25–q75 → 50% confidence band (dark band)
q10–q90 → 80% confidence band (light band)
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config (must match train_quantile_model.py)
# ---------------------------------------------------------------------------

HORIZONS  = [1, 3, 7, 14]
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

NUMERIC_COLS = [
    'days_to_departure', 'is_weekend_search', 'outbound_month',
    'is_peak_season', 'is_holiday_near', 'is_long_haul',
    'offer_count', 'nonstop_ratio', 'cheapest_nonstop_price',
    'cheapest_offer_has_layover', 'current_cheapest_price',
    'curr_gap_to_typical_min', 'curr_gap_to_typical_max',
    'hist_recent_std', 'hist_recent_slope', 'curr_vs_hist_mean',
    'lag_1_price', 'price_change_1', 'rolling_std_3', 'price_vs_rolling_mean_3',
]
CAT_COLS = ['route_id', 'searched_day_of_week', 'outbound_day_of_week', 'price_level']


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

class QuantileForecaster:
    """Loads and holds all trained models for inference."""

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        self.models: dict[tuple[int, float], object]  = {}
        self.imputers: dict[int, object]              = {}
        self.encoders: dict[str, dict[str, int]]      = {}
        self.feature_cols: list[str]                  = []
        self._load()

    def _load(self) -> None:
        enc_path = self.model_dir / 'encoders.json'
        meta_path = self.model_dir / 'run_metadata.json'

        if not enc_path.exists():
            raise FileNotFoundError(f'encoders.json not found in {self.model_dir}')
        with open(enc_path, encoding='utf-8') as f:
            self.encoders = json.load(f)
        if meta_path.exists():
            with open(meta_path, encoding='utf-8') as f:
                meta = json.load(f)
            self.feature_cols = meta.get('feature_cols', [])

        if not self.feature_cols:
            # Reconstruct if metadata missing
            self.feature_cols = NUMERIC_COLS + [f'{c}_enc' for c in self.encoders]

        for h in HORIZONS:
            imp_path = self.model_dir / f'imputer_{h}d.joblib'
            if imp_path.exists():
                self.imputers[h] = joblib.load(imp_path)

            for q in QUANTILES:
                q_tag = f'q{int(q * 100):02d}'
                m_path = self.model_dir / f'lgbm_{h}d_{q_tag}.joblib'
                if m_path.exists():
                    self.models[(h, q)] = joblib.load(m_path)

        loaded_models = len(self.models)
        expected      = len(HORIZONS) * len(QUANTILES)
        print(f'[QuantileForecaster] Loaded {loaded_models}/{expected} models from {self.model_dir}')

    # ------------------------------------------------------------------

    def _encode_features(self, features: dict) -> pd.DataFrame:
        """Convert a raw feature dict into the encoded DataFrame the models expect."""
        row = {col: features.get(col, np.nan) for col in NUMERIC_COLS}

        for cat_col, mapping in self.encoders.items():
            raw_val    = str(features.get(cat_col, '__missing__') or '__missing__')
            row[f'{cat_col}_enc'] = float(mapping.get(raw_val, mapping.get('__missing__', 0)))

        return pd.DataFrame([row])[self.feature_cols]

    def forecast(self, features: dict) -> dict:
        """Generate probabilistic forecasts for all horizons.

        Args:
            features: dict with keys matching NUMERIC_COLS + CAT_COLS
                      (any missing values are imputed with training medians)

        Returns:
            dict with 'current_price' and 'forecasts' list
        """
        df_row = self._encode_features(features)

        result_forecasts = []
        for h in HORIZONS:
            imputer = self.imputers.get(h)
            if imputer is None:
                # Fallback: use raw values (imputer missing for this horizon)
                X = df_row.astype(float).fillna(0).to_numpy()
            else:
                X = imputer.transform(df_row.astype(float))

            q_preds: dict[str, int] = {}
            available = True
            for q in QUANTILES:
                model = self.models.get((h, q))
                if model is None:
                    available = False
                    break
                raw = float(model.predict(X)[0])
                q_preds[f'q{int(q * 100):02d}'] = int(round(raw))

            if not available:
                continue

            # Enforce monotonicity: q10 ≤ q25 ≤ q50 ≤ q75 ≤ q90
            vals = [q_preds[f'q{int(q * 100):02d}'] for q in QUANTILES]
            vals = sorted(vals)
            for i, q in enumerate(QUANTILES):
                q_preds[f'q{int(q * 100):02d}'] = vals[i]

            result_forecasts.append({'horizon_days': h, **q_preds})

        return {
            'current_price': int(round(features.get('current_cheapest_price', 0))),
            'forecasts':     result_forecasts,
        }


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

_default_forecaster: QuantileForecaster | None = None


def load_forecaster(model_dir: str | Path) -> QuantileForecaster:
    """Load a QuantileForecaster and cache it as the module default."""
    global _default_forecaster
    _default_forecaster = QuantileForecaster(model_dir)
    return _default_forecaster


def forecast(features: dict, model_dir: str | Path | None = None) -> dict:
    """One-call inference. Loads models on first call; reuses cache after.

    Args:
        features:  dict of feature values for one search event
        model_dir: path to models/ folder (required on first call)

    Returns:
        forecast dict (JSON-serialisable)
    """
    global _default_forecaster
    if _default_forecaster is None:
        if model_dir is None:
            model_dir = Path(__file__).resolve().parent / 'models'
        _default_forecaster = QuantileForecaster(model_dir)
    return _default_forecaster.forecast(features)


# ---------------------------------------------------------------------------
# Example usage / smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    MODEL_DIR = Path(__file__).resolve().parent / 'models'

    if not (MODEL_DIR / 'encoders.json').exists():
        print('No trained models found. Run train_quantile_model.py first.')
    else:
        # Example feature dict — replace with actual values from the DB
        example_features = {
            # Static / route
            'route_id':                 'ICN-CDG',
            'days_to_departure':        21,
            'searched_day_of_week':     'MON',
            'outbound_month':           7,
            'outbound_day_of_week':     'FRI',
            'is_weekend_search':        0,
            'is_peak_season':           1,
            'is_holiday_near':          0,
            'is_long_haul':             1,
            # Offer structure
            'offer_count':              8,
            'nonstop_ratio':            0.25,
            'cheapest_nonstop_price':   890000,
            'cheapest_offer_has_layover': 1,
            # Price positioning
            'current_cheapest_price':   510000,
            'curr_gap_to_typical_min':  75000,
            'curr_gap_to_typical_max':  -230000,
            'price_level':              'typical',
            # History features
            'hist_recent_std':          8500,
            'hist_recent_slope':        -1200,
            'curr_vs_hist_mean':        0.97,
            # Lag / dynamics
            'lag_1_price':              522000,
            'price_change_1':           -12000,
            'rolling_std_3':            9800,
            'price_vs_rolling_mean_3':  -6000,
        }

        result = forecast(example_features, MODEL_DIR)

        print('\n=== Probabilistic Price Forecast ===')
        print(f"Current price: ₩{result['current_price']:,}")
        print()
        for fc in result['forecasts']:
            h = fc['horizon_days']
            print(f"  +{h:>2d}d │ "
                  f"q10=₩{fc['q10']:,}  "
                  f"q25=₩{fc['q25']:,}  "
                  f"q50=₩{fc['q50']:,}  "
                  f"q75=₩{fc['q75']:,}  "
                  f"q90=₩{fc['q90']:,}")

        print()
        print('JSON output:')
        print(json.dumps(result, ensure_ascii=False, indent=2))
