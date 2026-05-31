"""train_quantile_model.py

Trains per-horizon LightGBM quantile regression models with walk-forward CV.

For each of the 4 horizons (1, 3, 7, 14 days) and 5 quantiles (q10–q90):
  → One LightGBM model trained on all data, evaluated via walk-forward CV.

Walk-forward CV metrics per horizon:
  - q50 MAE / MAPE           (point-estimate accuracy)
  - 50% coverage (q25–q75)   (target ≈ 50%)
  - 80% coverage (q10–q90)   (target ≈ 80%)
  - Mean interval width       (narrower = more precise)

Saved outputs (models/):
  lgbm_{h}d_q{q}.joblib    — 20 trained models
  encoders.json              — categorical label encodings
  run_metadata.json          — feature list, config

Saved outputs (outputs/):
  fold_metrics_{h}d.csv      — per-fold metrics per horizon
  summary_metrics.csv        — mean metrics across folds, all horizons
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.impute import SimpleImputer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILE    = PROJECT_ROOT / 'data_0518' / 'processed' / 'quantile_dataset.csv'
MODEL_DIR    = PROJECT_ROOT / 'data_0518' / 'models'
OUTPUT_DIR   = PROJECT_ROOT / 'data_0518' / 'outputs'

HORIZONS     = [1, 3, 7, 14]
QUANTILES    = [0.10, 0.25, 0.50, 0.75, 0.90]

N_SPLITS      = 4
TEST_FRACTION = 0.2
MIN_TRAIN     = 20       # minimum rows in training set per fold
RANDOM_STATE  = 42

# Categorical features → label-encoded
CAT_COLS = ['route_id', 'searched_day_of_week', 'outbound_day_of_week', 'price_level']

# Numeric features
NUMERIC_COLS = [
    'days_to_departure', 'is_weekend_search', 'outbound_month',
    'is_peak_season', 'is_holiday_near', 'is_long_haul',
    'offer_count', 'nonstop_ratio', 'cheapest_nonstop_price',
    'cheapest_offer_has_layover', 'current_cheapest_price',
    'curr_gap_to_typical_min', 'curr_gap_to_typical_max',
    'hist_recent_std', 'hist_recent_slope', 'curr_vs_hist_mean',
    'lag_1_price', 'price_change_1', 'rolling_std_3', 'price_vs_rolling_mean_3',
]

LGBM_BASE_PARAMS = dict(
    n_estimators     = 400,
    max_depth        = 4,
    num_leaves       = 15,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_samples= 5,
    reg_alpha        = 0.5,
    reg_lambda       = 1.0,
    random_state     = RANDOM_STATE,
    verbose          = -1,
)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def build_encoders(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Label-encode each categorical column. Returns {col: {value: int}} mapping."""
    encoders: dict[str, dict[str, int]] = {}
    for col in CAT_COLS:
        if col not in df.columns:
            continue
        vals   = df[col].fillna('__missing__').astype(str).unique().tolist()
        vocab  = sorted(vals)
        encoders[col] = {v: i for i, v in enumerate(vocab)}
    return encoders


def apply_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in encoders.items():
        if col not in out.columns:
            continue
        out[f'{col}_enc'] = out[col].fillna('__missing__').astype(str).map(mapping).astype(float)
    return out


def get_feature_cols(encoders: dict) -> list[str]:
    num  = [c for c in NUMERIC_COLS]
    cats = [f'{c}_enc' for c in encoders]
    return num + cats


def prepare_X(df: pd.DataFrame, feature_cols: list[str], imputer: SimpleImputer | None = None):
    X = df[feature_cols].astype(float)
    if imputer is None:
        imputer = SimpleImputer(strategy='median')
        X_out = imputer.fit_transform(X)
    else:
        X_out = imputer.transform(X)
    return X_out, imputer


# ---------------------------------------------------------------------------
# Walk-forward splits
# ---------------------------------------------------------------------------

def make_splits(n: int, n_splits: int, test_frac: float, min_train: int):
    test_size = max(5, int(np.floor(n * test_frac)))
    max_splits = max(1, (n - min_train) // test_size)
    n_actual   = min(n_splits, max_splits)
    splits = []
    for i in range(n_actual):
        train_end = min_train + i * test_size
        test_end  = min(train_end + test_size, n)
        if test_end > train_end:
            splits.append((np.arange(0, train_end), np.arange(train_end, test_end)))
    return splits


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def coverage(y_true: np.ndarray, y_lo: np.ndarray, y_hi: np.ndarray) -> float:
    return float(np.mean((y_true >= y_lo) & (y_true <= y_hi)))


def mean_interval_width(y_lo: np.ndarray, y_hi: np.ndarray) -> float:
    return float(np.mean(y_hi - y_lo))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else np.nan


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_horizon(
    df: pd.DataFrame,
    horizon: int,
    feature_cols: list[str],
    encoders: dict,
) -> tuple[dict[float, lgb.LGBMRegressor], list[dict]]:
    """Train all quantile models for one horizon. Returns (models, fold_metrics)."""

    target_col = f'price_{horizon}d'
    valid_df   = df[df[target_col].notna()].sort_values('searched_at').reset_index(drop=True)

    if len(valid_df) < MIN_TRAIN + 5:
        print(f'  [SKIP horizon={horizon}d] Only {len(valid_df)} valid samples.')
        return {}, []

    splits = make_splits(len(valid_df), N_SPLITS, TEST_FRACTION, MIN_TRAIN)
    print(f'  horizon={horizon}d | {len(valid_df)} samples | {len(splits)} folds')

    fold_records: list[dict] = []

    for fold_no, (tr_idx, te_idx) in enumerate(splits, 1):
        tr_df = valid_df.iloc[tr_idx]
        te_df = valid_df.iloc[te_idx]

        y_tr  = tr_df[target_col].to_numpy(float)
        y_te  = te_df[target_col].to_numpy(float)

        X_tr, imputer = prepare_X(tr_df, feature_cols)
        X_te, _       = prepare_X(te_df, feature_cols, imputer)

        fold_preds: dict[float, np.ndarray] = {}
        for q in QUANTILES:
            m = lgb.LGBMRegressor(objective='quantile', alpha=q, **LGBM_BASE_PARAMS)
            m.fit(X_tr, y_tr)
            fold_preds[q] = m.predict(X_te)

        q10, q25, q50, q75, q90 = (fold_preds[q] for q in QUANTILES)

        rec = {
            'horizon_days':    horizon,
            'fold':            fold_no,
            'train_rows':      len(tr_idx),
            'test_rows':       len(te_idx),
            'q50_mae':         mae(y_te, q50),
            'q50_mape':        mape(y_te, q50),
            'coverage_50pct':  coverage(y_te, q25, q75),
            'coverage_80pct':  coverage(y_te, q10, q90),
            'width_50pct':     mean_interval_width(q25, q75),
            'width_80pct':     mean_interval_width(q10, q90),
        }
        fold_records.append(rec)

        print(
            f'    fold {fold_no} | '
            f'MAE={rec["q50_mae"]:,.0f}  '
            f'MAPE={rec["q50_mape"]:.1f}%  '
            f'cov50={rec["coverage_50pct"]:.2f}  '
            f'cov80={rec["coverage_80pct"]:.2f}  '
            f'width80={rec["width_80pct"]:,.0f}'
        )

    # ── Train final models on ALL valid data ──────────────────────────────
    X_all, final_imputer = prepare_X(valid_df, feature_cols)
    y_all = valid_df[target_col].to_numpy(float)

    final_models: dict[float, lgb.LGBMRegressor] = {}
    for q in QUANTILES:
        m = lgb.LGBMRegressor(objective='quantile', alpha=q, **LGBM_BASE_PARAMS)
        m.fit(X_all, y_all)
        final_models[q] = m

    # Save imputer alongside models (needed at inference)
    joblib.dump(final_imputer, MODEL_DIR / f'imputer_{horizon}d.joblib')

    return final_models, fold_records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Loading dataset...')
    df = pd.read_csv(DATA_FILE)
    df['searched_at'] = pd.to_datetime(df['searched_at'])

    print(f'Total rows: {len(df)}\n')

    # Build encoders from full dataset so all vocab is captured
    encoders = build_encoders(df)
    df       = apply_encoders(df, encoders)
    feat_cols = get_feature_cols(encoders)

    # Save encoders
    with open(MODEL_DIR / 'encoders.json', 'w', encoding='utf-8') as f:
        json.dump(encoders, f, ensure_ascii=False, indent=2)

    all_fold_records: list[dict] = []

    for h in HORIZONS:
        print(f'\n{"─"*55}')
        print(f'Horizon: {h} day(s) after search')
        print(f'{"─"*55}')

        models, fold_records = train_horizon(df, h, feat_cols, encoders)
        all_fold_records.extend(fold_records)

        if not models:
            continue

        # Save one model file per (horizon, quantile)
        for q, m in models.items():
            q_tag = f'q{int(q * 100):02d}'
            joblib.dump(m, MODEL_DIR / f'lgbm_{h}d_{q_tag}.joblib')

    # ── Aggregate & save metrics ──────────────────────────────────────────
    fold_df = pd.DataFrame(all_fold_records)
    fold_df.to_csv(OUTPUT_DIR / 'fold_metrics.csv', index=False, encoding='utf-8-sig')

    if not fold_df.empty:
        summary = (
            fold_df
            .groupby('horizon_days')[['q50_mae', 'q50_mape', 'coverage_50pct', 'coverage_80pct',
                                      'width_50pct', 'width_80pct']]
            .mean()
            .round(4)
        )
        summary.to_csv(OUTPUT_DIR / 'summary_metrics.csv', encoding='utf-8-sig')

        print(f'\n{"═"*55}')
        print('Summary (mean across folds):')
        print(f'{"═"*55}')
        print(summary.to_string())

    # Save metadata
    meta = {
        'trained_date':   str(date.today()),
        'horizons':       HORIZONS,
        'quantiles':      QUANTILES,
        'feature_cols':   feat_cols,
        'numeric_cols':   NUMERIC_COLS,
        'cat_cols':       CAT_COLS,
        'n_train_total':  int(len(df)),
        'lgbm_params':    LGBM_BASE_PARAMS,
    }
    with open(MODEL_DIR / 'run_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f'\nModels saved to: {MODEL_DIR}')
    print(f'Metrics saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
