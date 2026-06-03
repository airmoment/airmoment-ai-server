"""train_cqr_model.py

Split Conformal Prediction for multi-horizon flight price forecasting.
  • Train one LightGBM q50 model per horizon  →  4 models total
  • LightGBM handles NaN natively — no imputer needed
  • Everything bundled into one file: airmoment_model.joblib
  • External features (oil/FX) only used in 7d, 14d models

Bundle contents:
  {
    'models':      {1: lgbm, 3: lgbm, 7: lgbm, 14: lgbm},
    'conf':        {'1d': {'80pct': ..., '50pct': ...}, ...},
    'encoders':    {'route_id': {...}, ...},
    'feature_cols': {1: [...], 3: [...], 7: [...], 14: [...]},  ← horizon별 상이
  }

Saved outputs (models/):
  airmoment_model.joblib

Saved outputs (outputs/):
  fold_metrics.csv
  summary_metrics.csv
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILE    = PROJECT_ROOT / 'data_0603' / 'processed' / 'quantile_dataset_v2.csv'
MODEL_DIR    = PROJECT_ROOT / 'data_0603' / 'models'
OUTPUT_DIR   = PROJECT_ROOT / 'data_0603' / 'outputs'

HORIZONS = [1, 3, 7, 14]

BANDS = {
    '80pct': 0.80,
    '50pct': 0.50,
}

N_SPLITS       = 4
TEST_FRACTION  = 0.20
CALIB_FRACTION = 0.20
MIN_TRAIN      = 20
RANDOM_STATE   = 42

CAT_COLS = ['route_id', 'searched_day_of_week', 'outbound_day_of_week', 'price_level']

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

# 외부 요인 피처: 장기 모델(7d, 14d)에서만 사용
EXT_COLS       = ['oil_price_usd', 'oil_change_7d', 'arr_fx_change_7d']
EXT_HORIZONS   = {7, 14}

LGBM_PARAMS = dict(
    objective='quantile', alpha=0.50,
    n_estimators=400, max_depth=4, num_leaves=15,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    min_child_samples=5, reg_alpha=0.5, reg_lambda=1.0,
    random_state=RANDOM_STATE, verbose=-1,
)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def build_encoders(df: pd.DataFrame) -> dict:
    encoders = {}
    for col in CAT_COLS:
        if col not in df.columns:
            continue
        vocab = sorted(df[col].fillna('__missing__').astype(str).unique().tolist())
        encoders[col] = {v: i for i, v in enumerate(vocab)}
    return encoders


def apply_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in encoders.items():
        if col not in out.columns:
            continue
        out[f'{col}_enc'] = out[col].fillna('__missing__').astype(str).map(mapping).astype(float)
    return out


def get_feature_cols(encoders: dict, include_ext: bool = True) -> list[str]:
    num = [c for c in NUMERIC_COLS if include_ext or c not in EXT_COLS]
    return num + [f'{c}_enc' for c in encoders]


def get_X(df: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    """Extract feature matrix — LightGBM handles NaN natively."""
    return df[feat_cols].astype(float).to_numpy()


# ---------------------------------------------------------------------------
# Walk-forward splits
# ---------------------------------------------------------------------------

def make_splits(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    test_size  = max(5, int(np.floor(n * TEST_FRACTION)))
    max_splits = max(1, (n - MIN_TRAIN) // test_size)
    n_actual   = min(N_SPLITS, max_splits)
    splits = []
    for i in range(n_actual):
        train_end = MIN_TRAIN + i * test_size
        test_end  = min(train_end + test_size, n)
        if test_end > train_end:
            splits.append((np.arange(0, train_end), np.arange(train_end, test_end)))
    return splits


def split_train_calib(train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_calib  = max(5, int(np.floor(len(train_idx) * CALIB_FRACTION)))
    n_proper = len(train_idx) - n_calib
    return train_idx[:n_proper], train_idx[n_proper:]


# ---------------------------------------------------------------------------
# Conformal core
# ---------------------------------------------------------------------------

def conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    n     = len(scores)
    level = min(1.0, coverage * (1 + 1 / n))
    return float(np.quantile(scores, level))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def cov(y, lo, hi):   return float(np.mean((y >= lo) & (y <= hi)))
def width(lo, hi):    return float(np.mean(hi - lo))
def mae(y, p):        return float(np.mean(np.abs(y - p)))
def mape(y, p):
    mask = y != 0
    return float(np.mean(np.abs((y[mask]-p[mask])/y[mask]))*100) if mask.any() else np.nan


# ---------------------------------------------------------------------------
# Per-horizon training + evaluation
# ---------------------------------------------------------------------------

def train_horizon(
    df: pd.DataFrame,
    horizon: int,
    feat_cols: list[str],
) -> tuple[dict, list[dict]]:
    target_col = f'price_{horizon}d'
    valid_df   = df[df[target_col].notna()].sort_values('searched_at').reset_index(drop=True)
    n          = len(valid_df)

    if n < MIN_TRAIN + 10:
        print(f'  [SKIP {horizon}d] Only {n} valid samples.')
        return {}, []

    splits = make_splits(n)
    print(f'  horizon={horizon}d | {n} samples | {len(splits)} folds')

    fold_records: list[dict] = []

    for fold_no, (tr_idx, te_idx) in enumerate(splits, 1):
        proper_idx, cal_idx = split_train_calib(tr_idx)

        tr_df  = valid_df.iloc[proper_idx]
        cal_df = valid_df.iloc[cal_idx]
        te_df  = valid_df.iloc[te_idx]

        y_tr  = tr_df[target_col].to_numpy(float)
        y_cal = cal_df[target_col].to_numpy(float)
        y_te  = te_df[target_col].to_numpy(float)

        X_tr  = get_X(tr_df,  feat_cols)
        X_cal = get_X(cal_df, feat_cols)
        X_te  = get_X(te_df,  feat_cols)

        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(X_tr, y_tr)

        cal_q50     = model.predict(X_cal)
        conf_scores = np.abs(y_cal - cal_q50)

        conf_qhat: dict[str, float] = {}
        for band, target_cov in BANDS.items():
            conf_qhat[band] = conformal_quantile(conf_scores, target_cov)

        q50_te   = model.predict(X_te)
        q10_conf = q50_te - conf_qhat['80pct']
        q90_conf = q50_te + conf_qhat['80pct']
        q25_conf = q50_te - conf_qhat['50pct']
        q75_conf = q50_te + conf_qhat['50pct']

        rec = {
            'horizon_days': horizon,
            'fold':         fold_no,
            'proper_rows':  len(proper_idx),
            'calib_rows':   len(cal_idx),
            'test_rows':    len(te_idx),
            'q50_mae':      mae(y_te, q50_te),
            'q50_mape':     mape(y_te, q50_te),
            'conf_cov80':   cov(y_te, q10_conf, q90_conf),
            'conf_width80': width(q10_conf, q90_conf),
            'conf_cov50':   cov(y_te, q25_conf, q75_conf),
            'conf_width50': width(q25_conf, q75_conf),
            'conf_qhat80':  conf_qhat['80pct'],
            'conf_qhat50':  conf_qhat['50pct'],
        }
        fold_records.append(rec)

        print(
            f'    fold {fold_no} | '
            f'proper={len(proper_idx)} cal={len(cal_idx)} test={len(te_idx)} | '
            f'MAE={rec["q50_mae"]:,.0f} ({rec["q50_mape"]:.1f}%)'
        )
        print(
            f'           80% │ cov={rec["conf_cov80"]:.2f}  '
            f'width={rec["conf_width80"]/1000:,.0f}k  '
            f'q_hat={rec["conf_qhat80"]/1000:,.0f}k'
        )
        print(
            f'           50% │ cov={rec["conf_cov50"]:.2f}  '
            f'width={rec["conf_width50"]/1000:,.0f}k  '
            f'q_hat={rec["conf_qhat50"]/1000:,.0f}k'
        )

    # ── Final production model ────────────────────────────────────────────
    n_calib_f  = max(10, int(np.floor(n * CALIB_FRACTION)))
    n_proper_f = n - n_calib_f
    final_tr_df  = valid_df.iloc[:n_proper_f]
    final_cal_df = valid_df.iloc[n_proper_f:]

    X_tr_f  = get_X(final_tr_df,  feat_cols)
    X_cal_f = get_X(final_cal_df, feat_cols)
    y_cal_f = final_cal_df[target_col].to_numpy(float)

    final_model = lgb.LGBMRegressor(**LGBM_PARAMS)
    final_model.fit(X_tr_f, final_tr_df[target_col].to_numpy(float))

    final_q50_cal     = final_model.predict(X_cal_f)
    final_conf_scores = np.abs(y_cal_f - final_q50_cal)
    final_conf: dict[str, float] = {}
    for band, target_cov in BANDS.items():
        final_conf[band] = conformal_quantile(final_conf_scores, target_cov)

    print(
        f'  → q_hat | '
        f'80pct={final_conf["80pct"]:,.0f}  '
        f'50pct={final_conf["50pct"]:,.0f}'
    )

    return {'model': final_model, 'conf': final_conf}, fold_records


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

    encoders        = build_encoders(df)
    df              = apply_encoders(df, encoders)
    feat_cols_short = get_feature_cols(encoders, include_ext=False)  # 1d, 3d
    feat_cols_long  = get_feature_cols(encoders, include_ext=True)   # 7d, 14d

    all_fold_records: list[dict] = []
    bundle = {
        'models':       {},   # {horizon_int: lgbm_model}
        'conf':         {},   # {'1d': {'80pct': ..., '50pct': ...}, ...}
        'encoders':     encoders,
        'feature_cols': {     # horizon별 상이
            h: (feat_cols_long if h in EXT_HORIZONS else feat_cols_short)
            for h in HORIZONS
        },
        'trained_date': str(date.today()),
        'lgbm_params':  LGBM_PARAMS,
    }

    for h in HORIZONS:
        feat_cols = feat_cols_long if h in EXT_HORIZONS else feat_cols_short
        print(f'\n{"─"*65}')
        print(f'Horizon: {h} day(s)  │  피처 수: {len(feat_cols)}  (외부요인: {"포함" if h in EXT_HORIZONS else "제외"})')
        print(f'{"─"*65}')

        artifacts, fold_records = train_horizon(df, h, feat_cols)
        all_fold_records.extend(fold_records)

        if not artifacts:
            continue

        bundle['models'][h]      = artifacts['model']
        bundle['conf'][f'{h}d']  = artifacts['conf']

    joblib.dump(bundle, MODEL_DIR / 'airmoment_forecast.joblib')
    print(f'\nBundle saved → {MODEL_DIR / "airmoment_forecast.joblib"}')

    # ── Summary ──────────────────────────────────────────────────────────────
    fold_df = pd.DataFrame(all_fold_records)
    fold_df.to_csv(OUTPUT_DIR / 'fold_metrics.csv', index=False, encoding='utf-8-sig')

    if not fold_df.empty:
        metric_cols = [
            'q50_mae', 'q50_mape',
            'conf_cov80', 'conf_cov50',
            'conf_width80', 'conf_width50',
            'conf_qhat80', 'conf_qhat50',
        ]
        summary = (
            fold_df
            .groupby('horizon_days')[metric_cols]
            .mean()
            .round(4)
        )
        summary.to_csv(OUTPUT_DIR / 'summary_metrics.csv', encoding='utf-8-sig')

        print(f'\n{"═"*65}')
        print('Summary — mean across folds (Split Conformal)')
        print(f'{"═"*65}')
        print(f'\n  {"Horizon":>7}  │  {"── 80% band ──":^26}  │  {"── 50% band ──":^26}  │  {"── q50 accuracy ──":^22}')
        print(f'  {"":>7}  │  {"cov":>6}  {"width":>8}  {"q_hat":>8}  │  {"cov":>6}  {"width":>8}  {"q_hat":>8}  │  {"MAE (₩)":>10}  {"MAPE":>6}')
        print(f'  {"─"*7}  │  {"─"*6}  {"─"*8}  {"─"*8}  │  {"─"*6}  {"─"*8}  {"─"*8}  │  {"─"*10}  {"─"*6}')
        for h in HORIZONS:
            if h not in summary.index:
                continue
            r = summary.loc[h]
            print(
                f'  {h:>5}d   │  '
                f'{r["conf_cov80"]:>6.2f}  {r["conf_width80"]/1000:>7.0f}k  {r["conf_qhat80"]/1000:>7.0f}k  │  '
                f'{r["conf_cov50"]:>6.2f}  {r["conf_width50"]/1000:>7.0f}k  {r["conf_qhat50"]/1000:>7.0f}k  │  '
                f'{r["q50_mae"]:>10,.0f}  {r["q50_mape"]:>5.1f}%'
            )

    print(f'\nModel  → {MODEL_DIR / "airmoment_forecast.joblib"}')
    print(f'Metrics → {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
