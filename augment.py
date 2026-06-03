"""augment.py — 시계열 테이블 데이터 증강 기법

Methods
-------
jitter(df, sigma, n_copies)      연속형 피처에 Gaussian noise 추가
window_slice(df, n_slices, ...)  시간 순서 유지한 채 여러 하위 구간 겹쳐 붙이기

Rules
-----
- 이진 피처(BINARY_FEATURES)는 노이즈 적용 안 함
- 타겟 컬럼(target_*)은 변경하지 않음
- 원본 df 변경 없음 — 항상 새 DataFrame 반환
"""
from __future__ import annotations
import numpy as np
import pandas as pd

CONTINUOUS_FEATURES: list[str] = [
    'days_to_departure', 'offer_count', 'nonstop_ratio',
    'cheapest_nonstop_price', 'current_cheapest_price',
    'curr_gap_to_typical_min', 'curr_gap_to_typical_max',
    'hist_recent_std', 'hist_recent_slope', 'curr_vs_hist_mean',
    'price_change_1', 'rolling_std_3', 'price_vs_rolling_mean_3',
    'oil_price_usd', 'oil_change_7d', 'arr_fx_change_7d',
]

BINARY_FEATURES: list[str] = [
    'is_weekend_search', 'is_peak_season', 'is_holiday_near',
    'is_long_haul', 'cheapest_offer_has_layover',
    'outbound_month', 'searched_day_of_week', 'outbound_day_of_week', 'route_id',
]


def _present_continuous(df: pd.DataFrame) -> list[str]:
    return [c for c in CONTINUOUS_FEATURES if c in df.columns]


def _col_std(df: pd.DataFrame, col: str) -> float:
    s = df[col].std()
    return float(s) if (s and not np.isnan(s) and s > 0) else 1.0


def jitter(
    df: pd.DataFrame,
    sigma: float = 0.02,
    n_copies: int = 3,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """연속형 피처에 Gaussian noise를 추가한 복사본 n_copies개를 붙여 반환.
    반환 크기 = len(df) × (1 + n_copies).
    sigma : noise 크기 = sigma × col_std
    """
    if rng is None:
        rng = np.random.default_rng(42)
    cont_cols = _present_continuous(df)
    col_stds  = {c: _col_std(df, c) for c in cont_cols}
    parts = [df]
    for _ in range(n_copies):
        copy = df.copy()
        for col in cont_cols:
            copy[col] = copy[col] + rng.normal(0.0, sigma * col_stds[col], size=len(df))
        parts.append(copy)
    return pd.concat(parts, ignore_index=True)


def window_slice(
    df: pd.DataFrame,
    n_slices: int = 4,
    min_ratio: float = 0.65,
    max_ratio: float = 0.90,
    time_col: str = 'searched_at',
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """시간 순서를 유지하며 학습 세트의 여러 연속 구간을 겹쳐 붙임.
    n_slices개의 랜덤 구간(min_ratio~max_ratio 길이)을 원본에 추가.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(df)
    if n < 10:
        return df.copy()
    sorted_df = df.sort_values(time_col).reset_index(drop=True) if time_col in df.columns else df.copy()
    parts = [sorted_df]
    for _ in range(n_slices):
        ratio      = float(rng.uniform(min_ratio, max_ratio))
        window_len = max(5, int(np.floor(n * ratio)))
        max_start  = n - window_len
        if max_start <= 0:
            parts.append(sorted_df)
            continue
        start = int(rng.integers(0, max_start + 1))
        parts.append(sorted_df.iloc[start: start + window_len].copy())
    return pd.concat(parts, ignore_index=True)
