"""build_quantile_dataset.py

Feature pipeline for probabilistic multi-horizon price forecasting.

For each search event, extracts the actual future price at 4 horizons
(1, 3, 7, 14 days after the search) using trajectory-level price history
(all searches on the same route + departure date).

Tolerance: ±12 hours around each target timestamp.
If no observation exists within the window → NaN (excluded per horizon).

Output: processed/quantile_dataset.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import holidays


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT  = Path(__file__).resolve().parent
RAW_DIR       = PROJECT_ROOT / 'data_0518' / 'raw'
OUTPUT_DIR    = PROJECT_ROOT / 'data_0518' / 'processed'

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HORIZONS_DAYS      = [1, 3, 7, 14]
TOLERANCE_SEC      = 12 * 3600        # ±12 h around each target timestamp
HISTORY_RECENT_N   = 5
ROLLING_WINDOW     = 3
HOLIDAY_WINDOW     = 3                # ±days around Korean holiday

LONG_HAUL_AIRPORTS = {'CDG', 'JFK', 'LHR', 'FCO', 'MAD', 'SYD'}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def safe_divide(a, b) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.where(b != 0, a / b, np.nan)


def parse_dt(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')


def compute_recent_slope(values: np.ndarray) -> float:
    v = np.asarray(values, float)
    if len(v) < 2:
        return np.nan
    x = np.arange(len(v), dtype=float)
    xm, ym = x.mean(), v.mean()
    denom = np.sum((x - xm) ** 2)
    return float(np.sum((x - xm) * (v - ym)) / denom) if denom else np.nan


def is_peak_season(date: pd.Timestamp) -> int:
    return 0 if pd.isna(date) else int(date.month in {7, 8, 12})


def build_kr_holiday_set(dates: pd.Series) -> set[pd.Timestamp]:
    valid = pd.to_datetime(dates, errors='coerce').dropna()
    if valid.empty:
        return set()
    y0, y1 = int(valid.dt.year.min()) - 1, int(valid.dt.year.max()) + 1
    kr = holidays.country_holidays('KR', years=range(y0, y1 + 1))
    return {pd.Timestamp(d).normalize() for d in kr}


def is_holiday_near(date: pd.Timestamp, hset: set, window: int = 3) -> int:
    if pd.isna(date):
        return 0
    d = pd.Timestamp(date).normalize()
    return int(any(d + pd.Timedelta(days=o) in hset for o in range(-window, window + 1)))


# ---------------------------------------------------------------------------
# Stage 1 — Search summary (static features)
# ---------------------------------------------------------------------------

def build_search_summary() -> pd.DataFrame:
    search  = pd.read_csv(RAW_DIR / 'flight_search.csv')
    offer   = pd.read_csv(RAW_DIR / 'flight_offer.csv')
    insight = pd.read_csv(RAW_DIR / 'flight_price_insight.csv')

    parse_dt(search, 'searched_at')
    parse_dt(search, 'outbound_date')

    search = search.rename(columns={'id': 'flight_search_id'})
    search = search.drop_duplicates(subset=['flight_search_id'], keep='first').copy()

    search['route_id']      = search['departure_airport_code'].astype(str) + '-' + search['arrival_airport_code'].astype(str)
    search['trajectory_id'] = search['route_id'] + '|' + search['outbound_date'].dt.strftime('%Y-%m-%d')

    search['searched_day_of_week'] = search['searched_at'].dt.day_name().str[:3].str.upper()
    search['is_weekend_search']    = search['searched_day_of_week'].isin({'SAT', 'SUN'}).astype(int)
    search['days_to_departure']    = (search['outbound_date'].dt.normalize() - search['searched_at'].dt.normalize()).dt.days
    search['is_long_haul']         = search['arrival_airport_code'].isin(LONG_HAUL_AIRPORTS).astype(int)
    search['outbound_month']       = search['outbound_date'].dt.month.astype('Int64')
    search['outbound_day_of_week'] = search['outbound_date'].dt.day_name().str[:3].str.upper()
    search['is_peak_season']       = search['outbound_date'].apply(is_peak_season).astype(int)

    kr_hol = build_kr_holiday_set(search['outbound_date'])
    search['is_holiday_near'] = search['outbound_date'].apply(
        lambda x: is_holiday_near(x, kr_hol, HOLIDAY_WINDOW)
    ).astype(int)

    if 'direction' in offer.columns:
        offer = offer[offer['direction'].astype(str).str.upper() == 'OUTBOUND'].copy()
    offer = offer.rename(columns={'id': 'flight_offer_id'})

    offer_agg = offer.groupby('flight_search_id', as_index=False).agg(
        offer_count          = ('flight_offer_id', 'size'),
        current_cheapest_price = ('price', 'min'),
        nonstop_offer_count  = ('has_layover', lambda s: int((~s.astype(bool)).sum())),
        cheapest_nonstop_price = (
            'price',
            lambda s: s[offer.loc[s.index, 'has_layover'] == False].min()
            if (offer.loc[s.index, 'has_layover'] == False).any() else np.nan,
        ),
    )
    offer_agg['nonstop_ratio'] = safe_divide(offer_agg['nonstop_offer_count'], offer_agg['offer_count'])

    cheapest_layover = (
        offer.sort_values(['flight_search_id', 'price', 'total_duration', 'flight_offer_id'])
        .groupby('flight_search_id', as_index=False).first()
        [['flight_search_id', 'has_layover']]
        .rename(columns={'has_layover': 'cheapest_offer_has_layover'})
    )

    insight = insight.drop_duplicates(subset=['flight_search_id'], keep='first').copy()

    summary = (
        search
        .merge(offer_agg, on='flight_search_id', how='left')
        .merge(cheapest_layover, on='flight_search_id', how='left')
        .merge(
            insight[['flight_search_id', 'price_level', 'typical_price_min', 'typical_price_max']],
            on='flight_search_id', how='left',
        )
    )

    summary['curr_gap_to_typical_min']     = summary['current_cheapest_price'] - summary['typical_price_min']
    summary['curr_gap_to_typical_max']     = summary['current_cheapest_price'] - summary['typical_price_max']
    summary['cheapest_offer_has_layover']  = summary['cheapest_offer_has_layover'].fillna(False).astype(int)

    for col in ('is_weekend_search', 'is_long_haul', 'is_peak_season', 'is_holiday_near'):
        summary[col] = summary[col].fillna(0).astype(int)

    return summary.sort_values(['trajectory_id', 'searched_at', 'flight_search_id']).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stage 2 — History summary features (same as XGBoost v5)
# ---------------------------------------------------------------------------

def add_history_features(summary: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    hist = history.copy()
    hist['observed_at'] = pd.to_datetime(hist['time_stamp'], unit='s', errors='coerce')
    hist = hist.rename(columns={'price': 'history_price'})

    traj_map = summary[['flight_search_id', 'trajectory_id', 'searched_at']].copy()
    hist = hist.merge(traj_map[['flight_search_id', 'trajectory_id']], on='flight_search_id', how='left')

    summary = summary.sort_values(['trajectory_id', 'searched_at']).reset_index(drop=True)
    grouped = {tid: g.sort_values('observed_at') for tid, g in hist.groupby('trajectory_id')}

    rows: list[dict[str, Any]] = []
    for row in summary.itertuples(index=False):
        g = grouped.get(row.trajectory_id)
        if g is None:
            rows.append({})
            continue
        usable = g[g['observed_at'] <= pd.Timestamp(row.searched_at)]
        prices = usable['history_price'].dropna().astype(float).to_numpy()
        if len(prices) == 0:
            rows.append({})
            continue
        recent = prices[-HISTORY_RECENT_N:]
        rows.append({
            'hist_recent_std':   float(np.std(recent, ddof=1)) if len(recent) > 1 else 0.0,
            'hist_recent_slope': compute_recent_slope(recent),
            'hist_mean_price':   float(np.mean(prices)),
        })

    feat_df = pd.DataFrame(rows)
    out = pd.concat([summary.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
    out['curr_vs_hist_mean'] = safe_divide(out['current_cheapest_price'], out['hist_mean_price'])
    return out.drop(columns=['hist_mean_price'], errors='ignore')


# ---------------------------------------------------------------------------
# Stage 3 — Lag / rolling features within trajectory
# ---------------------------------------------------------------------------

def add_lag_features(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary.sort_values(['trajectory_id', 'searched_at', 'flight_search_id']).reset_index(drop=True).copy()
    grp = df.groupby('trajectory_id')['current_cheapest_price']

    df['lag_1_price']        = grp.shift(1)
    df['price_change_1']     = df['current_cheapest_price'] - df['lag_1_price']
    df['rolling_std_3']      = grp.rolling(ROLLING_WINDOW, min_periods=1).std().reset_index(level=0, drop=True)
    rolling_mean             = grp.rolling(ROLLING_WINDOW, min_periods=1).mean().reset_index(level=0, drop=True)
    df['price_vs_rolling_mean_3'] = df['current_cheapest_price'] - rolling_mean

    for col in ('hist_recent_std', 'rolling_std_3'):
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    return df


# ---------------------------------------------------------------------------
# Stage 4 — Multi-horizon future price targets
# ---------------------------------------------------------------------------

def extract_horizon_targets(summary: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """For each search event, find the actual price at +1/+3/+7/+14 days.

    Uses trajectory-level history (all searches on same route+departure_date).
    Only observations STRICTLY AFTER searched_at are considered (no leakage).
    Accepts the closest observation within ±TOLERANCE_SEC of each target timestamp.
    """
    hist = history.copy()
    hist = hist.rename(columns={'price': 'history_price'})
    hist = hist.dropna(subset=['history_price'])

    traj_map = summary[['flight_search_id', 'trajectory_id']].drop_duplicates('flight_search_id')
    hist = hist.merge(traj_map, on='flight_search_id', how='left').dropna(subset=['trajectory_id'])

    # Pre-group by trajectory, sorted by unix timestamp
    traj_hist: dict[str, pd.DataFrame] = {
        tid: g.sort_values('time_stamp').reset_index(drop=True)
        for tid, g in hist.groupby('trajectory_id')
    }

    records: list[dict] = []
    for row in summary.itertuples(index=False):
        searched_ts = pd.Timestamp(row.searched_at).timestamp()
        g = traj_hist.get(row.trajectory_id)

        rec: dict[str, Any] = {'flight_search_id': row.flight_search_id}
        for h in HORIZONS_DAYS:
            col = f'price_{h}d'
            if g is None:
                rec[col] = np.nan
                continue

            # Only future observations (strict)
            future = g[g['time_stamp'] > searched_ts]
            if len(future) == 0:
                rec[col] = np.nan
                continue

            target_ts = searched_ts + h * 86400
            ts_arr    = future['time_stamp'].to_numpy(dtype=float)
            diffs     = np.abs(ts_arr - target_ts)
            best_idx  = int(diffs.argmin())

            rec[col] = float(future.iloc[best_idx]['history_price']) if diffs[best_idx] <= TOLERANCE_SEC else np.nan

        records.append(rec)

    targets_df = pd.DataFrame(records)
    return summary.merge(targets_df, on='flight_search_id', how='left')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Loading raw data...')
    history = pd.read_csv(RAW_DIR / 'flight_price_history.csv')

    print('Building search summary (static features)...')
    summary = build_search_summary()

    print('Adding history features...')
    summary = add_history_features(summary, history)

    print('Adding lag / rolling features...')
    summary = add_lag_features(summary)

    print('Extracting multi-horizon targets...')
    df = extract_horizon_targets(summary, history)

    # Report target coverage per horizon
    print(f'\nTotal search events: {len(df)}')
    print('Target coverage per horizon:')
    for h in HORIZONS_DAYS:
        col   = f'price_{h}d'
        valid = df[col].notna().sum()
        pct   = valid / len(df) * 100
        print(f'  price_{h:>2d}d : {valid:>4d} / {len(df)} ({pct:.1f}%)')

    out_path = OUTPUT_DIR / 'quantile_dataset.csv'
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'\nSaved → {out_path}')
    print(f'Columns: {df.columns.tolist()}')


if __name__ == '__main__':
    main()
