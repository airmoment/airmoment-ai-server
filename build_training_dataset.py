from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import holidays


PROJECT_ROOT = Path(__file__).resolve().parent

RAW_SEARCH = PROJECT_ROOT / 'data_0428/raw/flight_search.csv'
RAW_OFFER = PROJECT_ROOT / 'data_0428/raw/flight_offer.csv'
RAW_SEGMENT = PROJECT_ROOT / 'data_0428/raw/flight_segment.csv'
RAW_LAYOVER = PROJECT_ROOT / 'data_0428/raw/flight_layover.csv'
RAW_INSIGHT = PROJECT_ROOT / 'data_0428/raw/flight_price_insight.csv'
RAW_HISTORY = PROJECT_ROOT / 'data_0428/raw/flight_price_history.csv'

OUTPUT_SUMMARY = PROJECT_ROOT / 'data_0428/processed/flight_search_summary_final_v2.csv'
OUTPUT_DATASET = PROJECT_ROOT / 'data_0428/processed/flight_model_dataset_final_v2.csv'

FUTURE_WINDOW_OBS = 40
TARGET_DROP_THRESHOLD = 2000
TARGET_DROP_RATIO_THRESHOLD = 0.03

ROLLING_WINDOW = 3
HISTORY_RECENT_POINTS = 5
HOLIDAY_NEAR_WINDOW_DAYS = 3


def safe_divide(a, b) -> np.ndarray:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    return np.where(b_arr != 0, a_arr / b_arr, np.nan)


def parse_datetime_col(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')


def bucket_hour(hour: float | int | None) -> str:
    if pd.isna(hour):
        return 'unknown'
    hour = int(hour)
    if 0 <= hour < 6:
        return 'late_night'
    if 6 <= hour < 12:
        return 'morning'
    if 12 <= hour < 18:
        return 'afternoon'
    return 'evening'


def compute_future_min_by_group(group: pd.DataFrame, future_window_obs: int) -> pd.Series:
    prices = group['current_cheapest_price'].to_numpy(dtype=float)
    future_min = np.full(len(group), np.nan, dtype=float)

    for i in range(len(group)):
        window_prices = prices[i + 1: i + 1 + future_window_obs]
        valid_prices = window_prices[~np.isnan(window_prices)]
        if len(valid_prices) > 0:
            future_min[i] = float(np.min(valid_prices))

    return pd.Series(future_min, index=group.index)


def compute_recent_slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.nan
    x = np.arange(len(values), dtype=float)
    x_mean = x.mean()
    y_mean = values.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return np.nan
    return float(np.sum((x - x_mean) * (values - y_mean)) / denom)


def clean_search_table(search: pd.DataFrame) -> pd.DataFrame:
    search = search.drop_duplicates().copy()
    dup_mask = search['flight_search_id'].duplicated(keep=False)
    if dup_mask.any():
        dup_ids = search.loc[dup_mask, 'flight_search_id'].drop_duplicates().tolist()
        print(
            'Warning: duplicated flight_search_id values found in flight_search.csv. '
            f'Keeping first occurrence for ids: {dup_ids}'
        )
        search = search.drop_duplicates(subset=['flight_search_id'], keep='first').copy()
    return search


def is_peak_season(date: pd.Timestamp) -> int:
    if pd.isna(date):
        return 0
    month = int(date.month)
    return 1 if month in [7, 8, 12] else 0


def build_kr_holiday_set(dates: pd.Series) -> set[pd.Timestamp]:
    valid_dates = pd.to_datetime(dates, errors='coerce').dropna()
    if valid_dates.empty:
        return set()

    min_year = int(valid_dates.dt.year.min()) - 1
    max_year = int(valid_dates.dt.year.max()) + 1

    kr_holidays = holidays.country_holidays('KR', years=range(min_year, max_year + 1))
    return {pd.Timestamp(d).normalize() for d in kr_holidays.keys()}


def is_holiday_near(date: pd.Timestamp, holiday_set: set[pd.Timestamp], window_days: int = 3) -> int:
    if pd.isna(date):
        return 0

    date = pd.Timestamp(date).normalize()
    for offset in range(-window_days, window_days + 1):
        if date + pd.Timedelta(days=offset) in holiday_set:
            return 1
    return 0


def build_search_summary() -> pd.DataFrame:
    search = pd.read_csv(RAW_SEARCH)
    offer = pd.read_csv(RAW_OFFER)
    segment = pd.read_csv(RAW_SEGMENT)
    layover = pd.read_csv(RAW_LAYOVER)
    insight = pd.read_csv(RAW_INSIGHT)

    parse_datetime_col(search, 'searched_at')
    parse_datetime_col(search, 'outbound_date')
    parse_datetime_col(segment, 'departure_time')

    search = search.rename(columns={'id': 'flight_search_id'})
    search = clean_search_table(search)

    search['route_id'] = search['departure_airport_code'].astype(str) + '-' + search['arrival_airport_code'].astype(str)
    search['trajectory_id'] = search['route_id'].astype(str) + '|' + search['outbound_date'].dt.strftime('%Y-%m-%d')

    search['searched_day_of_week'] = search['searched_at'].dt.day_name().str[:3].str.upper()
    search['is_weekend_search'] = search['searched_day_of_week'].isin(['SAT', 'SUN']).astype(int)

    search['days_to_departure'] = (
        search['outbound_date'].dt.normalize() - search['searched_at'].dt.normalize()
    ).dt.days

    search['is_long_haul'] = search['arrival_airport_code'].isin(['CDG', 'JFK', 'LHR', 'FCO', 'MAD', 'SYD']).astype(int)

    search['outbound_month'] = search['outbound_date'].dt.month.astype('Int64')
    search['outbound_day_of_week'] = search['outbound_date'].dt.day_name().str[:3].str.upper()
    search['is_peak_season'] = search['outbound_date'].apply(is_peak_season).astype(int)

    kr_holiday_set = build_kr_holiday_set(search['outbound_date'])
    search['is_holiday_near'] = search['outbound_date'].apply(
        lambda x: is_holiday_near(x, kr_holiday_set, HOLIDAY_NEAR_WINDOW_DAYS)
    ).astype(int)

    if 'direction' in offer.columns:
        offer = offer[offer['direction'].astype(str).str.upper() == 'OUTBOUND'].copy()
    offer = offer.rename(columns={'id': 'flight_offer_id'})

    offer_summary = (
        offer.groupby('flight_search_id', as_index=False)
        .agg(
            offer_count=('flight_offer_id', 'size'),
            current_cheapest_price=('price', 'min'),
            nonstop_offer_count=('has_layover', lambda s: int((~s.astype(bool)).sum())),
            cheapest_nonstop_price=(
                'price',
                lambda s: (
                    s[offer.loc[s.index, 'has_layover'] == False].min()
                    if (offer.loc[s.index, 'has_layover'] == False).any()
                    else np.nan
                ),
            ),
        )
    )
    offer_summary['nonstop_ratio'] = safe_divide(offer_summary['nonstop_offer_count'], offer_summary['offer_count'])

    cheapest_offer_map = (
        offer.sort_values(['flight_search_id', 'price', 'total_duration', 'flight_offer_id'])
        .groupby('flight_search_id', as_index=False)
        .first()[['flight_search_id', 'flight_offer_id', 'has_layover']]
        .rename(columns={
            'flight_offer_id': 'cheapest_offer_id',
            'has_layover': 'cheapest_offer_has_layover',
        })
    )

    first_segments = (
        segment.sort_values(['flight_offer_id', 'segment_order'])
        .groupby('flight_offer_id', as_index=False)
        .first()[['flight_offer_id', 'departure_time']]
    )
    first_segments['departure_time_bucket'] = first_segments['departure_time'].dt.hour.map(bucket_hour)

    insight = insight.drop_duplicates(subset=['flight_search_id'], keep='first').copy()

    summary = search.merge(offer_summary, on='flight_search_id', how='left')
    summary = summary.merge(cheapest_offer_map, on='flight_search_id', how='left')
    summary = summary.merge(
        first_segments[['flight_offer_id', 'departure_time_bucket']],
        left_on='cheapest_offer_id',
        right_on='flight_offer_id',
        how='left',
    )
    summary = summary.merge(
        insight[['flight_search_id', 'price_level', 'typical_price_min', 'typical_price_max']],
        on='flight_search_id',
        how='left',
    )

    summary['curr_gap_to_typical_min'] = summary['current_cheapest_price'] - summary['typical_price_min']
    summary['curr_gap_to_typical_max'] = summary['current_cheapest_price'] - summary['typical_price_max']

    summary['nonstop_offer_count'] = summary['nonstop_offer_count'].fillna(0).astype(int)
    summary['offer_count'] = summary['offer_count'].fillna(0).astype(int)
    summary['is_weekend_search'] = summary['is_weekend_search'].fillna(0).astype(int)
    summary['is_long_haul'] = summary['is_long_haul'].fillna(0).astype(int)
    summary['is_peak_season'] = summary['is_peak_season'].fillna(0).astype(int)
    summary['is_holiday_near'] = summary['is_holiday_near'].fillna(0).astype(int)
    summary['cheapest_offer_has_layover'] = summary['cheapest_offer_has_layover'].fillna(False).astype(int)

    keep_cols = [
        'flight_search_id',
        'searched_at',
        'route_id',
        'trajectory_id',
        'days_to_departure',
        'searched_day_of_week',
        'outbound_month',
        'outbound_day_of_week',
        'is_weekend_search',
        'is_peak_season',
        'is_holiday_near',
        'is_long_haul',
        'offer_count',
        'nonstop_ratio',
        'cheapest_nonstop_price',
        'cheapest_offer_has_layover',
        'current_cheapest_price',
        'curr_gap_to_typical_min',
        'curr_gap_to_typical_max',
    ]
    keep_cols = [c for c in keep_cols if c in summary.columns]
    return summary[keep_cols].copy()


def add_history_features(summary: pd.DataFrame) -> pd.DataFrame:
    history = pd.read_csv(RAW_HISTORY)
    history['observed_at'] = pd.to_datetime(history['time_stamp'], unit='s', errors='coerce')
    history = history.rename(columns={'price': 'history_price'})

    hist = history.merge(
        summary[['flight_search_id', 'trajectory_id', 'searched_at']],
        on='flight_search_id',
        how='left',
        validate='many_to_one',
    )

    summary = summary.sort_values(['trajectory_id', 'searched_at']).reset_index(drop=True)
    grouped_hist = {k: g.sort_values('observed_at').copy() for k, g in hist.groupby('trajectory_id')}

    rows: list[dict[str, Any]] = []
    for row in summary.itertuples(index=False):
        g = grouped_hist.get(row.trajectory_id)
        if g is None:
            rows.append({})
            continue

        usable = g[g['observed_at'] <= row.searched_at]
        prices = usable['history_price'].dropna().astype(float).to_numpy()

        if len(prices) == 0:
            rows.append({})
            continue

        recent = prices[-HISTORY_RECENT_POINTS:]
        rows.append(
            {
                'hist_recent_std': float(np.std(recent, ddof=1)) if len(recent) > 1 else 0.0,
                'hist_recent_slope': compute_recent_slope(recent),
                'hist_mean_price': float(np.mean(prices)),
            }
        )

    hist_feat_df = pd.DataFrame(rows)
    out = pd.concat([summary.reset_index(drop=True), hist_feat_df.reset_index(drop=True)], axis=1)
    out['curr_vs_hist_mean'] = safe_divide(out['current_cheapest_price'], out['hist_mean_price'])
    out = out.drop(columns=['hist_mean_price'], errors='ignore')
    return out


def build_training_dataset(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary.sort_values(['trajectory_id', 'searched_at', 'flight_search_id']).reset_index(drop=True).copy()

    grouped_price = df.groupby('trajectory_id')['current_cheapest_price']

    df['lag_1_price'] = grouped_price.shift(1)
    df['price_change_1'] = df['current_cheapest_price'] - df['lag_1_price']

    df['rolling_std_3'] = (
        grouped_price.rolling(window=ROLLING_WINDOW, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
    )

    rolling_mean_3 = (
        grouped_price.rolling(window=ROLLING_WINDOW, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df['price_vs_rolling_mean_3'] = df['current_cheapest_price'] - rolling_mean_3

    df['target_future_min_price'] = (
        df.groupby('trajectory_id', group_keys=False)
        .apply(lambda g: compute_future_min_by_group(g, FUTURE_WINDOW_OBS))
        .reset_index(level=0, drop=True)
    )

    raw_drop = df['current_cheapest_price'] - df['target_future_min_price']

    df['target_drop_amount'] = raw_drop.clip(lower=0)

    df['target_drop_ratio'] = safe_divide(df['target_drop_amount'], df['current_cheapest_price'])
    df['target_drop_ratio'] = pd.Series(df['target_drop_ratio'], index=df.index).clip(lower=0, upper=0.3)
    df['target_log_ratio'] = np.log1p(df['target_drop_ratio'])

    df['target_future_min_delta'] = df['target_future_min_price'] - df['current_cheapest_price']

    df['target_wait'] = (df['target_drop_ratio'] >= TARGET_DROP_RATIO_THRESHOLD).astype(int)

    df_model = df[df['target_future_min_price'].notna()].copy()

    print('\n=== target_wait distribution ===')
    print(df_model['target_wait'].value_counts(dropna=False))
    print('positive ratio:', df_model['target_wait'].mean())

    print('\n=== target_drop_amount summary ===')
    print(df_model['target_drop_amount'].describe())

    print('\n=== target_drop_ratio summary ===')
    print(df_model['target_drop_ratio'].describe())

    print('\n=== zero-drop ratio ===')
    print((df_model['target_drop_amount'] == 0).mean())

    for col in ['hist_recent_std', 'rolling_std_3']:
        if col in df_model.columns:
            df_model[col] = df_model[col].fillna(0.0)

    feature_cols = [
        'flight_search_id',
        'searched_at',
        'route_id',
        'trajectory_id',
        'days_to_departure',
        'searched_day_of_week',
        'outbound_month',
        'outbound_day_of_week',
        'is_weekend_search',
        'is_peak_season',
        'is_holiday_near',
        'is_long_haul',
        'offer_count',
        'nonstop_ratio',
        'cheapest_nonstop_price',
        'cheapest_offer_has_layover',
        'current_cheapest_price',
        'curr_gap_to_typical_min',
        'curr_gap_to_typical_max',
        'hist_recent_std',
        'hist_recent_slope',
        'curr_vs_hist_mean',
        'price_change_1',
        'rolling_std_3',
        'price_vs_rolling_mean_3',
        'target_future_min_price',
        'target_drop_amount',
        'target_drop_ratio',
        'target_log_ratio',
        'target_future_min_delta',
        'target_wait',
    ]
    feature_cols = [c for c in feature_cols if c in df_model.columns]
    return df_model[feature_cols].copy()


def main() -> None:
    summary = build_search_summary()
    summary = add_history_features(summary)
    summary.to_csv(OUTPUT_SUMMARY, index=False, encoding='utf-8-sig')

    dataset = build_training_dataset(summary)
    dataset.to_csv(OUTPUT_DATASET, index=False, encoding='utf-8-sig')

    print(f'Saved summary to: {OUTPUT_SUMMARY}')
    print(f'Saved model dataset to: {OUTPUT_DATASET}')
    print('Summary shape:', summary.shape)
    print('Dataset shape:', dataset.shape)
    print('Columns:')
    print(dataset.columns.tolist())


if __name__ == '__main__':
    main()
