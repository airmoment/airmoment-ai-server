
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# =========================
# Config
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent

RAW_SEARCH = PROJECT_ROOT / "data/raw/flight_search.csv"
RAW_OFFER = PROJECT_ROOT / "data/raw/flight_offer.csv"
RAW_SEGMENT = PROJECT_ROOT / "data/raw/flight_segment.csv"
RAW_LAYOVER = PROJECT_ROOT / "data/raw/flight_layover.csv"
RAW_INSIGHT = PROJECT_ROOT / "data/raw/flight_price_insight.csv"
RAW_HISTORY = PROJECT_ROOT / "data/raw/flight_price_history.csv"

OUTPUT_SUMMARY = PROJECT_ROOT / "data/processed/flight_search_summary_from_raw.csv"
OUTPUT_DATASET = PROJECT_ROOT / "data/processed/flight_model_dataset_from_raw.csv"

# Future target settings
FUTURE_WINDOW_OBS = 28
TARGET_DROP_THRESHOLD = 5000

# Time-series settings
ROLLING_WINDOW = 3
HISTORY_RECENT_POINTS = 5


# =========================
# Helpers
# =========================
def safe_divide(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> np.ndarray:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    return np.where(b_arr != 0, a_arr / b_arr, np.nan)


def parse_datetime_col(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")


def mode_or_nan(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return np.nan
    modes = s.mode()
    return modes.iloc[0] if not modes.empty else s.iloc[0]


def first_or_nan(series: pd.Series):
    s = series.dropna()
    return s.iloc[0] if not s.empty else np.nan


def bucket_hour(hour: float | int | None) -> str:
    if pd.isna(hour):
        return "unknown"
    hour = int(hour)
    if 0 <= hour < 6:
        return "late_night"
    if 6 <= hour < 12:
        return "morning"
    if 12 <= hour < 18:
        return "afternoon"
    return "evening"


def compute_future_min_by_group(group: pd.DataFrame, future_window_obs: int) -> pd.Series:
    prices = group["current_cheapest_price"].to_numpy(dtype=float)
    future_min = np.full(len(group), np.nan, dtype=float)

    for i in range(len(group)):
        start = i + 1
        end = i + 1 + future_window_obs
        window_prices = prices[start:end]
        if len(window_prices) > 0:
            future_min[i] = np.min(window_prices)

    return pd.Series(future_min, index=group.index)


def compute_recent_slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.nan
    x = np.arange(len(values), dtype=float)
    # least-squares slope
    x_mean = x.mean()
    y_mean = values.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return np.nan
    return float(np.sum((x - x_mean) * (values - y_mean)) / denom)


# =========================
# Build search summary
# =========================


def clean_search_table(search: pd.DataFrame) -> pd.DataFrame:
    # Remove exact duplicate rows first
    search = search.drop_duplicates().copy()

    # Some raw exports may contain duplicated search ids that point to conflicting rows.
    # Since downstream tables reference flight_search_id, keep the first occurrence and warn.
    dup_mask = search["flight_search_id"].duplicated(keep=False)
    if dup_mask.any():
        dup_ids = search.loc[dup_mask, "flight_search_id"].drop_duplicates().tolist()
        print(
            f"Warning: duplicated flight_search_id values found in flight_search.csv. "
            f"Keeping first occurrence for ids: {dup_ids}"
        )
        search = search.drop_duplicates(subset=["flight_search_id"], keep="first").copy()

    return search


def build_search_summary() -> pd.DataFrame:
    search = pd.read_csv(RAW_SEARCH)
    offer = pd.read_csv(RAW_OFFER)
    segment = pd.read_csv(RAW_SEGMENT)
    layover = pd.read_csv(RAW_LAYOVER)
    insight = pd.read_csv(RAW_INSIGHT)

    # Parse datetimes
    parse_datetime_col(search, "searched_at")
    parse_datetime_col(search, "outbound_date")
    parse_datetime_col(segment, "departure_time")
    parse_datetime_col(segment, "arrival_time")

    # Core search keys
    search = search.rename(columns={"id": "flight_search_id"})
    search = clean_search_table(search)
    search["route_id"] = (
        search["departure_airport_code"].astype(str)
        + "-"
        + search["arrival_airport_code"].astype(str)
    )
    search["trajectory_id"] = (
        search["route_id"].astype(str)
        + "|"
        + search["outbound_date"].dt.strftime("%Y-%m-%d")
    )
    search["searched_day_of_week"] = search["searched_at"].dt.day_name().str[:3].str.upper()
    search["searched_hour"] = search["searched_at"].dt.hour
    search["outbound_month"] = search["outbound_date"].dt.month
    search["is_weekend_search"] = search["searched_day_of_week"].isin(["SAT", "SUN"]).astype(int)
    search["days_to_departure"] = (
        search["outbound_date"].dt.normalize() - search["searched_at"].dt.normalize()
    ).dt.days

    # Only outbound offers for this project
    if "direction" in offer.columns:
        offer = offer[offer["direction"].astype(str).str.upper() == "OUTBOUND"].copy()

    offer = offer.rename(columns={"id": "flight_offer_id"})

    # Offer summary per search
    offer_group = offer.groupby("flight_search_id")
    offer_summary = offer_group.agg(
        offer_count=("flight_offer_id", "size"),
        current_cheapest_price=("price", "min"),
        avg_price=("price", "mean"),
        median_price=("price", "median"),
        price_std=("price", "std"),
        cheapest_nonstop_price=("price", lambda s: s[offer.loc[s.index, "has_layover"] == False].min() if (offer.loc[s.index, "has_layover"] == False).any() else np.nan),
        nonstop_offer_count=("has_layover", lambda s: int((~s.astype(bool)).sum())),
        cheapest_best_price=("price", lambda s: s[offer.loc[s.index, "is_best"].astype(bool)].min() if offer.loc[s.index, "is_best"].astype(bool).any() else np.nan),
    ).reset_index()

    # Cheapest / best offer ids
    cheapest_offer_map = (
        offer.sort_values(["flight_search_id", "price", "total_duration", "flight_offer_id"])
        .groupby("flight_search_id", as_index=False)
        .first()[["flight_search_id", "flight_offer_id", "price", "total_duration", "has_layover", "layover_count"]]
        .rename(columns={
            "flight_offer_id": "cheapest_offer_id",
            "price": "cheapest_offer_price_check",
            "total_duration": "cheapest_offer_total_duration",
            "has_layover": "cheapest_offer_has_layover",
            "layover_count": "cheapest_offer_layover_count",
        })
    )

    best_offer_map = (
        offer[offer["is_best"].astype(bool)]
        .sort_values(["flight_search_id", "price", "total_duration", "flight_offer_id"])
        .groupby("flight_search_id", as_index=False)
        .first()[["flight_search_id", "flight_offer_id", "price", "total_duration", "has_layover", "layover_count"]]
        .rename(columns={
            "flight_offer_id": "best_offer_id",
            "price": "best_offer_price",
            "total_duration": "best_offer_total_duration",
            "has_layover": "best_offer_has_layover",
            "layover_count": "best_offer_layover_count",
        })
    )

    # Segment-derived features for cheapest offer
    first_segments = (
        segment.sort_values(["flight_offer_id", "segment_order"])
        .groupby("flight_offer_id", as_index=False)
        .first()[[
            "flight_offer_id",
            "departure_time",
            "airline",
            "travel_class",
        ]]
        .rename(columns={
            "departure_time": "first_departure_time",
            "airline": "dominant_airline",
            "travel_class": "travel_class",
        })
    )

    segment_counts = (
        segment.groupby("flight_offer_id", as_index=False)
        .agg(
            segment_count=("id", "size"),
            avg_segment_duration=("duration", "mean"),
        )
    )

    cheapest_segment_features = cheapest_offer_map.merge(
        first_segments, left_on="cheapest_offer_id", right_on="flight_offer_id", how="left"
    ).merge(
        segment_counts, left_on="cheapest_offer_id", right_on="flight_offer_id", how="left", suffixes=("", "_seg")
    )

    cheapest_segment_features["departure_time_bucket"] = cheapest_segment_features["first_departure_time"].dt.hour.map(bucket_hour)

    cheapest_segment_features = cheapest_segment_features[[
        "flight_search_id",
        "dominant_airline",
        "travel_class",
        "segment_count",
        "avg_segment_duration",
        "departure_time_bucket",
    ]]

    # Layover features for cheapest offer
    layover_summary = (
        layover.groupby("flight_offer_id", as_index=False)
        .agg(
            total_layover_duration=("duration", "sum"),
            max_layover_duration=("duration", "max"),
            has_overnight_layover=("is_overnight", lambda s: int(pd.Series(s).astype(bool).any())),
        )
    )

    cheapest_layover_features = cheapest_offer_map.merge(
        layover_summary, left_on="cheapest_offer_id", right_on="flight_offer_id", how="left"
    )[[
        "flight_search_id",
        "total_layover_duration",
        "max_layover_duration",
        "has_overnight_layover",
    ]]

    # Insight
    insight = insight.drop_duplicates(subset=["flight_search_id"], keep="first").copy()
    insight = insight.rename(columns={
        "id": "flight_price_insight_id",
        "lowest_price": "insight_lowest_price",
    })

    # Merge all
    summary = search.merge(offer_summary, on="flight_search_id", how="left")
    summary = summary.merge(cheapest_offer_map, on="flight_search_id", how="left")
    summary = summary.merge(best_offer_map, on="flight_search_id", how="left")
    summary = summary.merge(cheapest_segment_features, on="flight_search_id", how="left")
    summary = summary.merge(cheapest_layover_features, on="flight_search_id", how="left")
    summary = summary.merge(
        insight[[
            "flight_search_id",
            "insight_lowest_price",
            "price_level",
            "typical_price_min",
            "typical_price_max",
        ]],
        on="flight_search_id",
        how="left",
    )

    # Additional summary features
    summary["nonstop_ratio"] = safe_divide(summary["nonstop_offer_count"], summary["offer_count"])
    summary["best_vs_cheapest_price_gap"] = summary["best_offer_price"] - summary["current_cheapest_price"]
    summary["best_vs_cheapest_duration_gap"] = summary["best_offer_total_duration"] - summary["cheapest_offer_total_duration"]
    summary["is_long_haul"] = summary["arrival_airport_code"].isin(["CDG", "JFK", "LHR", "FCO", "MAD", "SYD"]).astype(int)

    # Fill common nullable numerics
    for col in [
        "price_std",
        "nonstop_offer_count",
        "segment_count",
        "avg_segment_duration",
        "total_layover_duration",
        "max_layover_duration",
        "has_overnight_layover",
    ]:
        if col in summary.columns:
            summary[col] = summary[col].fillna(0)

    return summary


# =========================
# Add history-derived features
# =========================
def add_history_features(summary: pd.DataFrame) -> pd.DataFrame:
    history = pd.read_csv(RAW_HISTORY)
    history = history.rename(columns={"id": "flight_price_history_id"})
    history["observed_at"] = pd.to_datetime(history["time_stamp"], unit="s", errors="coerce")
    history = history.rename(columns={"price": "history_price"})

    # Attach route/trajectory to history rows via search id
    hist = history.merge(
        summary[["flight_search_id", "trajectory_id", "route_id", "outbound_date", "searched_at"]],
        on="flight_search_id",
        how="left",
        validate="many_to_one",
    )

    summary = summary.sort_values(["trajectory_id", "searched_at"]).reset_index(drop=True)

    # Per-row history features using same trajectory and only timestamps up to searched_at
    hist_features = []
    grouped_hist = {k: g.sort_values("observed_at").copy() for k, g in hist.groupby("trajectory_id")}

    for row in summary.itertuples(index=False):
        g = grouped_hist.get(row.trajectory_id)
        if g is None:
            hist_features.append({})
            continue

        usable = g[g["observed_at"] <= row.searched_at]
        prices = usable["history_price"].dropna().astype(float).to_numpy()

        if len(prices) == 0:
            hist_features.append({})
            continue

        recent = prices[-HISTORY_RECENT_POINTS:]
        feat = {
            "hist_min_price": float(np.min(prices)),
            "hist_max_price": float(np.max(prices)),
            "hist_mean_price": float(np.mean(prices)),
            "hist_median_price": float(np.median(prices)),
            "hist_std_price": float(np.std(prices, ddof=1)) if len(prices) > 1 else 0.0,
            "hist_price_range": float(np.max(prices) - np.min(prices)),
            "hist_recent_mean": float(np.mean(recent)),
            "hist_recent_std": float(np.std(recent, ddof=1)) if len(recent) > 1 else 0.0,
            "hist_recent_slope": compute_recent_slope(recent),
            "hist_points_available": int(len(prices)),
        }
        hist_features.append(feat)

    hist_feat_df = pd.DataFrame(hist_features)
    summary = pd.concat([summary.reset_index(drop=True), hist_feat_df.reset_index(drop=True)], axis=1)

    # Current vs history relative features
    summary["curr_vs_hist_mean"] = safe_divide(summary["current_cheapest_price"], summary["hist_mean_price"])
    summary["curr_vs_hist_median"] = safe_divide(summary["current_cheapest_price"], summary["hist_median_price"])
    summary["curr_gap_to_hist_min"] = summary["current_cheapest_price"] - summary["hist_min_price"]
    summary["curr_gap_to_hist_max"] = summary["current_cheapest_price"] - summary["hist_max_price"]
    summary["curr_pct_from_hist_min"] = safe_divide(
        summary["current_cheapest_price"] - summary["hist_min_price"], summary["hist_min_price"]
    )

    # Insight-relative
    summary["curr_gap_to_typical_min"] = summary["current_cheapest_price"] - summary["typical_price_min"]
    summary["curr_gap_to_typical_max"] = summary["current_cheapest_price"] - summary["typical_price_max"]
    summary["is_below_typical"] = np.where(
        summary["typical_price_min"].notna() & (summary["current_cheapest_price"] < summary["typical_price_min"]), 1, 0
    )
    summary["is_above_typical"] = np.where(
        summary["typical_price_max"].notna() & (summary["current_cheapest_price"] > summary["typical_price_max"]), 1, 0
    )

    return summary


# =========================
# Build final training dataset
# =========================
def build_training_dataset(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary.sort_values(["trajectory_id", "searched_at", "flight_search_id"]).reset_index(drop=True).copy()

    # Current-price time-series features from directly collected prices only
    grouped_price = df.groupby("trajectory_id")["current_cheapest_price"]

    df["lag_1_price"] = grouped_price.shift(1)
    df["price_change_1"] = df["current_cheapest_price"] - df["lag_1_price"]

    df["rolling_mean_3"] = (
        grouped_price.rolling(window=ROLLING_WINDOW, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["rolling_std_3"] = (
        grouped_price.rolling(window=ROLLING_WINDOW, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
    )
    df["rolling_min_3"] = (
        grouped_price.rolling(window=ROLLING_WINDOW, min_periods=1)
        .min()
        .reset_index(level=0, drop=True)
    )
    df["rolling_max_3"] = (
        grouped_price.rolling(window=ROLLING_WINDOW, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
    )

    df["price_vs_rolling_mean_3"] = df["current_cheapest_price"] - df["rolling_mean_3"]
    df["price_trend_3"] = df["rolling_mean_3"] - df["lag_1_price"]

    # Missing indicators for partial cold-start
    df["has_lag_1"] = df["lag_1_price"].notna().astype(int)
    df["has_rolling_std_3"] = (df.groupby("trajectory_id").cumcount() >= 1).astype(int)

    # Targets from directly collected current prices only
    df["target_future_min_price"] = (
        df.groupby("trajectory_id", group_keys=False)
        .apply(lambda g: compute_future_min_by_group(g, FUTURE_WINDOW_OBS))
        .reset_index(level=0, drop=True)
    )
    df["target_drop_amount"] = df["current_cheapest_price"] - df["target_future_min_price"]
    df["target_future_min_delta"] = df["target_future_min_price"] - df["current_cheapest_price"]
    df["target_wait"] = np.where(df["target_drop_amount"] >= TARGET_DROP_THRESHOLD, 1, 0)

    # Keep only rows with future target
    df_model = df[df["target_future_min_price"].notna()].copy()

    print("\n=== target_wait distribution ===")
    print(df_model["target_wait"].value_counts(dropna=False))
    print("positive ratio:", df_model["target_wait"].mean())

    print("\n=== target_drop_amount summary ===")
    print(df_model["target_drop_amount"].describe())

    print("\n=== zero-drop ratio ===")
    print((df_model["target_drop_amount"] == 0).mean())

    # Practical fills
    fill_zero_cols = [
        "price_std", "rolling_std_3", "hist_std_price", "hist_recent_std",
        "total_layover_duration", "max_layover_duration",
        "best_vs_cheapest_price_gap", "best_vs_cheapest_duration_gap",
    ]
    for col in fill_zero_cols:
        if col in df_model.columns:
            df_model[col] = df_model[col].fillna(0.0)

    # Keep lag values as NaN if absent; tree models can handle them.
    # Final feature / label columns
    feature_cols = [
        # Keys / metadata
        "flight_search_id", "searched_at", "route_id", "trajectory_id",
        "departure_airport_code", "arrival_airport_code", "outbound_date", "currency",

        # Time / calendar
        "days_to_departure", "searched_day_of_week", "searched_hour",
        "outbound_month", "is_weekend_search", "is_long_haul",

        # Current market snapshot
        "offer_count", "current_cheapest_price", "avg_price", "median_price",
        "price_std", "nonstop_offer_count", "nonstop_ratio",
        "cheapest_nonstop_price", "cheapest_best_price",
        "best_offer_price", "best_offer_total_duration", "best_offer_has_layover",
        "best_offer_layover_count",
        "cheapest_offer_total_duration", "cheapest_offer_has_layover",
        "cheapest_offer_layover_count",
        "best_vs_cheapest_price_gap", "best_vs_cheapest_duration_gap",

        # Segment / layover
        "dominant_airline", "travel_class", "segment_count", "avg_segment_duration",
        "departure_time_bucket", "total_layover_duration", "max_layover_duration",
        "has_overnight_layover",

        # Serp insight
        "price_level", "typical_price_min", "typical_price_max",
        "curr_gap_to_typical_min", "curr_gap_to_typical_max",
        "is_below_typical", "is_above_typical",

        # History-derived
        "hist_min_price", "hist_max_price", "hist_mean_price", "hist_median_price",
        "hist_std_price", "hist_price_range", "hist_recent_mean", "hist_recent_std",
        "hist_recent_slope", "hist_points_available",
        "curr_vs_hist_mean", "curr_vs_hist_median",
        "curr_gap_to_hist_min", "curr_gap_to_hist_max", "curr_pct_from_hist_min",

        # Current direct time-series
        "lag_1_price", "price_change_1", "rolling_mean_3", "rolling_std_3",
        "rolling_min_3", "rolling_max_3", "price_vs_rolling_mean_3",
        "price_trend_3", "has_lag_1", "has_rolling_std_3",

        # Targets
        "target_future_min_price", "target_drop_amount",
        "target_future_min_delta", "target_wait",
    ]

    feature_cols = [c for c in feature_cols if c in df_model.columns]
    df_model = df_model[feature_cols].copy()

    return df_model


def main() -> None:
    summary = build_search_summary()
    summary = add_history_features(summary)

    summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")

    dataset = build_training_dataset(summary)
    dataset.to_csv(OUTPUT_DATASET, index=False, encoding="utf-8-sig")

    print(f"Saved summary to: {OUTPUT_SUMMARY}")
    print(f"Saved model dataset to: {OUTPUT_DATASET}")
    print("Summary shape:", summary.shape)
    print("Dataset shape:", dataset.shape)
    if "target_wait" in dataset.columns:
        print("target_wait ratio:", float(dataset["target_wait"].mean()))
    print("Columns:")
    print(dataset.columns.tolist())


if __name__ == "__main__":
    main()
