from __future__ import annotations

from pathlib import Path
from typing import Tuple

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

OUTPUT_SUMMARY = PROJECT_ROOT / "data/processed/flight_search_summary_final.csv"
OUTPUT_DATASET = PROJECT_ROOT / "data/processed/flight_model_dataset_final.csv"

# Target settings
FUTURE_WINDOW_OBS = 28          # about 14 days if collected twice a day
TARGET_DROP_THRESHOLD = 5000    # wait=1 if future min is lower by at least this much

# Feature settings
ROLLING_WINDOW = 3
HISTORY_RECENT_POINTS = 5
LONG_HAUL_ARRIVALS = {"CDG", "JFK", "LHR", "FCO", "MAD", "SYD"}


# =========================
# Helpers
# =========================
def safe_divide(a, b):
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    return np.where(b_arr != 0, a_arr / b_arr, np.nan)


def parse_datetime_col(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")


def bucket_hour(hour) -> str:
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


def clean_search_table(search: pd.DataFrame) -> pd.DataFrame:
    search = search.drop_duplicates().copy()
    dup_mask = search["flight_search_id"].duplicated(keep=False)
    if dup_mask.any():
        dup_ids = search.loc[dup_mask, "flight_search_id"].drop_duplicates().tolist()
        print(
            "Warning: duplicated flight_search_id values found in flight_search.csv. "
            f"Keeping first occurrence for ids: {dup_ids}"
        )
        search = search.drop_duplicates(subset=["flight_search_id"], keep="first").copy()
    return search


# =========================
# Build summary
# =========================
def build_search_summary() -> pd.DataFrame:
    search = pd.read_csv(RAW_SEARCH)
    offer = pd.read_csv(RAW_OFFER)
    segment = pd.read_csv(RAW_SEGMENT)
    layover = pd.read_csv(RAW_LAYOVER)
    insight = pd.read_csv(RAW_INSIGHT)

    parse_datetime_col(search, "searched_at")
    parse_datetime_col(search, "outbound_date")
    parse_datetime_col(segment, "departure_time")

    search = search.rename(columns={"id": "flight_search_id"})
    search = clean_search_table(search)

    # Base ids / calendar
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
    search["is_long_haul"] = search["arrival_airport_code"].isin(LONG_HAUL_ARRIVALS).astype(int)

    # Only outbound offers
    if "direction" in offer.columns:
        offer = offer[offer["direction"].astype(str).str.upper() == "OUTBOUND"].copy()
    offer = offer.rename(columns={"id": "flight_offer_id"})

    offer_summary = (
        offer.groupby("flight_search_id", as_index=False)
        .agg(
            offer_count=("flight_offer_id", "size"),
            current_cheapest_price=("price", "min"),
            nonstop_offer_count=("has_layover", lambda s: int((~s.astype(bool)).sum())),
            cheapest_nonstop_price=("price", lambda s: s[offer.loc[s.index, "has_layover"] == False].min() if (offer.loc[s.index, "has_layover"] == False).any() else np.nan),
        )
    )
    offer_summary["nonstop_ratio"] = safe_divide(
        offer_summary["nonstop_offer_count"], offer_summary["offer_count"]
    )

    cheapest_offer_map = (
        offer.sort_values(["flight_search_id", "price", "total_duration", "flight_offer_id"])
        .groupby("flight_search_id", as_index=False)
        .first()[["flight_search_id", "flight_offer_id", "price", "has_layover"]]
        .rename(columns={
            "flight_offer_id": "cheapest_offer_id",
            "price": "cheapest_offer_price_check",
            "has_layover": "cheapest_offer_has_layover",
        })
    )

    # Segment-based categorical feature for cheapest offer
    first_segments = (
        segment.sort_values(["flight_offer_id", "segment_order"])
        .groupby("flight_offer_id", as_index=False)
        .first()[["flight_offer_id", "departure_time"]]
        .rename(columns={"departure_time": "first_departure_time"})
    )
    cheapest_segment_features = cheapest_offer_map.merge(
        first_segments, left_on="cheapest_offer_id", right_on="flight_offer_id", how="left"
    )
    cheapest_segment_features["departure_time_bucket"] = (
        cheapest_segment_features["first_departure_time"].dt.hour.map(bucket_hour)
    )
    cheapest_segment_features = cheapest_segment_features[[
        "flight_search_id", "departure_time_bucket"
    ]]

    # Insight
    insight = insight.drop_duplicates(subset=["flight_search_id"], keep="first").copy()
    insight = insight[["flight_search_id", "price_level", "typical_price_min", "typical_price_max"]]

    summary = (
        search[[
            "flight_search_id", "searched_at", "route_id", "trajectory_id",
            "departure_airport_code", "arrival_airport_code", "outbound_date",
            "days_to_departure", "searched_day_of_week", "searched_hour",
            "outbound_month", "is_weekend_search", "is_long_haul"
        ]]
        .merge(offer_summary, on="flight_search_id", how="left")
        .merge(cheapest_offer_map[["flight_search_id", "cheapest_offer_has_layover"]], on="flight_search_id", how="left")
        .merge(cheapest_segment_features, on="flight_search_id", how="left")
        .merge(insight, on="flight_search_id", how="left")
    )

    # Derived current-vs-typical features
    summary["curr_gap_to_typical_min"] = summary["current_cheapest_price"] - summary["typical_price_min"]
    summary["curr_gap_to_typical_max"] = summary["current_cheapest_price"] - summary["typical_price_max"]

    return summary


# =========================
# History features
# =========================
def add_history_features(summary: pd.DataFrame) -> pd.DataFrame:
    history = pd.read_csv(RAW_HISTORY)
    history = history.rename(columns={"id": "flight_price_history_id", "price": "history_price"})
    history["observed_at"] = pd.to_datetime(history["time_stamp"], unit="s", errors="coerce")

    hist = history.merge(
        summary[["flight_search_id", "trajectory_id", "searched_at"]],
        on="flight_search_id",
        how="left",
        validate="many_to_one",
    )

    summary = summary.sort_values(["trajectory_id", "searched_at"]).reset_index(drop=True)
    grouped_hist = {k: g.sort_values("observed_at").copy() for k, g in hist.groupby("trajectory_id")}

    hist_features = []
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
        hist_features.append({
            "hist_recent_std": float(np.std(recent, ddof=1)) if len(recent) > 1 else 0.0,
            "hist_recent_slope": compute_recent_slope(recent),
            "hist_mean_price": float(np.mean(prices)),
        })

    hist_feat_df = pd.DataFrame(hist_features)
    summary = pd.concat([summary.reset_index(drop=True), hist_feat_df.reset_index(drop=True)], axis=1)
    summary["curr_vs_hist_mean"] = safe_divide(summary["current_cheapest_price"], summary["hist_mean_price"])
    return summary


# =========================
# Final training dataset
# =========================
def build_training_dataset(summary: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = summary.sort_values(["trajectory_id", "searched_at", "flight_search_id"]).reset_index(drop=True).copy()

    grouped_price = df.groupby("trajectory_id")["current_cheapest_price"]
    df["lag_1_price"] = grouped_price.shift(1)
    df["price_change_1"] = df["current_cheapest_price"] - df["lag_1_price"]
    df["rolling_mean_3"] = grouped_price.rolling(window=ROLLING_WINDOW, min_periods=1).mean().reset_index(level=0, drop=True)
    df["rolling_std_3"] = grouped_price.rolling(window=ROLLING_WINDOW, min_periods=1).std().reset_index(level=0, drop=True)
    df["price_vs_rolling_mean_3"] = df["current_cheapest_price"] - df["rolling_mean_3"]
    df["has_lag_1"] = df["lag_1_price"].notna().astype(int)

    # Targets from directly collected current prices only
    df["target_future_min_price"] = (
        df.groupby("trajectory_id", group_keys=False)
        .apply(lambda g: compute_future_min_by_group(g, FUTURE_WINDOW_OBS))
        .reset_index(level=0, drop=True)
    )
    raw_drop = df["current_cheapest_price"] - df["target_future_min_price"]
    df["target_drop_amount"] = raw_drop.clip(lower=0)
    df["target_future_min_delta"] = df["target_future_min_price"] - df["current_cheapest_price"]
    df["target_wait"] = (df["target_drop_amount"] >= TARGET_DROP_THRESHOLD).astype(int)

    df_model = df[df["target_future_min_price"].notna()].copy()

    # Fill a few truly structural missing values
    for col in ["hist_recent_std", "curr_gap_to_typical_min", "curr_gap_to_typical_max", "rolling_std_3"]:
        if col in df_model.columns:
            df_model[col] = df_model[col].fillna(0.0)

    final_cols = [
        # metadata / keys kept for tracking and time split
        "flight_search_id", "searched_at", "route_id", "trajectory_id",
        "departure_airport_code", "arrival_airport_code", "outbound_date",

        # final selected features
        "days_to_departure",
        "searched_day_of_week",
        "is_weekend_search",
        "is_long_haul",
        "offer_count",
        "nonstop_ratio",
        "cheapest_nonstop_price",
        "cheapest_offer_has_layover",
        "current_cheapest_price",
        "curr_gap_to_typical_min",
        "curr_gap_to_typical_max",
        "hist_recent_std",
        "hist_recent_slope",
        "curr_vs_hist_mean",
        "price_change_1",
        "rolling_std_3",
        "price_vs_rolling_mean_3",

        # targets
        "target_future_min_price",
        "target_drop_amount",
        "target_future_min_delta",
        "target_wait",
    ]

    final_cols = [c for c in final_cols if c in df_model.columns]
    df_model = df_model[final_cols].copy()

    print("\n=== target_wait distribution ===")
    print(df_model["target_wait"].value_counts(dropna=False))
    print("positive ratio:", float(df_model["target_wait"].mean()))

    print("\n=== target_drop_amount summary ===")
    print(df_model["target_drop_amount"].describe())

    print("\n=== zero-drop ratio ===")
    print(float((df_model["target_drop_amount"] == 0).mean()))

    return summary, df_model


def main() -> None:
    summary = build_search_summary()
    summary = add_history_features(summary)
    summary_out, dataset = build_training_dataset(summary)

    OUTPUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary_out.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")
    dataset.to_csv(OUTPUT_DATASET, index=False, encoding="utf-8-sig")

    print(f"Saved summary to: {OUTPUT_SUMMARY}")
    print(f"Saved model dataset to: {OUTPUT_DATASET}")
    print("Summary shape:", summary_out.shape)
    print("Dataset shape:", dataset.shape)
    print("Columns:")
    print(dataset.columns.tolist())


if __name__ == "__main__":
    main()
