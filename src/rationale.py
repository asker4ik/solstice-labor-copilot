"""Deterministic forecast rationale — NO LLM.

Explains why the labor model's plan differs from the "historic average" plan a
manager would build from their gut/spreadsheet. Attribution is arithmetic:

- weather contribution  = prediction - prediction(with seasonal-normal temp, no rain)
- payday contribution   = prediction - prediction(is_payday=0)
- weekday pattern       = trailing same-weekday mean vs trailing overall mean (data, not model)
- momentum              = trailing 7-day mean vs trailing 28-day mean (data, not model)

Because every number is a counterfactual re-prediction or a direct data
comparison, the rationale can't hallucinate — it is exactly as right or wrong
as the model itself, which is the honest thing to show a skeptical manager.
"""

from datetime import date

import numpy as np
import pandas as pd

import config
from src import features

DOW_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _seasonal_temp(region: str, day_index: int) -> float:
    t0, t1 = config.WEATHER_TEMP_RANGE[region]
    return t0 + (t1 - t0) * day_index / (config.N_DAYS - 1)


def explain_day(model, feat_df: pd.DataFrame, store_guid: str, business_date: int,
                region: str) -> pd.DataFrame:
    """Per-daypart attribution for one store-day. Returns a dataframe with the
    RF prediction, the DOW-aware historic baseline, and named contributions."""
    rows = feat_df[(feat_df["store_guid"] == store_guid)
                   & (feat_df["business_date"] == business_date)].copy()
    rows = rows.dropna(subset=features.FEATURES)
    if rows.empty:
        return pd.DataFrame()
    X = rows[features.FEATURES]
    base = model.predict(X)

    # counterfactual: seasonal-normal weather (no rain, typical temp for the date)
    X_wx = X.copy()
    X_wx["precip_in"] = 0.0
    X_wx["temp_f"] = [_seasonal_temp(region, di) for di in rows["day_index"]]
    wx_delta = base - model.predict(X_wx)

    # counterfactual: not a payday
    X_pay = X.copy()
    X_pay["is_payday"] = 0
    pay_delta = base - model.predict(X_pay)

    out = rows[["daypart", "business_date", "dow", "orders"]].copy()
    out["pred_rf"] = base
    out["pred_hist"] = rows["pred_dow"].values          # DOW-aware trailing average
    out["pred_naive"] = rows["pred_naive"].values       # plain trailing average
    out["weather_delta"] = wx_delta
    out["payday_delta"] = pay_delta
    out["dow_pattern"] = rows["trail_dow_4"].values - rows["trail_28"].values
    out["momentum"] = rows["trail_7"].values - rows["trail_28"].values
    out["temp_f"] = rows["temp_f"].values
    out["precip_in"] = rows["precip_in"].values
    return out


def bullets(row: pd.Series) -> list[str]:
    """Manager-readable rationale lines for one daypart row from explain_day.

    Thresholds are relative to the daypart's own volume, so quiet evenings don't
    spam noise and busy peaks don't hide real drivers.
    """
    dow_name = DOW_NAMES[int(row["dow"])]
    base = max(float(row["pred_hist"]), 1.0)
    minor = max(0.04 * base, 2.0)      # driver worth mentioning
    out = []
    diff = row["pred_rf"] - row["pred_hist"]
    out.append(
        f"Your 4-week {dow_name} average says ~{row['pred_hist']:.0f} orders; "
        f"the model forecasts {row['pred_rf']:.0f} ({diff:+.0f})."
    )
    if abs(row["dow_pattern"]) >= minor:
        hotter = "run hotter than" if row["dow_pattern"] > 0 else "run quieter than"
        out.append(
            f"{dow_name}s {hotter} your average day here "
            f"({row['dow_pattern']:+.0f} orders in this daypart)."
        )
    if abs(row["weather_delta"]) >= minor:
        wx_bits = []
        if row["precip_in"] >= 0.05:
            wx_bits.append(f"{row['precip_in']:.1f}\" rain")
        wx_bits.append(f"{row['temp_f']:.0f}°F")
        out.append(
            f"Weather ({', '.join(wx_bits)}) moves the forecast {row['weather_delta']:+.0f} "
            f"orders vs a typical day this time of year."
        )
    if abs(row["payday_delta"]) >= minor:
        out.append(f"Payday bump: {row['payday_delta']:+.0f} orders.")
    if abs(row["momentum"]) >= minor:
        trend_word = "up" if row["momentum"] > 0 else "down"
        out.append(
            f"Recent momentum: last 7 days trending {trend_word} "
            f"({row['momentum']:+.0f} vs your 4-week average)."
        )
    return out


def test_window_dates(feat_df: pd.DataFrame) -> list[int]:
    d = feat_df[feat_df["day_index"] >= config.TRAIN_DAYS]
    return sorted(d["business_date"].unique().tolist())
