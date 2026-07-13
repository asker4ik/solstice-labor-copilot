"""Feature engineering shared by the forecast model and the app.

Leakage rule: every trailing feature applies shift(1) BEFORE rolling, so a row
only ever sees strictly-past data. The two baselines in the ladder (naive
trailing-28d mean, DOW-aware trailing mean) are computed here under the same
rule, so baselines and model are judged on identical information.
"""

import pandas as pd

import config

DAYPART_CODE = {dp: i for i, dp in enumerate(config.DAYPART_ORDER)}
TIER_CODE = {"slow": 0, "core": 1, "flagship": 2}
FORMAT_CODE = {"kiosk": 0, "cafe": 1, "drive_thru": 2}
REGION_CODE = {r: i for i, r in enumerate(config.REGIONS)}
POS_CODE = {"square": 0, "toast": 1}

FEATURES = [
    "daypart_code", "dow", "is_weekend", "day_of_month", "is_payday", "day_index",
    "store_code", "tier_code", "format_code", "region_code", "pos_code",
    "temp_f", "precip_in",
    "trail_7", "trail_28", "trail_dow_4", "dow_ratio",
]
TARGET = "orders"


def build_features(tables: dict) -> pd.DataFrame:
    """demand + stores + weather -> model table at store x business_date x daypart."""
    demand, stores, weather = tables["demand"], tables["stores"], tables["weather"]
    df = demand.merge(
        stores[["store_guid", "tier", "format", "region", "pos_system"]], on="store_guid")
    df = df.merge(weather, on=["region", "business_date"], how="left")

    dt = pd.to_datetime(df["business_date"].astype(str), format="%Y%m%d")
    df["date"] = dt
    df["dow"] = dt.dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["day_of_month"] = dt.dt.day
    df["is_payday"] = df["day_of_month"].isin(config.PAYDAY_DAYS_OF_MONTH).astype(int)

    days = sorted(df["business_date"].unique())
    df["day_index"] = df["business_date"].map({d: i for i, d in enumerate(days)})

    df["daypart_code"] = df["daypart"].map(DAYPART_CODE)
    df["tier_code"] = df["tier"].map(TIER_CODE)
    df["format_code"] = df["format"].map(FORMAT_CODE)
    df["region_code"] = df["region"].map(REGION_CODE)
    df["pos_code"] = df["pos_system"].map(POS_CODE)
    store_codes = {g: i for i, g in enumerate(sorted(df["store_guid"].unique()))}
    df["store_code"] = df["store_guid"].map(store_codes)

    # trailing features — strictly past-only via shift(1)
    df = df.sort_values(["store_guid", "daypart", "business_date"], ignore_index=True)
    g = df.groupby(["store_guid", "daypart"], sort=False)["orders"]
    df["trail_7"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
    df["trail_28"] = g.transform(lambda s: s.shift(1).rolling(28, min_periods=7).mean())
    gd = df.groupby(["store_guid", "daypart", "dow"], sort=False)["orders"]
    df["trail_dow_4"] = gd.transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())
    # DOW index: weekly shape separated from level (standard demand-forecast feature)
    df["dow_ratio"] = df["trail_dow_4"] / df["trail_28"].clip(lower=0.5)

    # the incumbent ladder, on the same past-only information
    df["pred_naive"] = df["trail_28"]      # "gut": trailing 4-week same-daypart average
    df["pred_dow"] = df["trail_dow_4"]     # "smart spreadsheet": same but DOW-aware
    return df
