"""Toast adapter — the ONLY module that knows Toast's field names.

Reads the raw Toast-shaped NDJSON extract under data/toast_extract/ and flattens
it into clean internal dataframes (snake_case, one grain per table). Everything
downstream (features, forecast, labor, pilot, app) speaks only the internal names.

In production you swap this module's *source* — synthetic files -> live Toast API
calls — and nothing downstream changes. The 15 Square-flagged stores would get a
second, equivalent square_adapter.py normalizing Square's schema (order.line_items[],
snake_case, location_id, Timecards) into these same tables.

Outputs (cached to data/processed/*.parquet by build_all):
  stores   — one row per store (guid, tier, format, region, pos, wages, labor pcts)
  menu     — one row per item (guid, name, category, price, prep_seconds)
  demand   — store x business_date x daypart: orders, items, net_revenue, prep_minutes
  labor    — store x business_date x daypart: labor_hours, labor_cost
  weather  — region x business_date: temp_f, precip_in
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import config


# --------------------------------------------------------------------------- raw readers

def _read_ndjson(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _parse_toast_ts_to_local(ts: str) -> datetime:
    """Toast ISO 8601 UTC ('...+0000') -> store-local wall clock (fixed offset)."""
    utc = datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")
    return utc + timedelta(hours=config.UTC_OFFSET_HOURS)


def _daypart_of(local_dt: datetime) -> str:
    h = local_dt.hour
    for dp, (h0, h1) in config.DAYPARTS.items():
        if h0 <= h < h1:
            return dp
    return "evening" if h >= config.DAYPARTS["evening"][0] else "early"


# --------------------------------------------------------------------------- tables

def load_stores() -> pd.DataFrame:
    rows = []
    for r in _read_ndjson(Path(config.EXTRACT_DIR) / "config" / "restaurants.ndjson"):
        syn = r["_synthetic"]
        rows.append({
            "store_guid": r["guid"],
            "store_name": r["general"]["name"],
            "location_name": r["general"]["locationName"],
            "tier": syn["tier"],
            "format": syn["format"],
            "region": syn["region"],
            "pos_system": syn["sourceSystem"],
            "base_daily_orders": syn["baseDailyOrders"],
            "target_labor_pct": syn["targetLaborPct"],
            "injected_labor_pct": syn["actualLaborPct"],
            "hourly_wage_avg": syn["hourlyWageAvg"],
        })
    return pd.DataFrame(rows)


def load_menu() -> pd.DataFrame:
    rows = []
    for m in _read_ndjson(Path(config.EXTRACT_DIR) / "config" / "menus.ndjson"):
        rows.append({
            "item_guid": m["guid"],
            "item_name": m["name"],
            "category": m["salesCategory"]["name"],
            "price": m["price"],
            "prep_seconds": m["_synthetic"]["prepSeconds"],
        })
    return pd.DataFrame(rows)


def load_employees() -> pd.DataFrame:
    """Roster per store. The employees endpoint is restaurant-scoped, so the store
    guid comes from the pull's filename (mirrors how a per-restaurant API pull lands)."""
    rows = []
    for path in sorted((Path(config.EXTRACT_DIR) / "config").glob("employees_*.ndjson")):
        store_guid = path.stem.replace("employees_", "")
        for e in _read_ndjson(path):
            rows.append({
                "store_guid": store_guid,
                "employee_guid": e["guid"],
                "first_name": e["firstName"],
                "last_name": e["lastName"],
                "wage": e["wageOverrides"][0]["wage"],
                "job_guid": e["jobReferences"][0]["guid"],
            })
    return pd.DataFrame(rows)


def load_demand(menu: pd.DataFrame) -> pd.DataFrame:
    """Flatten Order -> checks -> selections into store x date x daypart aggregates.

    Voided selections are dropped. Daypart is derived from openedDate (Toast has
    no daypart field). net_revenue uses pre-tax check amounts.
    """
    prep = dict(zip(menu["item_guid"], menu["prep_seconds"]))
    rows = []
    for path in sorted((Path(config.EXTRACT_DIR) / "orders").glob("orders_*.ndjson")):
        for o in _read_ndjson(path):
            local = _parse_toast_ts_to_local(o["openedDate"])
            dp = _daypart_of(local)
            n_items = 0
            prep_sec = 0.0
            net = 0.0
            loyal = 0
            for check in o["checks"]:
                net += check["amount"]
                if check.get("appliedLoyaltyInfo"):
                    loyal = 1
                for sel in check["selections"]:
                    if sel["voided"]:
                        continue
                    n_items += sel["quantity"]
                    prep_sec += prep.get(sel["item"]["guid"], 60) * sel["quantity"]
            rows.append((o["restaurantGuid"], int(o["businessDate"]), dp,
                         n_items, prep_sec, net, loyal))
    df = pd.DataFrame(rows, columns=["store_guid", "business_date", "daypart",
                                     "items", "prep_seconds", "net_revenue", "loyalty"])
    agg = (df.groupby(["store_guid", "business_date", "daypart"], as_index=False)
             .agg(orders=("items", "size"), items=("items", "sum"),
                  prep_minutes=("prep_seconds", lambda s: s.sum() / 60.0),
                  net_revenue=("net_revenue", "sum"),
                  loyalty_orders=("loyalty", "sum")))
    return _complete_grid(agg)


def _complete_grid(agg: pd.DataFrame) -> pd.DataFrame:
    """Ensure every store x date x daypart combination exists (zeros where closed/quiet)."""
    stores = agg["store_guid"].unique()
    dates = agg["business_date"].unique()
    idx = pd.MultiIndex.from_product(
        [stores, dates, config.DAYPART_ORDER],
        names=["store_guid", "business_date", "daypart"])
    out = (agg.set_index(["store_guid", "business_date", "daypart"])
              .reindex(idx, fill_value=0).reset_index())
    return out.sort_values(["store_guid", "business_date", "daypart"], ignore_index=True)


def load_labor() -> tuple[pd.DataFrame, pd.DataFrame]:
    """TimeEntries -> (store x date x daypart labor hours & cost,
                       employee x daypart historical hours).

    Each entry's clock time is allocated to dayparts by timestamp overlap, so real
    multi-daypart shifts from the live API would flow through unchanged. The
    employee-level table powers "coverage candidates" in the app (who historically
    works each daypart) — assignment itself stays in Toast/Square scheduling.
    """
    dp_windows = {dp: (h0 * 3600, h1 * 3600) for dp, (h0, h1) in config.DAYPARTS.items()}
    rows = []
    for path in sorted((Path(config.EXTRACT_DIR) / "labor").glob("timeEntries_*.ndjson")):
        for e in _read_ndjson(path):
            t_in = _parse_toast_ts_to_local(e["inDate"])
            t_out = _parse_toast_ts_to_local(e["outDate"])
            day0 = t_in.replace(hour=0, minute=0, second=0)
            s_in = (t_in - day0).total_seconds()
            s_out = (t_out - day0).total_seconds()
            wage = e["hourlyWage"]
            emp = e["employeeReference"]["guid"]
            for dp, (w0, w1) in dp_windows.items():
                overlap = max(0.0, min(s_out, w1) - max(s_in, w0))
                if overlap > 0:
                    hours = overlap / 3600.0
                    rows.append((e["restaurantGuid"], int(e["businessDate"]), dp,
                                 emp, hours, hours * wage))
    df = pd.DataFrame(rows, columns=["store_guid", "business_date", "daypart",
                                     "employee_guid", "labor_hours", "labor_cost"])
    agg = (df.groupby(["store_guid", "business_date", "daypart"], as_index=False)
             .agg(labor_hours=("labor_hours", "sum"), labor_cost=("labor_cost", "sum")))
    emp_dp = (df.groupby(["store_guid", "employee_guid", "daypart"], as_index=False)
                .agg(hours=("labor_hours", "sum"), days=("business_date", "nunique")))
    return _complete_grid(agg), emp_dp


def load_weather() -> pd.DataFrame:
    df = pd.read_csv(config.WEATHER_CSV)
    df["business_date"] = df["businessDate"].astype(int)
    return df[["region", "business_date", "temp_f", "precip_in"]]


# --------------------------------------------------------------------------- build + validate

TABLE_NAMES = ("stores", "menu", "employees", "demand", "labor", "labor_by_employee", "weather")


def build_all(write: bool = True) -> dict:
    stores = load_stores()
    menu = load_menu()
    employees = load_employees()
    demand = load_demand(menu)
    labor, labor_by_employee = load_labor()
    weather = load_weather()
    tables = {"stores": stores, "menu": menu, "employees": employees, "demand": demand,
              "labor": labor, "labor_by_employee": labor_by_employee, "weather": weather}
    if write:
        out = Path(config.PROCESSED_DIR)
        out.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            df.to_parquet(out / f"{name}.parquet", index=False)
    return tables


def load_processed() -> dict:
    out = Path(config.PROCESSED_DIR)
    return {name: pd.read_parquet(out / f"{name}.parquet") for name in TABLE_NAMES}


def validation_summary(tables: dict) -> None:
    """CHECKPOINT 1 gate: is the injected structure visibly present?"""
    stores, demand, labor = tables["stores"], tables["demand"], tables["labor"]
    merged = demand.merge(labor, on=["store_guid", "business_date", "daypart"])
    merged = merged.merge(stores[["store_guid", "tier"]], on="store_guid")

    print("\n=== CHECKPOINT 1 — VALIDATION SUMMARY (adapter output) ===")
    rev, cost = merged["net_revenue"].sum(), merged["labor_cost"].sum()
    print(f"\nFleet labor pct (target ~32%): {cost / rev:.1%}")
    by_tier = merged.groupby("tier").agg(rev=("net_revenue", "sum"), cost=("labor_cost", "sum"))
    for t in ("flagship", "core", "slow"):
        r = by_tier.loc[t]
        print(f"  {t:<9} {r['cost'] / r['rev']:.1%}")

    print("\nDaypart share of orders (morning_peak should dominate):")
    dp = merged.groupby("daypart")["orders"].sum()
    dp = dp.reindex(config.DAYPART_ORDER)
    for name, v in dp.items():
        print(f"  {name:<13} {v / dp.sum():.1%}")

    print("\nDay-of-week order index (fleet, 1.00 = mean):")
    dts = pd.to_datetime(merged["business_date"].astype(str), format="%Y%m%d")
    dow = merged.assign(dow=dts.dt.dayofweek).groupby("dow")["orders"].sum()
    dow = dow / dow.mean()
    for i, name in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
        print(f"  {name} {dow[i]:.2f}")

    print("\nLabor-vs-demand alignment at morning_peak (share of day's labor vs share of day's orders):")
    day_tot = merged.groupby(["store_guid", "business_date"]).agg(
        d_orders=("orders", "sum"), d_hours=("labor_hours", "sum")).reset_index()
    mp = merged[merged["daypart"] == "morning_peak"].merge(
        day_tot, on=["store_guid", "business_date"])
    mp = mp[(mp["d_orders"] > 0) & (mp["d_hours"] > 0)].copy()
    mp["order_share"] = mp["orders"] / mp["d_orders"]
    mp["labor_share"] = mp["labor_hours"] / mp["d_hours"]
    align = mp.groupby("tier")[["order_share", "labor_share"]].mean()
    for t in ("flagship", "core", "slow"):
        r = align.loc[t]
        print(f"  {t:<9} orders {r['order_share']:.0%} of day  vs  labor {r['labor_share']:.0%} of day")

    n_rows = demand[demand["orders"] > 0].shape[0]
    print(f"\nOrders total: {int(demand['orders'].sum()):,}   "
          f"line items: {int(demand['items'].sum()):,}   "
          f"model grain rows: {len(demand):,} (non-zero {n_rows:,})")
    print(f"Loyalty attach: {demand['loyalty_orders'].sum() / demand['orders'].sum():.0%} of orders")


if __name__ == "__main__":
    validation_summary(build_all())
