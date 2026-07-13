"""Component 4 — pilot simulation: treatment vs matched control.

The memo's proof design, in code: take the 15 slow-tier stores (where the gap
concentrates), split 8 treatment / 7 control MATCHED on format, region, and volume.
Treatment stores adopt the feasible plan and capture 50-65% of their gap (change
management is never 100%); controls keep running as-is. The treatment-vs-control
delta over the held-out window IS the measured value — no hand-waving.

Roll-forward: the measured capture rate applied to all 15 slow stores = the year-1
number the memo commits to. Nothing here feeds back into the forecast or the plan.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src import toast_adapter


def assign_groups(stores: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    slow = stores[stores["tier"] == "slow"].copy()
    # match on format/region/volume: sort into strata, alternate T/C down the list
    slow = slow.sort_values(["format", "region", "base_daily_orders"], ignore_index=True)
    labels = []
    flip = bool(rng.integers(0, 2))
    for i in range(len(slow)):
        labels.append("treatment" if (i % 2 == 0) != flip else "control")
    slow["group"] = labels
    # trim/pad to exactly PILOT_TREATMENT_N treatments
    n_t = (slow["group"] == "treatment").sum()
    if n_t != config.PILOT_TREATMENT_N:
        idx = slow.index[slow["group"] == ("treatment" if n_t > config.PILOT_TREATMENT_N else "control")]
        for k in idx[: abs(n_t - config.PILOT_TREATMENT_N)]:
            slow.loc[k, "group"] = "control" if n_t > config.PILOT_TREATMENT_N else "treatment"
    return slow[["store_guid", "location_name", "format", "region",
                 "base_daily_orders", "group"]]


def main(tables: dict | None = None) -> dict:
    if tables is None:
        tables = toast_adapter.load_processed()
    rng = np.random.default_rng(config.SEED + 1)
    plan = pd.read_parquet(Path(config.PROCESSED_DIR) / "labor_plan.parquet")
    groups = assign_groups(tables["stores"], rng)

    p = plan.merge(groups[["store_guid", "group"]], on="store_guid", how="inner")
    n_days = p["business_date"].nunique()
    ann = 365.0 / n_days

    per_store = p.groupby(["store_guid", "group"], as_index=False).agg(
        rev=("net_revenue_real", "sum"), actual=("actual_cost", "sum"),
        feasible=("feasible_cost", "sum"))
    capture = rng.uniform(*config.PILOT_CAPTURE_RANGE, size=len(per_store))
    is_t = (per_store["group"] == "treatment").to_numpy()
    per_store["after"] = np.where(
        is_t,
        per_store["actual"] - capture * (per_store["actual"] - per_store["feasible"]),
        per_store["actual"])
    per_store["pct_before"] = per_store["actual"] / per_store["rev"]
    per_store["pct_after"] = per_store["after"] / per_store["rev"]
    per_store["savings_ann"] = (per_store["actual"] - per_store["after"]) * ann
    per_store = per_store.merge(groups.drop(columns="group"), on="store_guid")

    g = per_store.groupby("group").agg(
        rev=("rev", "sum"), before=("actual", "sum"), after=("after", "sum"))
    t, c = g.loc["treatment"], g.loc["control"]
    t_delta_pts = (t["before"] - t["after"]) / t["rev"]
    c_delta_pts = (c["before"] - c["after"]) / c["rev"]          # 0 by construction
    measured_ann = (t["before"] - t["after"]) * ann
    capture_rate = float((t["before"] - t["after"]) /
                         (t["before"] - per_store.loc[is_t, "feasible"].sum()))

    # roll-forward: measured capture applied to ALL 15 slow stores' feasible gap
    slow_gap_ann = (per_store["actual"].sum() - per_store["feasible"].sum()) * ann
    rollforward_slow = slow_gap_ann * capture_rate

    result = {
        "eval_days": int(n_days),
        "treatment_stores": int(is_t.sum()),
        "control_stores": int((~is_t).sum()),
        "treatment_pct_before": float(t["before"] / t["rev"]),
        "treatment_pct_after": float(t["after"] / t["rev"]),
        "control_pct": float(c["before"] / c["rev"]),
        "treatment_delta_pts": float(t_delta_pts),
        "control_delta_pts": float(c_delta_pts),
        "measured_savings_annualized": float(measured_ann),
        "measured_capture_rate": capture_rate,
        "rollforward_all_slow_annual": float(rollforward_slow),
    }
    per_store.to_parquet(Path(config.PROCESSED_DIR) / "pilot.parquet", index=False)
    (Path(config.MODELS_DIR) / "pilot_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== CHECKPOINT 3b -- PILOT: TREATMENT vs MATCHED CONTROL (slow tier) ===")
    print(f"{result['treatment_stores']} treatment / {result['control_stores']} control, "
          f"matched on format x region x volume; {n_days}-day window, annualized")
    print(f"\n{'':<12} {'labor % before':>15} {'labor % after':>14}")
    print(f"{'treatment':<12} {result['treatment_pct_before']:>15.1%} {result['treatment_pct_after']:>14.1%}")
    print(f"{'control':<12} {result['control_pct']:>15.1%} {result['control_pct']:>14.1%}")
    print(f"\ntreatment-vs-control delta: {result['treatment_delta_pts']:.1%} of revenue")
    print(f"measured pilot savings (8 stores): ${result['measured_savings_annualized'] / 1e6:.2f}M/yr")
    print(f"measured capture of feasible gap:  {result['measured_capture_rate']:.0%}")
    print(f"roll-forward, all 15 slow stores:  ${result['rollforward_all_slow_annual'] / 1e6:.2f}M/yr")
    print("\nper-store (treatment):")
    for _, r in per_store[per_store["group"] == "treatment"].iterrows():
        print(f"  {r['location_name']:<22} {r['format']:<10} "
              f"{r['pct_before']:.1%} -> {r['pct_after']:.1%}   ${r['savings_ann'] / 1e3:,.0f}K/yr")
    return result


if __name__ == "__main__":
    main()
