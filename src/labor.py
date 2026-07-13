"""Component 3 — labor recommendation + value layer. Deterministic, not ML.

Chain: RF demand forecast (test window, days 71-90)
  -> required labor-hours per store x day x daypart
     (forecast orders x store prep-min/order x service standard)
  -> feasibility constraints (3h shift blocks, 2-person floor, contiguity, guardrails)
  -> feasible plan vs actual -> $ gap -> annualized opportunity
  -> roster resizing story (fewer-but-fuller schedules)

Design points a skeptic will probe, answered in code:
- The SERVICE STANDARD (total labor minutes per drink-prep-minute) is not invented:
  it is calibrated from flagship stores' demonstrated operations on the TRAINING
  window — the tier that already runs at ~target labor %. Every other store is
  measured against operations Solstice itself has proven feasible.
- SCALE: the sim is a ~1:16 scale model of the $140M fleet. One explicit factor
  (REVENUE_SCALE, printed) converts hours/dollars to fleet scale; percentages are
  scale-free. Constraints like "3-hour minimum shift" only make sense at real scale.
- The $ opportunity is computed off the FEASIBLE plan (post-constraints), not the
  theoretical gap. Guardrail: never recommend cutting below forecast-required
  coverage; understaffed peaks are flagged "do not cut — ADD".
- Human-in-the-loop: manager overrides with reason codes (seeded examples) zero out
  the recommendation for that slot; the tool is decision support, not auto-scheduling.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src import toast_adapter

DP_ORDER = config.DAYPART_ORDER
DP_HOURS = {dp: h1 - h0 for dp, (h0, h1) in config.DAYPARTS.items()}


# --------------------------------------------------------------------------- helpers

def revenue_scale(demand: pd.DataFrame) -> float:
    """Explicit sim -> $140M-fleet scale factor (revenue-anchored)."""
    sim_annual = demand["net_revenue"].sum() * 365.0 / config.N_DAYS
    return config.FLEET_ANNUAL_REVENUE / sim_annual


def service_standard(demand: pd.DataFrame, labor: pd.DataFrame,
                     stores: pd.DataFrame, day_index: pd.Series) -> float:
    """Total labor minutes per prep-minute, from the calibration tier's demonstrated
    ops on the TRAINING window only (no peeking at the evaluation period)."""
    anchor = stores.loc[stores["tier"] == config.SERVICE_STANDARD_TIER, "store_guid"]
    train_dates = sorted(demand["business_date"].unique())[:config.TRAIN_DAYS]
    d = demand[demand["store_guid"].isin(anchor) & demand["business_date"].isin(train_dates)]
    l = labor[labor["store_guid"].isin(anchor) & labor["business_date"].isin(train_dates)]
    return float(l["labor_hours"].sum() * 60.0 / d["prep_minutes"].sum())


def store_prep_min_per_order(demand: pd.DataFrame) -> pd.Series:
    """Each store's avg prep-minutes per order (its item mix), training window."""
    train_dates = sorted(demand["business_date"].unique())[:config.TRAIN_DAYS]
    d = demand[demand["business_date"].isin(train_dates)]
    g = d.groupby("store_guid").agg(p=("prep_minutes", "sum"), o=("orders", "sum"))
    return g["p"] / g["o"]


def quantize_day(deltas: np.ndarray, trim_room: np.ndarray) -> np.ndarray:
    """Turn ideal daypart-hour deltas for ONE store-day into a feasible plan:
    contiguous same-sign segments -> whole MIN_SHIFT_HOURS blocks -> distributed
    back proportionally. Trims are capped by trim_room (guardrail + 2-person floor).
    """
    feasible = np.zeros(len(deltas))
    i = 0
    while i < len(deltas):
        sign = np.sign(deltas[i]) if abs(deltas[i]) > 0.25 else 0.0
        if sign == 0:
            i += 1
            continue
        j = i
        while j < len(deltas) and np.sign(deltas[j]) == sign and abs(deltas[j]) > 0.25:
            j += 1
        seg = slice(i, j)
        total = deltas[seg].sum()
        if sign < 0:  # trim: cap by available room above floors/guardrails
            room = trim_room[seg].sum()
            total = -min(-total, room)
        blocks = round(abs(total) / config.MIN_SHIFT_HOURS)
        feas_total = np.sign(total) * blocks * config.MIN_SHIFT_HOURS
        if abs(total) > 1e-9 and blocks > 0:
            feasible[seg] = deltas[seg] / deltas[seg].sum() * abs(feas_total) * np.sign(total)
        i = j
    return feasible


# --------------------------------------------------------------------------- main build

def build_plan(tables: dict) -> tuple[pd.DataFrame, dict]:
    stores, demand, labor = tables["stores"], tables["demand"], tables["labor"]
    preds = pd.read_parquet(Path(config.PROCESSED_DIR) / "predictions.parquet")

    F = revenue_scale(demand)
    std = service_standard(demand, labor, stores, None)
    prep_po = store_prep_min_per_order(demand)

    plan = preds.drop(columns=["tier"]).merge(
        labor, on=["store_guid", "business_date", "daypart"], how="left")
    plan = plan.merge(
        stores[["store_guid", "store_name", "location_name", "tier", "format",
                "region", "pos_system", "hourly_wage_avg", "injected_labor_pct"]],
        on="store_guid")
    plan["prep_min_per_order"] = plan["store_guid"].map(prep_po)
    plan["window_h"] = plan["daypart"].map(DP_HOURS)

    # real-scale quantities (hours x F; wages are already real)
    plan["actual_h"] = plan["labor_hours"] * F
    plan["required_h"] = plan["pred_rf"] * F * plan["prep_min_per_order"] * std / 60.0
    plan["floor_h"] = config.MIN_HEADS_OPEN * plan["window_h"]
    plan["recommended_h"] = plan[["required_h", "floor_h"]].max(axis=1)
    plan["delta_h"] = plan["recommended_h"] - plan["actual_h"]
    # guardrail: cutting below required coverage is never recommended
    plan["do_not_cut"] = plan["actual_h"] < plan["required_h"]
    plan["trim_room"] = (plan["actual_h"] - plan["recommended_h"]).clip(lower=0.0)

    # feasibility quantization per store-day (3h blocks, contiguity, guardrail caps)
    plan = plan.sort_values(["store_guid", "business_date", "daypart"],
                            key=lambda s: s.map({dp: i for i, dp in enumerate(DP_ORDER)})
                            if s.name == "daypart" else s, ignore_index=True)
    feas = np.zeros(len(plan))
    actual_h = plan["actual_h"].to_numpy(dtype=float)
    rev_real = (plan["net_revenue"] * F).to_numpy(dtype=float)
    wage = plan["hourly_wage_avg"].to_numpy(dtype=float)
    for _, idx in plan.groupby(["store_guid", "business_date"]).indices.items():
        idx = np.sort(idx)
        f = quantize_day(plan["delta_h"].values[idx].astype(float),
                         plan["trim_room"].values[idx].astype(float))
        # store-day plan floor: never plan below PLAN_FLOOR_PCT labor — the edge of
        # demonstrated operations. Scale trims back pro-rata if they'd cross it.
        day_floor_h = config.PLAN_FLOOR_PCT * rev_real[idx].sum() / wage[idx].mean()
        trims = f[f < 0].sum()
        max_trim = actual_h[idx].sum() + f[f > 0].sum() - day_floor_h
        if trims < 0 and -trims > max(0.0, max_trim):
            f[f < 0] *= max(0.0, max_trim) / -trims
        feas[idx] = f
    plan["feasible_delta_h"] = feas
    plan["feasible_h"] = plan["actual_h"] + plan["feasible_delta_h"]
    plan["net_revenue_real"] = plan["net_revenue"] * F
    plan["actual_cost"] = plan["actual_h"] * plan["hourly_wage_avg"]
    plan["feasible_cost"] = plan["feasible_h"] * plan["hourly_wage_avg"]
    plan["savings_day"] = plan["actual_cost"] - plan["feasible_cost"]
    return plan, {"F": F, "service_standard": std}


def seed_overrides(plan: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """A few seeded manager overrides (human-in-the-loop): the override wins, the
    recommendation for that slot is not counted in the value math."""
    trims = plan[(plan["feasible_delta_h"] < -1.0)]
    picks = trims.groupby("store_guid").head(1).sample(3, random_state=config.SEED)
    notes = [
        ("local_event", "Street festival on this block — keeping current staffing."),
        ("training", "Two new hires shadowing this week; extra coverage intentional."),
        ("staffing_constraint", "No available part-timer to split this block yet."),
    ]
    rows = []
    for (code, note), (_, r) in zip(notes, picks.iterrows()):
        rows.append({
            "store_guid": r["store_guid"], "location_name": r["location_name"],
            "business_date": r["business_date"], "daypart": r["daypart"],
            "reason_code": code, "note": note,
        })
    return pd.DataFrame(rows)


def apply_overrides(plan: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
    key = ["store_guid", "business_date", "daypart"]
    plan = plan.merge(overrides[key + ["reason_code"]], on=key, how="left")
    hit = plan["reason_code"].notna()
    plan.loc[hit, ["feasible_delta_h"]] = 0.0
    plan.loc[hit, "feasible_h"] = plan.loc[hit, "actual_h"]
    plan.loc[hit, "feasible_cost"] = plan.loc[hit, "actual_cost"]
    plan.loc[hit, "savings_day"] = 0.0
    plan["overridden"] = hit
    return plan


def roster_plan(plan: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """Weekly hours -> roster story: fewer-but-fuller schedules, never below the
    20h floor, capped at 40h. CURRENT_AVG_WEEKLY_HOURS is a labeled assumption."""
    n_days = plan["business_date"].nunique()
    g = plan.groupby(["store_guid", "location_name", "tier"], as_index=False).agg(
        actual_h=("actual_h", "sum"), feasible_h=("feasible_h", "sum"))
    g["weekly_actual_h"] = g["actual_h"] / n_days * 7
    g["weekly_feasible_h"] = g["feasible_h"] / n_days * 7
    g["heads_now"] = g["weekly_actual_h"] / config.CURRENT_AVG_WEEKLY_HOURS
    g["heads_reco"] = g["weekly_feasible_h"] / config.PREFERRED_WEEKLY_HOURS
    g["avg_hours_now"] = config.CURRENT_AVG_WEEKLY_HOURS
    g["avg_hours_reco"] = (g["weekly_feasible_h"] / g["heads_reco"].round()).clip(
        config.MIN_WEEKLY_HOURS_PER_EMP, config.MAX_WEEKLY_HOURS)
    return g


def action_plan(plan: pd.DataFrame) -> pd.DataFrame:
    """Manager-facing shift changes: avg feasible delta by store x weekday/weekend x
    daypart -> contiguous segments -> '+/- N x 3h shift' rows with flags."""
    dts = pd.to_datetime(plan["business_date"].astype(str), format="%Y%m%d")
    plan = plan.assign(day_type=np.where(dts.dt.dayofweek >= 5, "weekend", "weekday"))
    g = (plan.groupby(["store_guid", "location_name", "tier", "day_type", "daypart"])
             .agg(delta=("feasible_delta_h", "mean"), flag=("do_not_cut", "mean"),
                  wage=("hourly_wage_avg", "first")).reset_index())
    rows = []
    for (sg, loc, tier, dt_), grp in g.groupby(["store_guid", "location_name", "tier", "day_type"]):
        grp = grp.set_index("daypart").reindex(DP_ORDER).reset_index()
        deltas = grp["delta"].fillna(0.0).values
        i = 0
        while i < len(DP_ORDER):
            sign = np.sign(deltas[i]) if abs(deltas[i]) > 0.5 else 0.0
            if sign == 0:
                i += 1
                continue
            j = i
            while j < len(DP_ORDER) and np.sign(deltas[j]) == sign and abs(deltas[j]) > 0.5:
                j += 1
            total = deltas[i:j].sum()
            blocks = max(1, round(abs(total) / config.MIN_SHIFT_HOURS))
            span = DP_ORDER[i] if j - i == 1 else f"{DP_ORDER[i]}–{DP_ORDER[j - 1]}"
            verb = "ADD" if sign > 0 else "TRIM"
            flagged = bool(grp["flag"].iloc[i:j].max() > 0.3) and sign > 0
            wage = grp["wage"].iloc[0]
            days = 5 if dt_ == "weekday" else 2
            rows.append({
                "store_guid": sg, "location_name": loc, "tier": tier, "day_type": dt_,
                "dayparts": span,
                "action": f"{verb} {blocks} x {config.MIN_SHIFT_HOURS:.0f}h shift"
                          f"{'s' if blocks > 1 else ''}",
                "hours_per_day": round(float(total), 1),
                "weekly_dollars": round(float(-total) * wage * days, 0),
                "service_flag": "do not cut — understaffed at peak" if flagged else "",
            })
            i = j
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- value math

def value_summary(plan: pd.DataFrame, stores: pd.DataFrame, meta: dict) -> dict:
    n_days = plan["business_date"].nunique()
    ann = 365.0 / n_days
    rev = plan["net_revenue_real"].sum()
    actual_cost = plan["actual_cost"].sum()
    feasible_cost = plan["feasible_cost"].sum()
    fleet_actual_pct = actual_cost / rev
    fleet_feasible_pct = feasible_cost / rev
    theoretical_gap = (fleet_actual_pct - config.TARGET_LABOR_PCT) * config.FLEET_ANNUAL_REVENUE
    feasible_savings_ann = (actual_cost - feasible_cost) * ann

    by_tier = plan.groupby("tier").agg(
        rev=("net_revenue_real", "sum"), a=("actual_cost", "sum"), f=("feasible_cost", "sum"))
    tier_detail = {
        t: {"actual_pct": r["a"] / r["rev"], "feasible_pct": r["f"] / r["rev"],
            "savings_ann": (r["a"] - r["f"]) * ann}
        for t, r in by_tier.iterrows()
    }
    per_store = plan.groupby(["store_guid", "location_name", "tier"], as_index=False).agg(
        rev=("net_revenue_real", "sum"), a=("actual_cost", "sum"), f=("feasible_cost", "sum"),
        flags=("do_not_cut", "sum"))
    per_store["actual_pct"] = per_store["a"] / per_store["rev"]
    per_store["feasible_pct"] = per_store["f"] / per_store["rev"]
    per_store["savings_ann"] = (per_store["a"] - per_store["f"]) * ann

    # SYSTEMATIC understaffing: window-average actual below window-average required
    # for a store x daypart (the day-level do_not_cut guardrail still blocks daily
    # cuts, but day-level flags are Poisson-noisy; the narrative flag is systematic)
    sysf = plan.groupby(["store_guid", "tier", "daypart"], as_index=False).agg(
        a=("actual_h", "mean"), r=("required_h", "mean"))
    sysf["understaffed"] = sysf["a"] < sysf["r"]
    flag_mix = (sysf[sysf["understaffed"]].groupby(["tier", "daypart"]).size()
                    .sort_values(ascending=False).head(5))
    sys_flags = sysf

    return {
        "revenue_scale_factor": meta["F"],
        "service_standard_min_per_prep_min": meta["service_standard"],
        "eval_days": int(n_days),
        "fleet_actual_labor_pct": fleet_actual_pct,
        "fleet_feasible_labor_pct": fleet_feasible_pct,
        "theoretical_gap_to_28_annual": theoretical_gap,
        "feasible_savings_annual": feasible_savings_ann,
        "by_tier": tier_detail,
        "per_store": per_store,
        "flag_mix": flag_mix,
        "sys_flags": sys_flags,
        "labor_hours_per_store_day_real": plan["actual_h"].sum() / n_days / plan["store_guid"].nunique(),
    }


def main(tables: dict | None = None) -> dict:
    if tables is None:
        tables = toast_adapter.load_processed()
    rng = np.random.default_rng(config.SEED)
    plan, meta = build_plan(tables)
    overrides = seed_overrides(plan, rng)
    plan = apply_overrides(plan, overrides)
    actions = action_plan(plan)
    roster = roster_plan(plan, tables["stores"])
    summary = value_summary(plan, tables["stores"], meta)

    out = Path(config.PROCESSED_DIR)
    plan.to_parquet(out / "labor_plan.parquet", index=False)
    actions.to_parquet(out / "action_plan.parquet", index=False)
    roster.to_parquet(out / "roster_plan.parquet", index=False)
    overrides.to_parquet(out / "overrides.parquet", index=False)
    summary["per_store"].to_parquet(out / "store_value.parquet", index=False)
    summary["sys_flags"].to_parquet(out / "sys_flags.parquet", index=False)
    slim = {k: v for k, v in summary.items() if k not in ("per_store", "flag_mix", "sys_flags")}
    (Path(config.MODELS_DIR) / "value_summary.json").write_text(
        json.dumps(slim, indent=2, default=float), encoding="utf-8")

    print("\n=== CHECKPOINT 3a -- LABOR PLAN + VALUE (real scale, eval window) ===")
    print(f"scale factor sim->fleet: x{summary['revenue_scale_factor']:.1f}   "
          f"service standard: {summary['service_standard_min_per_prep_min']:.2f} labor-min/prep-min "
          f"(calibrated from {config.SERVICE_STANDARD_TIER})")
    print(f"labor-hours per store-day (real scale): {summary['labor_hours_per_store_day_real']:.0f}")
    print(f"\nfleet labor pct: actual {summary['fleet_actual_labor_pct']:.1%} -> "
          f"feasible plan {summary['fleet_feasible_labor_pct']:.1%} (target 28%)")
    print(f"theoretical gap to 28%:  ${summary['theoretical_gap_to_28_annual'] / 1e6:.1f}M / yr")
    print(f"FEASIBLE opportunity:    ${summary['feasible_savings_annual'] / 1e6:.2f}M / yr "
          f"(post-constraints, guardrails, overrides)")
    print("\nby tier (actual -> feasible, annual $):")
    for t in ("flagship", "core", "slow"):
        d = summary["by_tier"][t]
        print(f"  {t:<9} {d['actual_pct']:.1%} -> {d['feasible_pct']:.1%}   "
              f"${d['savings_ann'] / 1e6:.2f}M")
    print("\nsystematically understaffed store-dayparts (tier x daypart, of 60 stores):")
    for (t, dp), n in summary["flag_mix"].items():
        print(f"  {t:<9} {dp:<13} {n} stores")
    top = summary["per_store"].sort_values("savings_ann", ascending=False).head(5)
    print("\ntop-5 store opportunities ($/yr):")
    for _, r in top.iterrows():
        print(f"  {r['location_name']:<22} {r['tier']:<9} "
              f"{r['actual_pct']:.1%} -> {r['feasible_pct']:.1%}   ${r['savings_ann'] / 1e3:,.0f}K")
    print(f"\nseeded manager overrides: {len(overrides)} (reason-coded, excluded from value)")
    return summary


if __name__ == "__main__":
    main()
