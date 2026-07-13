"""Component 2 — demand forecast with an honest baseline ladder.

Three predictors of orders at store x business_date x daypart, all evaluated on
the same held-out time window (train days 1-70, test days 71-90 — never a
random split on a time series):

  1. naive      — trailing-28-day same-store-daypart mean ("what Solstice does by gut")
  2. dow_aware  — trailing mean of the last 4 same-day-of-week values
                  (a smarter spreadsheet; ~what Sling Pro's projection does)
  3. rf         — RandomForestRegressor on calendar/store/weather/lag features

The ladder answers "did you even need ML?" with numbers. No LLM anywhere here.
Primary metric is weighted MAPE (sum |err| / sum actual) — plain MAPE explodes
on near-zero evening dayparts and would overstate everyone's error.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import config
from src import features, toast_adapter


def wmape(actual: pd.Series, pred: pd.Series) -> float:
    return float(np.abs(actual - pred).sum() / actual.sum())


def mae(actual: pd.Series, pred: pd.Series) -> float:
    return float(np.abs(actual - pred).mean())


def evaluate(test: pd.DataFrame) -> dict:
    out = {}
    for name, col in [("naive", "pred_naive"), ("dow_aware", "pred_dow"), ("rf", "pred_rf")]:
        m = {"mae": mae(test["orders"], test[col]),
             "wmape": wmape(test["orders"], test[col])}
        m["by_tier"] = {
            t: {"mae": mae(g["orders"], g[col]), "wmape": wmape(g["orders"], g[col])}
            for t, g in test.groupby("tier")
        }
        out[name] = m
    out["improvement"] = {
        "rf_vs_naive_wmape": 1 - out["rf"]["wmape"] / out["naive"]["wmape"],
        "rf_vs_dow_wmape": 1 - out["rf"]["wmape"] / out["dow_aware"]["wmape"],
        "dow_vs_naive_wmape": 1 - out["dow_aware"]["wmape"] / out["naive"]["wmape"],
    }
    return out


def main(tables: dict | None = None) -> dict:
    if tables is None:
        tables = toast_adapter.load_processed()
    df = features.build_features(tables)

    feat_ok = df[features.FEATURES].notna().all(axis=1)
    train = df[(df["day_index"] < config.TRAIN_DAYS) & feat_ok]
    test = df[(df["day_index"] >= config.TRAIN_DAYS) & feat_ok].copy()

    rf = RandomForestRegressor(**config.RF_PARAMS)
    rf.fit(train[features.FEATURES], train[features.TARGET])
    test["pred_rf"] = rf.predict(test[features.FEATURES])

    metrics = evaluate(test)
    imp = (pd.DataFrame({"feature": features.FEATURES,
                         "importance": rf.feature_importances_})
             .sort_values("importance", ascending=False, ignore_index=True))

    models_dir = Path(config.MODELS_DIR)
    models_dir.mkdir(exist_ok=True)
    with open(models_dir / "model.pkl", "wb") as f:
        pickle.dump({"model": rf, "features": features.FEATURES}, f)
    (models_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    imp.to_csv(models_dir / "importances.csv", index=False)

    # test-window predictions for the labor layer + app
    keep = ["store_guid", "business_date", "daypart", "day_index", "dow", "tier",
            "orders", "items", "net_revenue", "prep_minutes",
            "pred_naive", "pred_dow", "pred_rf"]
    test[keep].to_parquet(Path(config.PROCESSED_DIR) / "predictions.parquet", index=False)

    print("\n=== CHECKPOINT 2 — FORECAST LADDER (held-out days 71-90) ===")
    print(f"{'model':<12} {'MAE':>7} {'wMAPE':>8}")
    for name in ("naive", "dow_aware", "rf"):
        print(f"{name:<12} {metrics[name]['mae']:>7.2f} {metrics[name]['wmape']:>8.1%}")
    impv = metrics["improvement"]
    print(f"\nDOW-aware vs naive : {impv['dow_vs_naive_wmape']:+.1%} wMAPE reduction")
    print(f"RF vs naive        : {impv['rf_vs_naive_wmape']:+.1%}")
    print(f"RF vs DOW-aware    : {impv['rf_vs_dow_wmape']:+.1%}")
    print("\nwMAPE by tier (naive -> rf):")
    for t in ("flagship", "core", "slow"):
        n = metrics["naive"]["by_tier"][t]["wmape"]
        r = metrics["rf"]["by_tier"][t]["wmape"]
        print(f"  {t:<9} {n:.1%} -> {r:.1%}")
    print("\nTop feature importances:")
    for _, row in imp.head(8).iterrows():
        print(f"  {row['feature']:<14} {row['importance']:.3f}")
    return metrics


if __name__ == "__main__":
    main()
