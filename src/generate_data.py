"""Component 1 — synthetic data generator.

Writes a Toast-API-shaped extract to data/toast_extract/: real Toast object and
field names (camelCase, `guid` ids), NDJSON one object per line, nesting
Order -> checks -> selections/payments, organized like a paginated pull
(one file per restaurant per businessDate). Synthetic-only attributes live under
a namespaced `_synthetic:{}` block so they never masquerade as Toast fields, and
EXTRACT_MANIFEST.json declares `"synthetic": true`.

The *content* baked into those objects carries the learnable structure the model
must recover: daypart curve, day-of-week curve, weather, payday, mild trend —
plus the injected labor misalignment (slow-tier stores over-staffed off-peak and
under-staffed at morning peak).

Weather is written separately to data/weather_daily.csv — it is not Toast data.
"""

import json
import shutil
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

import config

DAYPART_LIST = config.DAYPART_ORDER
CATEGORIES = ["espresso", "brewed", "cold", "food", "bakery"]

STORE_NAMES = {
    "Texas": [
        "Lower Greenville", "Bishop Arts", "Deep Ellum", "McKinney Ave", "Legacy West",
        "Frisco Star", "Southlake Square", "Fort Worth Magnolia", "South Congress",
        "The Domain", "Houston Heights", "Montrose", "San Antonio Pearl", "Katy Grand",
        "Woodlands Market",
    ],
    "Southeast": [
        "Ponce City", "Decatur Square", "Buckhead Village", "12 South", "The Gulch",
        "Charlotte South End", "Raleigh Glenwood", "King Street", "Savannah Broughton",
        "Hyde Park Village", "Winter Park", "Wynwood", "Pepper Place", "Market Square",
        "NorthShore",
    ],
    "Mountain": [
        "LoHi", "RiNo", "Pearl Street", "Old Town Fort Collins", "Tejon Street",
        "Sugar House", "Provo Center", "Boise 8th Street", "Roosevelt Row",
        "Old Town Scottsdale", "Mill Avenue", "Nob Hill", "Santa Fe Plaza",
        "Fourth Avenue", "Flagstaff Downtown",
    ],
    "Midwest": [
        "Wicker Park", "Lincoln Park", "Evanston Davis", "Third Ward", "State Street",
        "North Loop", "Grand Avenue", "Crossroads KC", "Central West End",
        "Short North", "Over-the-Rhine", "Mass Ave", "Detroit Midtown",
        "Ann Arbor Main", "East Village",
    ],
}

FIRST_NAMES = ["Ava", "Liam", "Noah", "Mia", "Ethan", "Sofia", "Lucas", "Emma",
               "Diego", "Priya", "Hannah", "Marcus", "Nina", "Omar", "Grace", "Jordan"]
LAST_NAMES = ["Nguyen", "Smith", "Garcia", "Patel", "Johnson", "Kim", "Brown",
              "Martinez", "Lee", "Walker", "Chen", "Lopez", "Davis", "Turner", "Ramos"]


# --------------------------------------------------------------------------- helpers

def new_guid(rng: np.random.Generator) -> str:
    return str(uuid.UUID(bytes=rng.bytes(16), version=4))


def to_toast_ts(local_dt: datetime) -> str:
    """Local wall-clock -> Toast-style UTC ISO 8601 string (fixed UTC-6 offset)."""
    utc = local_dt - timedelta(hours=config.UTC_OFFSET_HOURS)
    return utc.strftime("%Y-%m-%dT%H:%M:%S.000+0000")


def business_dates():
    start = date.fromisoformat(config.START_DATE)
    return [start + timedelta(days=i) for i in range(config.N_DAYS)]


# --------------------------------------------------------------------------- stores

def build_stores(rng: np.random.Generator) -> list[dict]:
    """60 store dicts (internal representation; serialized to Toast Restaurant objects)."""
    tiers = []
    for tier, spec in config.TIERS.items():
        tiers += [tier] * spec["n"]
    tiers = list(rng.permutation(tiers))

    formats = list(rng.choice(
        list(config.FORMAT_WEIGHTS), size=config.N_STORES,
        p=list(config.FORMAT_WEIGHTS.values())))

    # regions: 15 stores each, names drawn in order
    regions = []
    for r in config.REGIONS:
        regions += [r] * 15
    order = rng.permutation(config.N_STORES)

    square_idx = set(rng.choice(config.N_STORES, size=config.N_SQUARE_STORES, replace=False))

    name_cursor = {r: 0 for r in config.REGIONS}
    stores = []
    for i in range(config.N_STORES):
        region = regions[order[i]]
        name = STORE_NAMES[region][name_cursor[region]]
        name_cursor[region] += 1
        tier = tiers[i]
        spec = config.TIERS[tier]
        lo, hi = spec["actual_labor_pct"]
        wage = config.HOURLY_WAGE_BY_REGION[region] + rng.uniform(
            -config.WAGE_STORE_JITTER, config.WAGE_STORE_JITTER)
        store = {
            "guid": new_guid(rng),
            "name": f"Solstice Coffee — {name}",
            "location_name": name,
            "tier": tier,
            "format": formats[i],
            "region": region,
            "pos_system": "square" if i in square_idx else "toast",
            "base_daily_orders": config.BASE_DAILY_ORDERS * spec["demand_mult"],
            "target_labor_pct": config.TARGET_LABOR_PCT,
            "actual_labor_pct": round(float(rng.uniform(lo, hi)), 4),
            "hourly_wage_avg": round(float(wage), 2),
            "revenue_center_guid": new_guid(rng),
        }
        # small employee roster with per-employee wages and names
        store["employees"] = [
            {
                "guid": new_guid(rng),
                "wage": round(float(np.clip(
                    store["hourly_wage_avg"] + rng.uniform(
                        -config.WAGE_EMPLOYEE_JITTER, config.WAGE_EMPLOYEE_JITTER),
                    13.0, 25.0)), 2),
                "job_guid": new_guid(rng),
                "first_name": FIRST_NAMES[int(rng.integers(0, len(FIRST_NAMES)))],
                "last_name": LAST_NAMES[int(rng.integers(0, len(LAST_NAMES)))],
                "external_id": f"S{i:02d}-E{k:02d}",
            }
            for k in range(config.EMPLOYEES_PER_STORE)
        ]
        stores.append(store)
    return stores


def restaurant_object(store: dict) -> dict:
    """Toast Restaurants-API shape; synthetic-only attributes namespaced under _synthetic."""
    return {
        "guid": store["guid"],
        "general": {"name": store["name"], "locationName": store["location_name"],
                    "timeZone": "America/Chicago"},
        "schedules": {
            "daySchedules": {
                "allWeek": {"openTime": f"{config.OPEN_HOUR:02d}:00:00",
                            "closeTime": f"{config.CLOSE_HOUR:02d}:00:00"}
            }
        },
        "_synthetic": {
            "sourceSystem": store["pos_system"],
            "tier": store["tier"],
            "format": store["format"],
            "region": store["region"],
            "baseDailyOrders": store["base_daily_orders"],
            "targetLaborPct": store["target_labor_pct"],
            "actualLaborPct": store["actual_labor_pct"],
            "hourlyWageAvg": store["hourly_wage_avg"],
        },
    }


def employee_objects(store: dict) -> list[dict]:
    """Toast Labor-API /employees shape. The endpoint is restaurant-scoped, so the
    file is named per restaurant (like the orders/labor pulls)."""
    return [
        {
            "guid": e["guid"],
            "entityType": "Employee",
            "externalEmployeeId": e["external_id"],
            "firstName": e["first_name"],
            "lastName": e["last_name"],
            "chosenName": None,
            "email": f"{e['first_name'].lower()}.{e['last_name'].lower()}"
                     f".{e['external_id'].lower()}@solsticecoffee.example",
            "deleted": False,
            "jobReferences": [{"guid": e["job_guid"]}],
            "wageOverrides": [{"wage": e["wage"], "jobReference": {"guid": e["job_guid"]}}],
        }
        for e in store["employees"]
    ]


# --------------------------------------------------------------------------- menu

def build_menu(rng: np.random.Generator) -> list[dict]:
    cat_guids = {c: new_guid(rng) for c in CATEGORIES}
    items = []
    for name, cat, price, prep in config.MENU:
        items.append({
            "guid": new_guid(rng),
            "name": name,
            "price": price,
            "salesCategory": {"guid": cat_guids[cat], "name": cat},
            "_synthetic": {"prepSeconds": prep},   # Toast has no prep-time field — ours
        })
    return items


# --------------------------------------------------------------------------- weather

def build_weather(rng: np.random.Generator) -> dict:
    """(region, date) -> (temp_f, precip_in). Also written to data/weather_daily.csv."""
    dates = business_dates()
    weather = {}
    rows = ["region,businessDate,temp_f,precip_in"]
    for region in config.REGIONS:
        t0, t1 = config.WEATHER_TEMP_RANGE[region]
        prev_anom = 0.0
        for i, d in enumerate(dates):
            seasonal = t0 + (t1 - t0) * i / (config.N_DAYS - 1)
            anom = config.WEATHER_AR1 * prev_anom + rng.normal(0, config.WEATHER_TEMP_SD)
            prev_anom = anom
            temp = round(float(seasonal + anom), 1)
            precip = 0.0
            if rng.random() < config.PRECIP_PROB:
                precip = round(float(rng.exponential(config.PRECIP_MEAN_IN)), 2)
            weather[(region, d)] = (temp, precip)
            rows.append(f"{region},{d.strftime('%Y%m%d')},{temp},{precip}")
    Path(config.WEATHER_CSV).write_text("\n".join(rows) + "\n", encoding="utf-8")
    return weather


# --------------------------------------------------------------------------- demand

def expected_orders(store: dict, d: date, day_idx: int, temp: float, precip: float) -> float:
    dow_mult = config.DOW_MULT[store["format"]][d.weekday()]
    payday = config.PAYDAY_MULT if d.day in config.PAYDAY_DAYS_OF_MONTH else 1.0
    trend = 1.0 + config.TREND_TOTAL * day_idx / (config.N_DAYS - 1)
    rain = 1.0 - config.PRECIP_SUPPRESSION_MAX * min(precip / config.PRECIP_SATURATION_IN, 1.0)
    warmth = 1.0 + config.TEMP_VOLUME_SLOPE * (temp - 65.0)
    return store["base_daily_orders"] * dow_mult * payday * trend * rain * warmth


def daypart_shares(store: dict, d: date) -> list[float]:
    shares = dict(config.DAYPART_SHARE[store["format"]])
    if d.weekday() >= 5 and store["format"] != "drive_thru":
        for dp, delta in config.WEEKEND_DAYPART_SHIFT.items():
            shares[dp] = max(0.01, shares[dp] + delta)
    total = sum(shares.values())
    return [shares[dp] / total for dp in DAYPART_LIST]


def item_probs_by_daypart(menu: list[dict], temp: float) -> dict:
    """daypart -> probability vector over menu items, with cold share shifted by temp."""
    by_cat = {c: [i for i, m in enumerate(menu) if m["salesCategory"]["name"] == c]
              for c in CATEGORIES}
    out = {}
    for dp in DAYPART_LIST:
        w = dict(config.CATEGORY_WEIGHTS[dp])
        shift = np.clip(config.COLD_SHARE_TEMP_SLOPE * (temp - 65.0), -0.5, 0.8)
        w["cold"] *= (1.0 + shift)
        w["espresso"] *= (1.0 - 0.4 * shift)   # hot drinks give up what cold gains
        w["brewed"] *= (1.0 - 0.4 * shift)
        total = sum(w.values())
        p = np.zeros(len(menu))
        for cat, cw in w.items():
            idxs = by_cat[cat]
            for i in idxs:
                p[i] = (cw / total) / len(idxs)
        out[dp] = p / p.sum()
    return out


def gen_orders_store_day(rng, store, menu, d, day_idx, temp, precip, dining_guids):
    """Returns (list of Toast Order objects, net_revenue, {daypart: order_count})."""
    mean = expected_orders(store, d, day_idx, temp, precip)
    n_orders = int(rng.poisson(mean))
    dp_counts = rng.multinomial(n_orders, daypart_shares(store, d))
    item_p = item_probs_by_daypart(menu, temp)

    behaviors = config.DINING_BEHAVIOR_P[store["format"]]
    beh_names = list(behaviors)
    beh_p = list(behaviors.values())
    items_k = np.array(list(config.ITEMS_PER_ORDER_P))
    items_p = np.array(list(config.ITEMS_PER_ORDER_P.values()))
    pay_names = list(config.PAYMENT_TYPE_P)
    pay_p = list(config.PAYMENT_TYPE_P.values())

    bd_int = int(d.strftime("%Y%m%d"))
    orders = []
    net_revenue = 0.0
    daypart_counts = {}
    for dp, n_dp in zip(DAYPART_LIST, dp_counts):
        daypart_counts[dp] = int(n_dp)
        if n_dp == 0:
            continue
        h0, h1 = config.DAYPARTS[dp]
        # timestamps uniform inside the daypart window
        secs = np.sort(rng.integers(0, (h1 - h0) * 3600, size=n_dp))
        n_items_all = rng.choice(items_k, size=n_dp, p=items_p)
        for j in range(n_dp):
            opened = datetime(d.year, d.month, d.day, h0) + timedelta(seconds=int(secs[j]))
            closed = opened + timedelta(seconds=int(rng.integers(90, 420)))
            picks = rng.choice(len(menu), size=int(n_items_all[j]), p=item_p[dp])

            selections, check_amount = [], 0.0
            for pi in picks:
                m = menu[int(pi)]
                voided = bool(rng.random() < config.VOID_RATE)
                if not voided:
                    check_amount += m["price"]
                selections.append({
                    "guid": new_guid(rng),
                    "entityType": "MenuItemSelection",
                    "item": {"guid": m["guid"], "entityType": "MenuItem"},
                    "displayName": m["name"],
                    "quantity": 1,
                    "price": m["price"],
                    "preDiscountPrice": m["price"],
                    "voided": voided,
                    "modifiers": [],
                    "salesCategory": dict(m["salesCategory"]),
                })
            check_amount = round(check_amount, 2)
            tax = round(check_amount * config.TAX_RATE, 2)
            total = round(check_amount + tax, 2)

            loyalty = None
            customer = None
            if rng.random() < config.LOYALTY_ATTACH_RATE:
                member = int(rng.integers(0, config.LOYALTY_POOL_SIZE))
                loyalty = {"loyaltyIdentifier": f"SOL-{member:06d}"}
                if rng.random() < config.CUSTOMER_ON_LOYALTY_P:
                    fn = FIRST_NAMES[int(rng.integers(0, len(FIRST_NAMES)))]
                    ln = LAST_NAMES[int(rng.integers(0, len(LAST_NAMES)))]
                    customer = {
                        "guid": new_guid(rng), "firstName": fn, "lastName": ln,
                        "email": f"{fn.lower()}.{ln.lower()}{member % 97}@example.com",
                        "phone": None,
                    }

            pay_type = str(rng.choice(pay_names, p=pay_p))
            tip = 0.0
            if pay_type == "CREDIT":
                tip = round(total * float(rng.uniform(*config.TIP_RATE_CREDIT)), 2)

            behavior = str(rng.choice(beh_names, p=beh_p))
            orders.append({
                "guid": new_guid(rng),
                "entityType": "Order",
                "restaurantGuid": store["guid"],
                "businessDate": bd_int,
                "openedDate": to_toast_ts(opened),
                "closedDate": to_toast_ts(closed),
                "revenueCenter": {"guid": store["revenue_center_guid"]},
                "diningOption": {"guid": dining_guids[behavior], "behavior": behavior},
                "checks": [{
                    "guid": new_guid(rng),
                    "entityType": "Check",
                    "amount": check_amount,
                    "taxAmount": tax,
                    "totalAmount": total,
                    "customer": customer,
                    "appliedLoyaltyInfo": loyalty,
                    "selections": selections,
                    "payments": [{
                        "guid": new_guid(rng), "amount": total,
                        "tipAmount": tip, "type": pay_type,
                    }],
                }],
            })
            net_revenue += check_amount
    return orders, net_revenue, daypart_counts


# --------------------------------------------------------------------------- labor

def gen_labor_store_day(rng, store, d, net_revenue, daypart_counts):
    """Toast TimeEntry objects whose aggregate cost/revenue ≈ the store's injected
    actual_labor_pct, with the tier's daypart misalignment baked in."""
    jitter = float(np.exp(rng.normal(0, config.LABOR_COST_DAILY_JITTER_SD)))
    target_cost = store["actual_labor_pct"] * net_revenue * jitter
    if target_cost <= 0:
        return []

    # allocate cost across dayparts: demand_share ** exponent, renormalized.
    # slow tier gets a much flatter allocation -> over-staffed off-peak,
    # under-staffed at morning peak. THIS is the inefficiency the model exposes.
    exponent = config.LABOR_ALIGNMENT_EXPONENT[store["tier"]]
    total_orders = max(1, sum(daypart_counts.values()))
    shares = np.array([max(daypart_counts[dp], 0.5) / total_orders for dp in DAYPART_LIST])
    alloc = shares ** exponent
    alloc = alloc / alloc.sum()

    employees = store["employees"]
    entries = []
    emp_cursor = int(rng.integers(0, len(employees)))
    bd_int = int(d.strftime("%Y%m%d"))

    for dp, frac in zip(DAYPART_LIST, alloc):
        dp_cost = target_cost * float(frac)
        h0, h1 = config.DAYPARTS[dp]
        window_h = h1 - h0
        # hours at the store-average wage, then realized via specific employees
        dp_hours = dp_cost / store["hourly_wage_avg"]
        n_chunks = max(1, int(np.ceil(dp_hours / min(config.SHIFT_MAX_CHUNK_HOURS, window_h))))
        chunk_hours = dp_hours / n_chunks
        for _ in range(n_chunks):
            emp = employees[emp_cursor % len(employees)]
            emp_cursor += 1
            dur_h = min(chunk_hours, window_h)
            # scale duration so cost is exact given this employee's actual wage
            dur_h *= store["hourly_wage_avg"] / emp["wage"]
            dur_h = float(min(dur_h, window_h))
            latest_start = (h1 - h0) * 3600 - int(dur_h * 3600)
            start_s = int(rng.integers(0, max(1, latest_start)))
            t_in = datetime(d.year, d.month, d.day, h0) + timedelta(seconds=start_s)
            t_out = t_in + timedelta(seconds=int(dur_h * 3600))
            entries.append({
                "guid": new_guid(rng),
                "entityType": "TimeEntry",
                "restaurantGuid": store["guid"],
                "employeeReference": {"guid": emp["guid"]},
                "jobReference": {"guid": emp["job_guid"]},
                "inDate": to_toast_ts(t_in),
                "outDate": to_toast_ts(t_out),
                "businessDate": bd_int,
                "regularHours": round(dur_h, 4),
                "overtimeHours": 0.0,
                "hourlyWage": emp["wage"],
                "breaks": [],
            })
    return entries


# --------------------------------------------------------------------------- main

def write_ndjson(path: Path, objs: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o, separators=(",", ":")) + "\n")


def main():
    rng = np.random.default_rng(config.SEED)
    extract = Path(config.EXTRACT_DIR)
    if extract.exists():
        shutil.rmtree(extract)
    for sub in ("orders", "labor", "config"):
        (extract / sub).mkdir(parents=True, exist_ok=True)
    Path(config.PROCESSED_DIR).mkdir(parents=True, exist_ok=True)

    stores = build_stores(rng)
    menu = build_menu(rng)
    weather = build_weather(rng)
    dining_guids = {b: new_guid(rng) for b in ("DINE_IN", "TAKE_OUT", "DELIVERY")}

    write_ndjson(extract / "config" / "restaurants.ndjson",
                 [restaurant_object(s) for s in stores])
    write_ndjson(extract / "config" / "menus.ndjson", menu)
    for s in stores:
        write_ndjson(extract / "config" / f"employees_{s['guid']}.ndjson",
                     employee_objects(s))

    dates = business_dates()
    n_orders = n_items = n_entries = 0
    labor_cost = revenue = 0.0
    tier_cost = {t: 0.0 for t in config.TIERS}
    tier_rev = {t: 0.0 for t in config.TIERS}

    for store in stores:
        for day_idx, d in enumerate(dates):
            temp, precip = weather[(store["region"], d)]
            orders, net_rev, dp_counts = gen_orders_store_day(
                rng, store, menu, d, day_idx, temp, precip, dining_guids)
            entries = gen_labor_store_day(rng, store, d, net_rev, dp_counts)

            bd = d.strftime("%Y%m%d")
            write_ndjson(extract / "orders" / f"orders_{store['guid']}_{bd}.ndjson", orders)
            write_ndjson(extract / "labor" / f"timeEntries_{store['guid']}_{bd}.ndjson", entries)

            n_orders += len(orders)
            n_items += sum(len(o["checks"][0]["selections"]) for o in orders)
            n_entries += len(entries)
            day_cost = sum(e["regularHours"] * e["hourlyWage"] for e in entries)
            labor_cost += day_cost
            revenue += net_rev
            tier_cost[store["tier"]] += day_cost
            tier_rev[store["tier"]] += net_rev

    manifest = {
        "source": "toast_api",
        "synthetic": True,
        "generator": "src/generate_data.py",
        "seed": config.SEED,
        "dateRange": {"start": dates[0].isoformat(), "end": dates[-1].isoformat()},
        "endpoints": {
            "orders": "/orders/v2/ordersBulk (impersonated)",
            "labor": "/labor/v1/timeEntries (impersonated)",
            "restaurants": "/restaurants/v1/restaurants (impersonated)",
            "menus": "/menus/v2/menus (impersonated)",
            "employees": "/labor/v1/employees (impersonated)",
        },
        "counts": {"restaurants": len(stores), "orders": n_orders,
                   "lineItems": n_items, "timeEntries": n_entries,
                   "employees": len(stores) * config.EMPLOYEES_PER_STORE},
        "notes": "All values synthetic. _synthetic:{} blocks are generator metadata, "
                 "not Toast fields. 15 locations flagged sourceSystem=square would need "
                 "a square_adapter in production.",
    }
    (extract / "EXTRACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    print("=== GENERATOR SUMMARY ===")
    print(f"orders: {n_orders:,}   line items: {n_items:,}   time entries: {n_entries:,}")
    print(f"net revenue (90d, sim scale): ${revenue:,.0f}")
    print(f"fleet labor pct: {labor_cost / revenue:.1%}")
    for t in config.TIERS:
        print(f"  {t:<9} labor pct: {tier_cost[t] / tier_rev[t]:.1%}")
    return manifest


if __name__ == "__main__":
    main()
