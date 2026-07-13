"""Solstice Labor Copilot — all tunable constants in one place.

Every knob the generator, model, labor layer, and pilot use lives here.
No magic numbers in src/.
"""

# ---------------------------------------------------------------- reproducibility
SEED = 42

# ---------------------------------------------------------------- simulation window
N_DAYS = 90
START_DATE = "2026-02-02"          # a Monday; businessDates run through 2026-05-02

# ---------------------------------------------------------------- paths
DATA_DIR = "data"
EXTRACT_DIR = "data/toast_extract"
PROCESSED_DIR = "data/processed"
WEATHER_CSV = "data/weather_daily.csv"   # weather is NOT Toast data — separate source
MODELS_DIR = "models"

# ---------------------------------------------------------------- fleet
N_STORES = 60
# tier -> (store count, demand multiplier, actual labor pct range to inject)
# slow tier is the concentrated tail: 8-14 pts hot vs the 28% target
TIERS = {
    "flagship": {"n": 10, "demand_mult": 1.6, "actual_labor_pct": (0.285, 0.300)},
    "core":     {"n": 35, "demand_mult": 1.0, "actual_labor_pct": (0.300, 0.325)},
    "slow":     {"n": 15, "demand_mult": 0.6, "actual_labor_pct": (0.360, 0.420)},
}
BASE_DAILY_ORDERS = 50             # fleet mean ~50 orders/store/day (scale model)
TARGET_LABOR_PCT = 0.28
N_SQUARE_STORES = 15               # legacy POS; flagged via _synthetic.sourceSystem
FORMAT_WEIGHTS = {"drive_thru": 0.33, "cafe": 0.50, "kiosk": 0.17}
REGIONS = ["Texas", "Southeast", "Mountain", "Midwest"]
HOURLY_WAGE_BY_REGION = {          # store-average wage $
    "Texas": 16.5, "Southeast": 16.0, "Mountain": 18.0, "Midwest": 17.5,
}
WAGE_STORE_JITTER = 0.75           # +/- $ per store
WAGE_EMPLOYEE_JITTER = 1.50        # +/- $ per employee around store average
EMPLOYEES_PER_STORE = 9

# fleet-level P&L anchors (the case facts) — used ONLY to scale sim ratios up to
# the real P&L in reporting; never fed into the model
FLEET_ANNUAL_REVENUE = 140_000_000
STATED_FLEET_LABOR_PCT = 0.32

# ---------------------------------------------------------------- store hours & dayparts
OPEN_HOUR = 6                      # local
CLOSE_HOUR = 20                    # local
UTC_OFFSET_HOURS = -6              # all stores treated as UTC-6 (DST ignored; see README)
# daypart -> (start_hour, end_hour) local
DAYPARTS = {
    "early":        (6, 8),
    "morning_peak": (8, 11),
    "midday":       (11, 14),
    "afternoon":    (14, 17),
    "evening":      (17, 20),
}
DAYPART_ORDER = ["early", "morning_peak", "midday", "afternoon", "evening"]

# daypart share of daily orders, by store format (must each sum to 1.0)
# coffee is morning-heavy: morning_peak ~35-40%, evening ~5%
DAYPART_SHARE = {
    "drive_thru": {"early": 0.20, "morning_peak": 0.42, "midday": 0.18, "afternoon": 0.14, "evening": 0.06},
    "cafe":       {"early": 0.12, "morning_peak": 0.37, "midday": 0.26, "afternoon": 0.18, "evening": 0.07},
    "kiosk":      {"early": 0.15, "morning_peak": 0.40, "midday": 0.25, "afternoon": 0.15, "evening": 0.05},
}

# day-of-week demand multipliers (Mon..Sun), by format.
# amplitudes calibrated so the systematic signal dominates Poisson noise —
# commuter-driven drive-thrus die on weekends, cafes surge (brunch effect)
DOW_MULT = {
    "drive_thru": [1.10, 1.10, 1.10, 1.10, 1.15, 0.70, 0.60],
    "cafe":       [0.80, 0.90, 0.95, 1.00, 1.15, 1.50, 1.30],
    "kiosk":      [0.85, 0.95, 0.95, 1.00, 1.15, 1.40, 1.25],
}

# weekends shift cafe/kiosk demand later in the day (brunch): additive daypart-share
# deltas applied Sat/Sun then renormalized (drive_thru unaffected)
WEEKEND_DAYPART_SHIFT = {
    "early": -0.04, "morning_peak": -0.06, "midday": 0.05, "afternoon": 0.04, "evening": 0.01,
}

# ---------------------------------------------------------------- calendar effects
PAYDAY_DAYS_OF_MONTH = (1, 15)
PAYDAY_MULT = 1.08
TREND_TOTAL = 0.08                 # linear growth over the 90 days (SSS mandate)

# ---------------------------------------------------------------- weather (per region, daily)
# seasonal mean temp ramps linearly Feb -> May
WEATHER_TEMP_RANGE = {             # region -> (mean temp day 0, mean temp day 89)
    "Texas": (54, 82), "Southeast": (52, 80), "Mountain": (38, 68), "Midwest": (32, 66),
}
WEATHER_AR1 = 0.65                 # day-to-day temperature persistence
WEATHER_TEMP_SD = 5.0
PRECIP_PROB = 0.30
PRECIP_MEAN_IN = 0.30              # exponential mean when it rains
# graded weather-volume response (learnable, realistic amplitudes):
PRECIP_SUPPRESSION_MAX = 0.30      # volume falls up to 30% ...
PRECIP_SATURATION_IN = 0.60        # ... saturating at this rainfall
TEMP_VOLUME_SLOPE = 0.0025         # +/- per degF around 65F (~+/-8% across a 30F swing)
COLD_SHARE_TEMP_SLOPE = 0.010      # cold-drink weight scales with (temp - 65) * slope

# ---------------------------------------------------------------- order composition
ITEMS_PER_ORDER_P = {1: 0.55, 2: 0.30, 3: 0.15}
LOYALTY_ATTACH_RATE = 0.60         # ~60% of revenue is loyalty-driven
LOYALTY_POOL_SIZE = 180_000        # ~180K active members
CUSTOMER_ON_LOYALTY_P = 0.25       # subset of loyalty checks carry a customer object
VOID_RATE = 0.005                  # fraction of selections voided (adapter must drop)
TAX_RATE = 0.0825
TIP_RATE_CREDIT = (0.0, 0.22)      # uniform range on credit payments
PAYMENT_TYPE_P = {"CREDIT": 0.80, "CASH": 0.15, "OTHER": 0.05}
DINING_BEHAVIOR_P = {              # by format
    "drive_thru": {"TAKE_OUT": 0.97, "DINE_IN": 0.00, "DELIVERY": 0.03},
    "cafe":       {"TAKE_OUT": 0.55, "DINE_IN": 0.40, "DELIVERY": 0.05},
    "kiosk":      {"TAKE_OUT": 1.00, "DINE_IN": 0.00, "DELIVERY": 0.00},
}

# category weights by daypart (before weather adjustment); categories:
# espresso / brewed / cold / food / bakery
CATEGORY_WEIGHTS = {
    "early":        {"espresso": 0.40, "brewed": 0.28, "cold": 0.10, "food": 0.08, "bakery": 0.14},
    "morning_peak": {"espresso": 0.42, "brewed": 0.22, "cold": 0.12, "food": 0.10, "bakery": 0.14},
    "midday":       {"espresso": 0.30, "brewed": 0.12, "cold": 0.20, "food": 0.28, "bakery": 0.10},
    "afternoon":    {"espresso": 0.28, "brewed": 0.10, "cold": 0.34, "food": 0.14, "bakery": 0.14},
    "evening":      {"espresso": 0.26, "brewed": 0.10, "cold": 0.36, "food": 0.16, "bakery": 0.12},
}

# menu: name, category, price, prep seconds (prepSeconds is OUR metadata, not a Toast field)
MENU = [
    ("Drip Coffee",          "brewed",   2.95,  25),
    ("Pour Over",            "brewed",   4.50, 150),
    ("Espresso",             "espresso", 3.25,  40),
    ("Americano",            "espresso", 3.75,  45),
    ("Cappuccino",           "espresso", 4.95,  85),
    ("Latte",                "espresso", 5.25,  90),
    ("Mocha",                "espresso", 5.75, 105),
    ("Cold Brew",            "cold",     4.75,  35),
    ("Iced Latte",           "cold",     5.50,  95),
    ("Iced Tea",             "cold",     3.50,  30),
    ("Blended Frappe",       "cold",     6.25, 150),
    ("Breakfast Sandwich",   "food",     6.50, 120),
    ("Avocado Toast",        "food",     7.25, 100),
    ("Steel-Cut Oatmeal",    "food",     4.95,  60),
    ("Butter Croissant",     "bakery",   3.75,  10),
    ("Blueberry Muffin",     "bakery",   3.25,  10),
]

# ---------------------------------------------------------------- noise
LABOR_COST_DAILY_JITTER_SD = 0.03  # lognormal sd on daily labor cost vs injected pct
# demand noise is Poisson on the store-day order count (integer, realistic);
# at these volumes Poisson already contributes meaningful noise — keep signal dominant

# labor misalignment: allocation across dayparts ~ demand_share ** exponent, renormalized.
# 1.0 = perfectly demand-matched; lower = flatter (over-staffed off-peak, under-staffed peak).
# slow tier is nearly flat: enough that even their inflated totals miss peak coverage —
# the "expensive AND understaffed at rush" thesis must be visible in the flags
LABOR_ALIGNMENT_EXPONENT = {"flagship": 0.90, "core": 0.80, "slow": 0.10}
SHIFT_MAX_CHUNK_HOURS = 3.0

# ---------------------------------------------------------------- forecast (component 2)
TRAIN_DAYS = 70                    # train on days 1-70, test 71-90 (time split, never random)
BASELINE_TRAILING_DAYS = 28        # incumbent gut-feel: trailing-4-week same-daypart average
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 16,
    "min_samples_leaf": 15,        # smoother leaves — under-smoothing just fits Poisson noise
    "n_jobs": -1,
    "random_state": SEED,
}

# ---------------------------------------------------------------- labor layer (component 3)
# required labor = forecast orders x store prep-minutes/order x SERVICE STANDARD, where the
# standard (total labor minutes per prep-minute: register, handoff, restock, cleaning, idle
# coverage) is CALIBRATED FROM the tier that demonstrably runs near target — not invented.
SERVICE_STANDARD_TIER = "flagship" # anchor the standard to best-demonstrated operations
PLAN_FLOOR_PCT = 0.27              # never plan a store below 27% labor — 1pt under target is
                                   # the edge of demonstrated operations; lower is untested
# feasibility constraints — recommendations must survive contact with real employment:
MIN_SHIFT_HOURS = 3.0              # can't schedule a human for less (reporting-time-pay reality)
MIN_HEADS_OPEN = 2                 # never one person alone in a store (safety/cash handling)
MIN_WEEKLY_HOURS_PER_EMP = 20      # part-time floor
PREFERRED_WEEKLY_HOURS = 35        # fuller schedules -> lower turnover (cap 40, no OT)
MAX_WEEKLY_HOURS = 40
CURRENT_AVG_WEEKLY_HOURS = 26      # illustrative current avg (labeled assumption in UI/README)
OVERRIDE_REASON_CODES = ["weather", "local_event", "training", "staffing_constraint", "manager_judgment"]

# ---------------------------------------------------------------- pilot (component 4)
PILOT_TREATMENT_N = 8              # of the 15 slow-tier stores; rest are matched controls
PILOT_CAPTURE_RANGE = (0.50, 0.65) # treatment stores capture 50-65% of their gap
