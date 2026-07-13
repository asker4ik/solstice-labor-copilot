"""Solstice Labor Copilot — Streamlit app (component 5).

A thin view over precomputed outputs (run `python run.py` first). Two audiences:
- STORE MANAGER (default): my staffing plan + Ask-the-Copilot assistant
- LEADERSHIP: fleet overview, pilot result, model quality

Design rules honored here: the forecast and labor math are done offline and
deterministically; this app only displays them. The GenAI chat reads FROM those
outputs and never feeds INTO them. Manager overrides (reason-coded) win over
the model. Shift assignment stays in Toast Scheduling (Sling) / Square Shifts.
"""

import json
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from src import copilot_chat, features, rationale, toast_adapter

st.set_page_config(page_title="Solstice Labor Copilot", page_icon="☀️", layout="wide")

TIER_COLORS = {"flagship": "#2a9d8f", "core": "#457b9d", "slow": "#e76f51"}
DP_LABELS = {"early": "Early (6–8)", "morning_peak": "Morning peak (8–11)",
             "midday": "Midday (11–14)", "afternoon": "Afternoon (14–17)",
             "evening": "Evening (17–20)"}


# ------------------------------------------------------------------ cached loads

@st.cache_data(show_spinner="Loading model outputs ...")
def load_outputs():
    p = Path(config.PROCESSED_DIR)
    out = {name: pd.read_parquet(p / f"{name}.parquet")
           for name in ("labor_plan", "action_plan", "roster_plan", "store_value",
                        "pilot", "overrides", "sys_flags", "predictions")}
    m = Path(config.MODELS_DIR)
    out["metrics"] = json.loads((m / "metrics.json").read_text(encoding="utf-8"))
    out["value"] = json.loads((m / "value_summary.json").read_text(encoding="utf-8"))
    out["pilot_summary"] = json.loads((m / "pilot_summary.json").read_text(encoding="utf-8"))
    out["importances"] = pd.read_csv(m / "importances.csv")
    return out


@st.cache_data(show_spinner="Loading adapter tables ...")
def load_tables():
    return toast_adapter.load_processed()


@st.cache_resource(show_spinner="Preparing forecast features ...")
def load_model_and_features():
    with open(Path(config.MODELS_DIR) / "model.pkl", "rb") as f:
        bundle = pickle.load(f)
    feat = features.build_features(load_tables())
    return bundle, feat


def fmt_date(bd: int) -> str:
    return datetime.strptime(str(bd), "%Y%m%d").strftime("%a %b %d, %Y")


def hours_to_heads(h: float, window: float) -> float:
    return h / window if window else 0.0


def md_safe(text: str) -> str:
    """Escape $ so chat replies with dollar amounts don't trigger LaTeX rendering."""
    return text.replace("$", "\\$")


def dollar_safe_stream(text_stream):
    for chunk in text_stream:
        yield chunk.replace("$", "\\$")


# ------------------------------------------------------------------ sidebar

tables = load_tables()
out = load_outputs()
stores = tables["stores"].sort_values("location_name")

st.sidebar.title("☀️ Solstice Labor Copilot")
view = st.sidebar.radio("View as", ["Store Manager", "Leadership (HQ)"])
st.sidebar.caption(
    "Decision support, not auto-scheduling: managers override with a reason code, "
    "and staffing targets are enacted in Toast Scheduling (Sling) / Square Shifts. "
    "All data is **synthetic** (Toast-shaped extract)."
)

# ================================================================== MANAGER VIEW
if view == "Store Manager":
    loc = st.sidebar.selectbox("My store", stores["location_name"].tolist())
    srow = stores[stores["location_name"] == loc].iloc[0]
    sguid = srow["store_guid"]

    plan = out["labor_plan"]
    splan = plan[plan["store_guid"] == sguid].copy()
    sval = out["store_value"][out["store_value"]["store_guid"] == sguid].iloc[0]
    sroster = out["roster_plan"][out["roster_plan"]["store_guid"] == sguid].iloc[0]
    sactions = out["action_plan"][out["action_plan"]["store_guid"] == sguid]

    st.title(f"☀️ {srow['store_name']}")
    st.caption(f"{srow['format'].replace('_', ' ')} · {srow['region']} · "
               f"POS: {srow['pos_system'].title()} · tier: {srow['tier']}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Labor % of revenue (now)", f"{sval['actual_pct']:.1%}",
              f"{(sval['actual_pct'] - config.TARGET_LABOR_PCT):+.1%} vs 28% target",
              delta_color="inverse")
    k2.metric("Labor % on the plan", f"{sval['feasible_pct']:.1%}")
    k3.metric("Annualized opportunity", f"${sval['savings_ann'] / 1e3:,.0f}K")
    peak_flag = out["sys_flags"][(out["sys_flags"]["store_guid"] == sguid)
                                 & (out["sys_flags"]["understaffed"])]
    k4.metric("Understaffed dayparts", f"{len(peak_flag)}",
              "service risk — plan ADDS here" if len(peak_flag) else "coverage OK",
              delta_color="inverse" if len(peak_flag) else "off")

    tab_plan, tab_chat = st.tabs(["📅 My staffing plan", "💬 Ask the Copilot"])

    # ---------------------------------------------------------- tab: plan
    with tab_plan:
        bundle, feat = load_model_and_features()
        dates = rationale.test_window_dates(feat)
        bd = st.select_slider("Day", options=dates, value=dates[0], format_func=fmt_date)

        day = splan[splan["business_date"] == bd].copy()
        day["dp_order"] = day["daypart"].map(
            {dp: i for i, dp in enumerate(config.DAYPART_ORDER)})
        day = day.sort_values("dp_order")
        F = out["value"]["revenue_scale_factor"]

        # historic-average plan = same labor math driven by the DOW-aware baseline
        ratio = (day["pred_dow"] / day["pred_rf"].clip(lower=0.1)).clip(0.2, 5.0)
        day["hist_required_h"] = (day["required_h"] * ratio).clip(lower=day["floor_h"])

        left, right = st.columns([3, 2])
        with left:
            st.subheader("Demand: history vs model")
            chart = pd.DataFrame({
                "daypart": [DP_LABELS[d] for d in day["daypart"]],
                "Historic weekday average": day["pred_dow"] * F,
                "Labor-model forecast": day["pred_rf"] * F,
                "Actual (what happened)": day["orders"] * F,
            }).melt(id_vars="daypart", var_name="series", value_name="orders")
            fig = px.bar(chart, x="daypart", y="orders", color="series", barmode="group",
                         color_discrete_sequence=["#a8a8a8", "#2a9d8f", "#1d3557"])
            fig.update_layout(height=340, legend_title="", margin=dict(t=10, b=0),
                              xaxis_title="", yaxis_title="orders (fleet scale)")
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("Staffing plan")

            def change_label(flagged: bool, delta_h: float) -> str:
                if flagged:
                    return "⚠ do not cut — ADD"
                n = round(abs(delta_h) / config.MIN_SHIFT_HOURS)
                if n == 0 or abs(delta_h) < 1.5:
                    return "hold"
                verb = "ADD" if delta_h > 0 else "TRIM"
                return f"{verb} {n} × 3h shift{'s' if n > 1 else ''}"

            tbl = pd.DataFrame({
                "Daypart": [DP_LABELS[d] for d in day["daypart"]],
                "Historic plan (avg on floor)": (day["hist_required_h"] / day["window_h"]).round(1),
                "Model plan (avg on floor)": (day["recommended_h"] / day["window_h"]).round(1),
                "You scheduled (avg on floor)": (day["actual_h"] / day["window_h"]).round(1),
                "Change (whole shifts)": [
                    change_label(f, d)
                    for f, d in zip(day["do_not_cut"], day["feasible_delta_h"])
                ],
            })
            st.dataframe(tbl, hide_index=True, use_container_width=True)
            st.caption("**Avg on floor** = average concurrent headcount (labor-hours ÷ "
                       "window length) — decimals come from shifts that overlap daypart "
                       "boundaries, not fractional people. The model *plans* in hours but "
                       "*acts* only in whole 3-hour shifts (Change column), never below "
                       "the 2-person floor or forecast-required coverage.")

        st.subheader("Why the model differs from your averages")
        expl = rationale.explain_day(bundle["model"], feat, sguid, bd, srow["region"])
        if not expl.empty:  # present order counts at fleet scale, same as the chart
            for c in ("pred_rf", "pred_hist", "pred_naive", "orders",
                      "weather_delta", "payday_delta", "dow_pattern", "momentum"):
                expl[c] = expl[c] * F
        cols = st.columns(len(config.DAYPART_ORDER))
        for col, dp in zip(cols, config.DAYPART_ORDER):
            r = expl[expl["daypart"] == dp]
            with col:
                st.markdown(f"**{DP_LABELS[dp]}**")
                if r.empty:
                    st.caption("no data")
                else:
                    for b in rationale.bullets(r.iloc[0]):
                        st.caption(f"• {b}")
        st.caption("_Rationale is computed by counterfactual re-prediction "
                   "(e.g. re-running the forecast with the rain removed) — "
                   "arithmetic, not generated text._")

        st.divider()
        st.subheader("🗳️ This week's action plan — accept or override")
        st.caption("Every change defaults to **accepted**. Pick a reason code to "
                   "override a row — it drops out of the value math and your "
                   "judgment wins.")
        if sactions.empty:
            st.success("Your staffing is already demand-matched — no changes recommended.")
        emp_names = tables["employees"].set_index("employee_guid")
        emp_dp = tables["labor_by_employee"]
        if not sactions.empty:
            h1, h2, h3 = st.columns([3, 2, 2])
            h1.markdown("**Recommended change**")
            h2.markdown("**Weekly impact**")
            h3.markdown("**Your decision**")
        session_overrides = []
        accepted_net = 0.0
        for i, (_, a) in enumerate(sactions.iterrows()):
            key = f"ovr_{sguid}_{i}"
            choice = st.session_state.get(key, "— accept —")
            is_overridden = choice != "— accept —"
            c1, c2, c3 = st.columns([3, 2, 2])
            with c1:
                icon = "🔺" if a["action"].startswith("ADD") else "🔻"
                st.markdown(f"{icon} **{a['action']}** — {a['dayparts']}, "
                            f"{a['day_type']}s ({a['hours_per_day']:+.1f}h/day)")
                if a["service_flag"]:
                    st.warning(a["service_flag"], icon="⚠️")
            with c2:
                impact = a["weekly_dollars"]
                impact_txt = (f"${impact:,.0f}/week" if impact >= 0
                              else f"–${-impact:,.0f}/week (service investment)")
                if is_overridden:
                    st.markdown(f"~~{impact_txt}~~")
                    st.caption(f"✋ overridden ({choice.replace('_', ' ')}) — "
                               "excluded from value math")
                else:
                    st.markdown(f"**{impact_txt}**")
                    accepted_net += impact
                span_dps = a["dayparts"].split("–")
                span = [d.strip() for d in span_dps]
                dp_span = config.DAYPART_ORDER[
                    config.DAYPART_ORDER.index(span[0]):
                    config.DAYPART_ORDER.index(span[-1]) + 1]
                cands = (emp_dp[(emp_dp["store_guid"] == sguid)
                                & (emp_dp["daypart"].isin(dp_span))]
                         .groupby("employee_guid")["hours"].sum()
                         .sort_values(ascending=False).head(3))
                names = [f"{emp_names.loc[g, 'first_name']} {emp_names.loc[g, 'last_name'][0]}."
                         for g in cands.index if g in emp_names.index]
                if names and a["action"].startswith("ADD"):
                    st.caption("Usually works this window: " + ", ".join(names))
            with c3:
                st.selectbox("Your decision", ["— accept —"] + config.OVERRIDE_REASON_CODES,
                             key=key, label_visibility="collapsed")
                if is_overridden:
                    st.text_input("Override note", key=f"ovr_note_{sguid}_{i}",
                                  placeholder="optional note for the override log ...",
                                  label_visibility="collapsed")
            if is_overridden:
                session_overrides.append(
                    (a, choice, st.session_state.get(f"ovr_note_{sguid}_{i}", "")))

        if not sactions.empty:
            n_acc = len(sactions) - len(session_overrides)
            s1, s2, s3 = st.columns(3)
            s1.metric("Changes accepted", f"{n_acc} of {len(sactions)}",
                      f"{len(session_overrides)} overridden — your judgment wins"
                      if session_overrides else "full plan accepted",
                      delta_color="off")
            s2.metric("Net value accepted / week", f"${accepted_net:,.0f}")
            s3.metric("Annualized if sustained", f"${accepted_net * 52 / 1e3:,.0f}K")
        st.caption("Overridden slots are excluded from the value math — your judgment wins. "
                   "Enact accepted changes in "
                   + ("**Toast Scheduling (powered by Sling)**" if srow["pos_system"] == "toast"
                      else "**Square Shifts**")
                   + " — this tool sets targets; people decisions stay with you.")

        for a, reason, note in session_overrides:
            st.info(f"Override this session — {a['action']}, {a['dayparts']} "
                    f"({a['day_type']}s): **{reason}**"
                    + (f" · “{note}”" if note else ""), icon="✋")
        seeded = out["overrides"][out["overrides"]["store_guid"] == sguid]
        for _, o in seeded.iterrows():
            st.info(f"Override on file — {fmt_date(o['business_date'])}, {o['daypart']}: "
                    f"**{o['reason_code']}** · “{o['note']}”", icon="✋")

        st.divider()
        st.subheader("Your roster, restructured — not shrunk schedules")
        r1, r2, r3 = st.columns(3)
        r1.metric("Weekly labor hours", f"{sroster['weekly_feasible_h']:,.0f}",
                  f"{sroster['weekly_feasible_h'] - sroster['weekly_actual_h']:+,.0f}h",
                  delta_color="inverse")
        r2.metric("Roster (people)", f"{sroster['heads_reco']:.0f}",
                  f"{sroster['heads_reco'] - sroster['heads_now']:+.0f} via natural attrition",
                  delta_color="off")
        r3.metric("Avg hours / barista / week", f"{sroster['avg_hours_reco']:.0f}h",
                  f"{sroster['avg_hours_reco'] - sroster['avg_hours_now']:+.0f}h — fuller schedules",
                  delta_color="normal")
        st.caption(f"Cuts are taken as *fewer, fuller* schedules (≥{config.MIN_WEEKLY_HOURS_PER_EMP}h, "
                   f"target ~{config.PREFERRED_WEEKLY_HOURS}h, {config.MAX_WEEKLY_HOURS}h cap) "
                   "absorbed by turnover — not by thinning everyone's hours. "
                   f"Current {config.CURRENT_AVG_WEEKLY_HOURS}h/week average is a stated assumption.")

    # ---------------------------------------------------------- tab: chat
    with tab_chat:
        st.subheader("💬 Ask the Copilot about your staffing plan")
        st.caption("An AI assistant **grounded in your store's computed plan** — it explains "
                   "and answers what-ifs; it never changes the plan or the forecast. "
                   "Disagree? It will route you to a manager override.")

        api_key = copilot_chat.get_api_key()

        if not api_key:
            st.info("The Copilot chat isn't configured on this deployment. "
                    "(Operator note: set `ANTHROPIC_API_KEY` or add it to "
                    "`.streamlit/secrets.toml` — managers never handle keys.)")
        else:
            try:
                import anthropic
            except ImportError:
                anthropic = None
                st.warning("`pip install anthropic` to enable the chat.")
            if anthropic:
                # grounding context — deterministic outputs only
                F_scale = out["value"]["revenue_scale_factor"]
                dts = pd.to_datetime(splan["business_date"].astype(str), format="%Y%m%d")
                week_df = (splan.assign(day_type=dts.dt.dayofweek.map(
                               lambda d: "weekend" if d >= 5 else "weekday"))
                           .groupby(["day_type", "daypart"])
                           .agg(actual_hours=("actual_h", "mean"),
                                planned_hours=("feasible_h", "mean"),
                                forecast_orders=("pred_rf", "mean"),
                                understaffed_rate=("do_not_cut", "mean"))
                           .reset_index())
                week_df["forecast_orders"] *= F_scale   # same fleet scale as hours
                week = week_df.round(1).to_dict("records")
                n_days = splan["business_date"].nunique()
                ctx = copilot_chat.build_context(
                    store={
                        "name": srow["store_name"], "tier": srow["tier"],
                        "format": srow["format"], "region": srow["region"],
                        "pos_system": srow["pos_system"],
                        "avg_hourly_wage": float(srow["hourly_wage_avg"]),
                        "labor_pct_now": round(float(sval["actual_pct"]), 3),
                        "labor_pct_planned": round(float(sval["feasible_pct"]), 3),
                        "target_labor_pct": config.TARGET_LABOR_PCT,
                        "annual_savings_opportunity_usd": round(float(sval["savings_ann"])),
                    },
                    plan_dayparts=week,
                    actions=sactions.drop(columns=["store_guid"]).to_dict("records"),
                    roster={
                        "weekly_hours_now": round(float(sroster["weekly_actual_h"])),
                        "weekly_hours_planned": round(float(sroster["weekly_feasible_h"])),
                        "people_now": round(float(sroster["heads_now"])),
                        "people_planned": round(float(sroster["heads_reco"])),
                        "avg_weekly_hours_now": round(float(sroster["avg_hours_now"])),
                        "avg_weekly_hours_planned": round(float(sroster["avg_hours_reco"])),
                    },
                    rationale_lines={},
                    weekly={
                        "weekly_savings_usd": round(float(sval["savings_ann"]) / 52),
                        "eval_window_days": int(n_days),
                    },
                )
                system = copilot_chat.make_system(
                    srow["store_name"], srow["pos_system"], ctx, config.OVERRIDE_REASON_CODES)

                chat_key = f"chat_{sguid}"
                if chat_key not in st.session_state:
                    st.session_state[chat_key] = []
                for m in st.session_state[chat_key]:
                    with st.chat_message(m["role"]):
                        st.markdown(md_safe(m["content"]))

                if prompt := st.chat_input("e.g. Why am I losing a Tuesday midday shift?"):
                    st.session_state[chat_key].append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    client = anthropic.Anthropic(api_key=api_key)
                    try:
                        with st.chat_message("assistant"):
                            with copilot_chat.stream_reply(
                                    client, copilot_chat.DEFAULT_MODEL, system,
                                    st.session_state[chat_key]) as stream:
                                st.write_stream(dollar_safe_stream(stream.text_stream))
                                text = stream.get_final_message().content[0].text
                        st.session_state[chat_key].append(
                            {"role": "assistant", "content": text})
                    except anthropic.AuthenticationError:
                        st.error("Invalid API key.")
                    except anthropic.RateLimitError:
                        st.error("Rate limited — try again in a moment.")
                    except anthropic.APIStatusError as e:
                        st.error(f"API error: {e.message}")

# ================================================================== LEADERSHIP VIEW
else:
    st.title("☀️ Solstice Labor Copilot — Leadership")
    v = out["value"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Fleet labor % (actual)", f"{v['fleet_actual_labor_pct']:.1%}",
              f"{v['fleet_actual_labor_pct'] - config.TARGET_LABOR_PCT:+.1%} vs 28% target",
              delta_color="inverse")
    k2.metric("Theoretical gap to 28%", f"${v['theoretical_gap_to_28_annual'] / 1e6:.1f}M/yr")
    k3.metric("Feasible opportunity", f"${v['feasible_savings_annual'] / 1e6:.2f}M/yr",
              "post shift-size, floors, guardrails, overrides", delta_color="off")
    k4.metric("Pilot roll-forward (15 slow stores)",
              f"${out['pilot_summary']['rollforward_all_slow_annual'] / 1e6:.2f}M/yr")

    tab_fleet, tab_pilot, tab_model = st.tabs(
        ["🏪 Fleet overview", "🧪 Pilot: treatment vs control", "📈 Model quality"])

    with tab_fleet:
        sv = out["store_value"].sort_values("actual_pct", ascending=False)
        fig = px.bar(sv, x="location_name", y="actual_pct", color="tier",
                     color_discrete_map=TIER_COLORS,
                     labels={"actual_pct": "labor % of revenue", "location_name": ""})
        fig.add_hline(y=config.TARGET_LABOR_PCT, line_dash="dash",
                      annotation_text="28% target")
        fig.update_layout(height=380, yaxis_tickformat=".0%", margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("The gap is not evenly spread: it concentrates in the slow tier — "
                   "the same stores that are systematically understaffed at morning peak.")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Top opportunities ($/yr)")
            top = sv.sort_values("savings_ann", ascending=False).head(10)
            st.dataframe(top[["location_name", "tier", "actual_pct", "feasible_pct",
                              "savings_ann"]].style.format(
                {"actual_pct": "{:.1%}", "feasible_pct": "{:.1%}",
                 "savings_ann": "${:,.0f}"}), hide_index=True, use_container_width=True)
        with c2:
            st.subheader("Systematic understaffing (stores flagged)")
            sf = out["sys_flags"]
            heat = (sf[sf["understaffed"]].groupby(["tier", "daypart"]).size()
                    .unstack(fill_value=0).reindex(index=["flagship", "core", "slow"],
                                                   columns=config.DAYPART_ORDER, fill_value=0))
            fig2 = px.imshow(heat, text_auto=True, color_continuous_scale="OrRd",
                             labels=dict(color="stores"))
            fig2.update_layout(height=300, margin=dict(t=10))
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("All 15 slow-tier stores are understaffed at morning peak — "
                       "the labor problem and the quality problem are the same problem.")

    with tab_pilot:
        ps = out["pilot_summary"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Treatment stores", ps["treatment_stores"],
                  f"{ps['control_stores']} matched controls", delta_color="off")
        c2.metric("Treatment labor %", f"{ps['treatment_pct_after']:.1%}",
                  f"from {ps['treatment_pct_before']:.1%}", delta_color="inverse")
        c3.metric("Measured savings (annualized)",
                  f"${ps['measured_savings_annualized'] / 1e6:.2f}M",
                  f"{ps['measured_capture_rate']:.0%} of feasible gap captured",
                  delta_color="off")
        pl = out["pilot"].copy()
        pl["label"] = pl["location_name"] + " (" + pl["group"].str[0].str.upper() + ")"
        fig = go.Figure()
        fig.add_bar(name="before", x=pl["label"], y=pl["pct_before"], marker_color="#a8a8a8")
        fig.add_bar(name="after (simulated adoption)", x=pl["label"], y=pl["pct_after"],
                    marker_color=["#2a9d8f" if g == "treatment" else "#a8a8a8"
                                  for g in pl["group"]])
        fig.add_hline(y=config.TARGET_LABOR_PCT, line_dash="dash", annotation_text="28%")
        fig.update_layout(barmode="group", height=380, yaxis_tickformat=".0%",
                          margin=dict(t=20), yaxis_title="labor % of revenue")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Controls (grey pairs) keep running as-is; the treatment-vs-control delta "
                   f"IS the measured value: {ps['treatment_delta_pts']:.1%} of revenue. "
                   "Treatment stores adopt the feasible plan at 50–65% capture — change "
                   "management is never 100%.")

    with tab_model:
        m = out["metrics"]
        st.subheader("The honest ladder: did we even need ML?")
        ladder = pd.DataFrame({
            "Forecaster": ["Naive trailing 28-day average (today's gut)",
                           "Weekday-aware trailing average (a smarter spreadsheet)",
                           "Random Forest (this tool)"],
            "wMAPE": [m["naive"]["wmape"], m["dow_aware"]["wmape"], m["rf"]["wmape"]],
            "MAE (orders)": [m["naive"]["mae"], m["dow_aware"]["mae"], m["rf"]["mae"]],
        })
        st.dataframe(ladder.style.format({"wMAPE": "{:.1%}", "MAE (orders)": "{:.2f}"}),
                     hide_index=True, use_container_width=True)
        imp = m["improvement"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Spreadsheet vs gut", f"{imp['dow_vs_naive_wmape']:+.1%}")
        c2.metric("RF vs gut", f"{imp['rf_vs_naive_wmape']:+.1%}")
        c3.metric("RF vs spreadsheet", f"{imp['rf_vs_dow_wmape']:+.1%}")
        st.caption("Evaluated on a held-out time window (days 71–90) the model never saw. "
                   "A weekday-aware spreadsheet captures half the win; ML earns the rest "
                   "through weather, payday, trend, and pooling across 60 stores. "
                   "No LLM anywhere in this path.")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("What the model learned")
            impdf = out["importances"].head(10).iloc[::-1]
            fig = px.bar(impdf, x="importance", y="feature", orientation="h",
                         color_discrete_sequence=["#2a9d8f"])
            fig.update_layout(height=350, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Recent same-weekday history dominates — exactly what sane demand "
                       "forecasting should look like. Weather and calendar add the rest.")
        with c2:
            st.subheader("Error by store tier (naive → RF)")
            rowsT = []
            for t in ("flagship", "core", "slow"):
                rowsT.append({"tier": t, "model": "naive",
                              "wMAPE": m["naive"]["by_tier"][t]["wmape"]})
                rowsT.append({"tier": t, "model": "rf",
                              "wMAPE": m["rf"]["by_tier"][t]["wmape"]})
            fig = px.bar(pd.DataFrame(rowsT), x="tier", y="wMAPE", color="model",
                         barmode="group", color_discrete_sequence=["#a8a8a8", "#2a9d8f"])
            fig.update_layout(height=350, yaxis_tickformat=".0%", margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)
