# ☀️ Solstice Labor Copilot

A demand-driven labor planning prototype for Solstice Coffee: a store-level
forecast (evaluated against an honest baseline ladder), deterministic labor
math with service guardrails, a manager-facing action plan with reason-coded
overrides (human in the loop), and a treatment-vs-control pilot design that
measures value instead of asserting it.

**Live demo:** https://solstice-labor-copilot.streamlit.app

## Two views

- **Store Manager** — this week's staffing plan for one store: forecast vs
  history, whole-shift ADD/TRIM actions with dollar impact, accept-or-override
  workflow (overridden rows drop out of the value math), and a grounded
  AI chat that explains the plan but can never change it.
- **Leadership (HQ)** — fleet-wide labor % vs target, where the gap
  concentrates, feasible opportunity after constraints, pilot
  treatment-vs-control results, and model quality vs baselines.

## Honest caveats

- **All data is synthetic** (a Toast-shaped POS extract, generated with a fixed
  seed). The point is the decision architecture, not the numbers.
- Pilot value assumes 50–65% capture of the identified gap — change management
  is never 100%.
- This tool sets staffing *targets*; people decisions and scheduling stay with
  the manager in Toast Scheduling (Sling) / Square Shifts. It never
  auto-schedules.

## Run locally

Requires Python 3.10+ (3.12 recommended).

```bash
pip install -r requirements.txt
streamlit run app.py
```

Precomputed artifacts (`data/`, `models/`) are committed, so the app starts
instantly. To regenerate everything from scratch (synthetic data → forecast →
labor plan → pilot, ~a minute):

```bash
python run.py
```

The **Ask the Copilot** chat tab needs an Anthropic API key in
`.streamlit/secrets.toml` (`ANTHROPIC_API_KEY = "..."`) or the environment;
without one, every other feature still works.

If `models/model.pkl` complains about a scikit-learn version mismatch,
`python run.py` retrains everything from scratch in about a minute.

## Where AI helped (and where the human overrode it)

This was built AI-assisted end to end — data generator, adapter, forecast,
labor math, app. [NOTES_AI_LOG.md](NOTES_AI_LOG.md) is the running log kept
during the build: the judgment calls AI made, the two forecast-calibration
iterations (including the first run that honestly failed its own quality
gate), and the places the human overrode the AI — real employment
constraints on shifts, the 27% plan floor, keeping API keys away from the
UI, and keeping the LLM out of the numbers entirely.
