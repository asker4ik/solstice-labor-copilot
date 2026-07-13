"""One-command pipeline: generate -> adapt -> forecast -> labor -> pilot.

Fixed seed; deletes nothing outside data/ and models/. Run:
    python run.py            # full pipeline
    python run.py --stage 1  # just generate + adapt + checkpoint-1 summary
"""

import argparse
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=99,
                    help="run through this stage (1=data, 2=forecast, 3=labor, 4=pilot)")
    args = ap.parse_args()

    t0 = time.time()
    from src import generate_data, toast_adapter
    print(">> Stage 1a: generating Toast-shaped extract ...")
    generate_data.main()
    print(f"\n>> Stage 1b: adapter flattening extract ({time.time() - t0:.0f}s elapsed) ...")
    tables = toast_adapter.build_all()
    toast_adapter.validation_summary(tables)
    if args.stage <= 1:
        return

    print("\n>> Stage 2: training demand forecast ...")
    from src import forecast
    forecast.main(tables)
    if args.stage <= 2:
        return

    print("\n>> Stage 3: labor recommendations + value ...")
    from src import labor
    labor.main(tables)
    if args.stage <= 3:
        return

    print("\n>> Stage 4: pilot simulation ...")
    from src import pilot
    pilot.main(tables)
    print(f"\nDone in {time.time() - t0:.0f}s. Launch the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
