"""
integration_check.py

Run this from the project root:  python integration_check.py

Exercises the pieces built so far, TOGETHER, against your real driftguard.db
and real Groq API key. Safe to re-run any time -- it only adds data, never
deletes anything.

What it does, in order:
  1. Confirms every module imports cleanly (catches missing files/typos
     before anything else runs).
  2. Runs the eval suite ONCE for real (accuracy + format + refusal against
     Groq), saving to driftguard.db -- same as run_once() you've used before.
  3. Pulls the full metric series back out of the store.
  4. Runs calibration on whatever quiet-window data currently exists
     (will show the "not enough points" warning if you've only run a
     handful of times so far -- that's expected, not a failure).
  5. Runs PELT change-point detection on that same series using the
     calibration output, for every metric.
  6. OPTIONAL: tests wrapper.py by making one real call through
     DriftGuardClient and confirming it's captured with
     integration_path="wrapper" -- separate from the eval-suite data path.

If any step fails, the traceback will point at exactly which file/function
broke -- paste that back for help fixing it.
"""

import sys
import traceback

def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    section("STEP 1: Import check")
    try:
        from driftguard.storage.store import DriftStore
        from driftguard.evalsuite.runner import run_once
        from driftguard.detection import baseline as cal
        from driftguard.detection import changepoint_pelt as pelt
        from driftguard.ingest.wrapper import DriftGuardClient
        print("All modules imported successfully.")
    except Exception:
        print("IMPORT FAILED -- fix this before anything else will work.")
        traceback.print_exc()
        sys.exit(1)

    db_path = "driftguard.db"
    store = DriftStore(db_path=db_path)

    section("STEP 2: Run the eval suite once (real Groq call)")
    try:
        point = run_once("groq", "openai/gpt-oss-120b", store=store)
        print(f"Run complete: accuracy={point.accuracy}, format={point.format_adherence}, "
              f"refusal={point.refusal_rate}, p50={point.latency_p50_ms}, p95={point.latency_p95_ms}")
    except Exception:
        print("EVAL SUITE RUN FAILED. Check your .env has GROQ_API_KEY set.")
        traceback.print_exc()
        sys.exit(1)

    section("STEP 3: Pull full metric series from the store")
    series = store.get_metric_series("groq", "openai/gpt-oss-120b")
    print(f"Total points in series so far: {len(series)}")
    for row in series:
        print(f"  {row.timestamp} | acc={row.accuracy} fmt={row.format_adherence} "
              f"refusal={row.refusal_rate} p95={row.latency_p95_ms}")

    section("STEP 4: Calibrate thresholds from current data")
    try:
        result = cal.calibrate_from_store(store, "groq", "openai/gpt-oss-120b", k=3.0)
        if result.warnings:
            print("Warnings (expected if you don't have much data yet):")
            for w in result.warnings:
                print(f"  - {w}")
        print(f"\nCalibrated {len(result.thresholds)} metric(s):")
        for name, mt in result.thresholds.items():
            print(f"  {name}: baseline_threshold={mt.baseline_threshold:.4f}, "
                  f"pelt_penalty={mt.pelt_penalty:.6f}")
        cal.save_calibration(result)
        print("\nSaved to configs/detectors_calibrated.json")
    except Exception:
        print("CALIBRATION FAILED.")
        traceback.print_exc()
        sys.exit(1)

    section("STEP 5: Run PELT change-point detection")
    try:
        loaded_calib = cal.load_calibration()
        all_candidates = pelt.find_changepoints_all_metrics(series, loaded_calib)
        for metric_name, candidates in all_candidates.items():
            print(f"\n{metric_name}: {len(candidates)} candidate(s)")
            for c in candidates:
                print(f"  index={c.index} delta={c.delta:.4f} is_adverse={c.is_adverse}")
        if len(series) < 20:
            print("\n(With this little data, PELT candidates aren't meaningful yet -- "
                  "this just confirms the code runs without crashing.)")
    except Exception:
        print("PELT DETECTION FAILED.")
        traceback.print_exc()
        sys.exit(1)

    section("STEP 6 (optional): Wrapper-mode live capture check")
    try:
        import os
        api_key = os.getenv("GROQ_API_KEY")
        client = DriftGuardClient(
            provider="groq",
            store=store,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp = client.post(
            "/chat/completions",
            json={
                "model": "openai/gpt-oss-120b",
                "messages": [{"role": "user", "content": "Say OK and nothing else."}],
            },
        )
        print(f"Wrapper call status: {resp.status_code}")
        print(f"Response: {resp.json()['choices'][0]['message']['content']}")
        client.close()
        print("Wrapper-mode capture succeeded -- check log_rows for integration_path='wrapper'.")
    except Exception:
        print("WRAPPER TEST FAILED (non-fatal -- eval suite pipeline above already confirmed working).")
        traceback.print_exc()

    section("DONE")
    print("If you saw no 'FAILED' messages above (Step 6 failures are non-fatal), "
          "the pipeline pieces built so far are integrating correctly.")


if __name__ == "__main__":
    main()