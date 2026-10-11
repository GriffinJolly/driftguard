import sqlite3
import sys

DB = "driftguard.db"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6   # how many recent runs to show

con = sqlite3.connect(DB)
cols = [r[1] for r in con.execute("pragma table_info(log_rows)")]


def pct(values, q):
    if not values:
        return None
    s = sorted(values)
    return s[int(q * (len(s) - 1))]


runs = con.execute(
    "select eval_run_id, provider, model_id, min(rowid), count(*) "
    "from log_rows where eval_run_id is not null "
    "group by eval_run_id order by min(rowid) desc limit ?", (N,)
).fetchall()

if not runs:
    print("no runs found")
    sys.exit(0)

for run_id, provider, model_id, _, total in reversed(runs):
    print("=" * 78)
    print(f"{provider} | {model_id} | run {run_id[:8]} | {total} calls")

    outcomes = con.execute(
        "select outcome, count(*) from log_rows where eval_run_id=? group by outcome",
        (run_id,),
    ).fetchall()
    ok = sum(n for o, n in outcomes if o == "ok")
    print("  outcomes:", ", ".join(f"{o}={n}" for o, n in outcomes))
    flag = "  <-- MOSTLY FAILED" if ok / total < 0.5 else ""
    print(f"  ok rate: {ok}/{total} = {ok/total:.0%}{flag}")

    if "eval_task_type" in cols:
        print("  by task (ok/total):",
              ", ".join(f"{t}={o}/{n}" for t, o, n in con.execute(
                  "select eval_task_type, coalesce(sum(outcome='ok'),0), count(*) "
                  "from log_rows where eval_run_id=? group by eval_task_type", (run_id,))))

    if "latency_ms" in cols:
        lat = [r[0] for r in con.execute(
            "select latency_ms from log_rows where eval_run_id=? and outcome='ok' "
            "and latency_ms is not null", (run_id,))]
        if lat:
            print(f"  latency ms: p50={pct(lat,.5):.0f} p95={pct(lat,.95):.0f} max={max(lat):.0f}")

    if "prompt_tokens" in cols and "completion_tokens" in cols:
        pt, ct = con.execute(
            "select coalesce(sum(prompt_tokens),0), coalesce(sum(completion_tokens),0) "
            "from log_rows where eval_run_id=?", (run_id,)).fetchone()
        print(f"  tokens: prompt={pt} completion={ct} total={pt+ct}")

    if "response_text" in cols:
        empty = con.execute(
            "select count(*) from log_rows where eval_run_id=? and outcome='ok' "
            "and (response_text is null or trim(response_text)='')", (run_id,)
        ).fetchone()[0]
        if empty:
            print(f"  WARNING: {empty} OK calls have empty response_text")

    if "error_message" in cols:
        for (msg,) in con.execute(
            "select distinct substr(error_message,1,140) from log_rows "
            "where eval_run_id=? and outcome!='ok' and error_message is not null "
            "limit 3", (run_id,)):
            print("  error sample:", msg)

con.close()