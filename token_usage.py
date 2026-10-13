import sqlite3

c = sqlite3.connect("driftguard.db")

print("provider | calls | tokens | rate_limited | tokens x24 (per day)")
rows = c.execute("""
       SELECT provider, COUNT(*),
              SUM(COALESCE(prompt_tokens,0) + COALESCE(completion_tokens,0)),
              SUM(outcome = 'rate_limited')
       FROM log_rows
       WHERE eval_run_id IS NOT NULL
       GROUP BY provider, eval_run_id
       ORDER BY MIN(timestamp) DESC
       LIMIT 10
   """).fetchall()
for p, n, t, rl in rows:
    print(p, n, t, rl, t * 24)

print("\n429 messages:")
for (msg,) in c.execute("SELECT substr(error_message,1,300) FROM log_rows WHERE outcome='rate_limited' LIMIT 3"):
    print(msg)

print("\nok calls with empty response:")
print(c.execute("SELECT COUNT(*) FROM log_rows WHERE outcome='ok' AND (response_text IS NULL OR response_text='')").fetchone()[0])