import os
import sqlite3

import httpx
from dotenv import load_dotenv

load_dotenv()

# --- 1. Groq refusal rows ---------------------------------------------------
c = sqlite3.connect("driftguard.db")
tables = [r[0] for r in c.execute("select name from sqlite_master where type='table'")]
tbl = None
for t in tables:
    cols = [r[1] for r in c.execute(f"pragma table_info({t})")]
    if "eval_task_type" in cols and "response_text" in cols:
        tbl = t
        break
print("log table:", tbl)

rows = c.execute(
    f"select eval_task_id, outcome, substr(response_text,1,200) from {tbl} "
    "where eval_task_type like '%refus%' and provider='groq' "
    "order by rowid desc limit 6"
).fetchall()
for r in rows:
    print(r)

# --- 2. OpenRouter 404 reason -----------------------------------------------
print("\n--- OpenRouter probe ---")
r = httpx.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]},
    json={"model": "openai/gpt-oss-120b:free",
          "messages": [{"role": "user", "content": "hi"}]},
    timeout=30,
)
print(r.status_code, r.text[:500])

print("\n--- gpt-oss models currently listed ---")
ids = [m["id"] for m in httpx.get("https://openrouter.ai/api/v1/models").json()["data"]]
print([i for i in ids if "gpt-oss" in i])