import sqlite3, json, sys

db = sys.argv[1] if len(sys.argv) > 1 else "driftguard.db"
con = sqlite3.connect(db)
con.row_factory = sqlite3.Row

out = {}
for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
    schema = [dict(r) for r in con.execute(f"PRAGMA table_info({name})")]
    rows = [dict(r) for r in con.execute(f"SELECT * FROM {name} ORDER BY rowid DESC LIMIT 5")]
    out[name] = {"columns": [(c["name"], c["type"]) for c in schema], "sample_rows": rows}

with open("results_format.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, default=str)
print("Wrote results_format.json")