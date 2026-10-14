import sqlite3
c = sqlite3.connect("driftguard.db")
n = c.execute("delete from log_rows where eval_run_id = ?",
              ("92ea4f33-5f5e-4e33-9f75-ff157d11ed1e",)).rowcount
c.commit()
print("deleted", n)