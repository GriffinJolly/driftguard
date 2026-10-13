import httpx

data = httpx.get("https://openrouter.ai/api/v1/models", timeout=30).json()["data"]
for m in data:
    p = m.get("pricing", {})
    if p.get("prompt") == "0" and p.get("completion") == "0":
        print(m["id"], "| ctx:", m.get("context_length"))