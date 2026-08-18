import os
from dotenv import load_dotenv
from driftguard.ingest.transport_hook import make_capturing_client
from driftguard.storage.store import DriftStore

load_dotenv() 

groq_key = os.getenv("GROQ_API_KEY")

store = DriftStore()
client = make_capturing_client(store=store, base_url="https://api.groq.com")

resp = client.post(
    "/openai/v1/chat/completions",
    headers={"Authorization": f"Bearer {groq_key}"},
    json={
        "model": "openai/gpt-oss-120b",
        "messages": [{"role": "user", "content": "say hi in one word"}],
    },
)
print(resp.status_code, resp.json())