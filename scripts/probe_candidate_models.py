import os
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

CANDIDATES = [
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "google/gemma-4-26b-a4b-it:free",
    "thinkingmachines/inkling:free",
    "poolside/laguna-s-2.1:free",
]
PROMPT = "What is 2+2? Answer with one letter: A) 3 B) 4 C) 5 D) 6"

for model in CANDIDATES:
    try:
        r = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]},
            json={"model": model,
                  "messages": [{"role": "user", "content": PROMPT}],
                  "max_tokens": 300},
            timeout=60,
        )
        if r.status_code == 200:
            ch = r.json()["choices"][0]
            print(f"{model} | 200 | finish={ch.get('finish_reason')} | "
                  f"content={repr((ch['message'].get('content') or '')[:80])}")
        else:
            print(f"{model} | {r.status_code} | {r.text[:120]}")
    except Exception as e:
        print(f"{model} | EXC | {e}")
    time.sleep(5)