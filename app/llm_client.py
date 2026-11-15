import os
import time
import requests

MODELS = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"]


def ask_llm(prompt):
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY set nahi hai. CMD mein set GEMINI_API_KEY=... chalayen.")
    last_error = None
    for model in MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        for attempt in range(3):
            try:
                resp = requests.post(
                    url,
                    headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": 0.1},
                    },
                    timeout=40,
                )
            except requests.exceptions.RequestException as e:
                last_error = f"{model}: {type(e).__name__}"
                time.sleep(3)
                continue
            if resp.status_code in (429, 500, 503, 504):
                last_error = f"{model}: {resp.status_code}"
                time.sleep(4 * (attempt + 1))
                continue
            if resp.status_code == 404:
                last_error = f"{model}: model nahi mila"
                break
            if resp.status_code != 200:
                raise RuntimeError(f"{model} -> {resp.status_code}: {resp.text[:300]}")
            parts = resp.json()["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts).strip()
    raise RuntimeError(f"LLM jawab nahi de saka. Aakhri masla: {last_error}")


if __name__ == "__main__":
    print(ask_llm("Say hello in one short sentence."))