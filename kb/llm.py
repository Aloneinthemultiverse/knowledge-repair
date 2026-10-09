"""Grounded answer sentences from OpenRouter free models.

The model sees only the retrieved record and the question. Its sentence is kept
only if it contains the value the tool already read from that record; otherwise it
is discarded, so the LLM can phrase the answer but never change it.
Key: OPENROUTER_API_KEY in .env (git-ignored).
"""
import json
import os
import urllib.error
import urllib.request

MODELS = ["google/gemma-4-31b-it:free", "nvidia/nemotron-3-super-120b-a12b:free", "google/gemma-4-26b-a4b-it:free"]
ENV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")


def _key():
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    if os.path.exists(ENV):
        for line in open(ENV, encoding="utf-8"):
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


def available():
    return bool(_key())


def _call(model, messages, timeout=40):
    body = json.dumps({"model": model, "messages": messages, "temperature": 0, "max_tokens": 120}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body, headers={
        "Authorization": f"Bearer {_key()}", "Content-Type": "application/json",
        "HTTP-Referer": "http://127.0.0.1:8766", "X-Title": "Knowledge Repair"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    return (out["choices"][0]["message"]["content"] or "").strip()


def grounded_sentence(question, record, value):
    """Return {sentence, model, status}. status: ok | rejected | unavailable | error."""
    if not available():
        return {"sentence": None, "model": None, "status": "unavailable", "note": "No OPENROUTER_API_KEY in .env"}
    if not record:
        return {"sentence": None, "model": None, "status": "rejected", "note": "No record was retrieved"}
    messages = [
        {"role": "system", "content": "You answer questions about a knowledge base. Use ONLY the record given. "
                                      "Answer in one short sentence. If the record does not contain the answer, "
                                      "say that the record does not say. Never add facts that are not in the record."},
        {"role": "user", "content": f"Record: {record}\n\nQuestion: {question}"},
    ]
    last = "no model answered"
    for m in MODELS:
        try:
            text = _call(m, messages)
        except urllib.error.HTTPError as e:
            last = f"{m}: HTTP {e.code}"
            continue
        except Exception as e:                      # timeout / network
            last = f"{m}: {type(e).__name__}"
            continue
        if not text:
            continue
        if value and str(value).lower() not in text.lower():
            return {"sentence": text, "model": m, "status": "rejected",
                    "note": f"The sentence did not contain the record's value ({value}), so it was not used."}
        return {"sentence": text, "model": m, "status": "ok", "note": None}
    return {"sentence": None, "model": None, "status": "error", "note": f"Free models unavailable right now ({last})"}
