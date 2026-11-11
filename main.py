import os
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE = Path(__file__).parent

app = FastAPI(title="Local Humanizer")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("HUMANIZER_MODEL", "llama3.2")


class HumanizeRequest(BaseModel):
    text: str
    style: str = "natural"


STYLE_PROMPTS = {
    "natural": (
        "Rewrite the text below. Your goal is to destroy the statistical patterns "
        "AI detectors look for. Follow every rule:\n"
        "1. Vary sentence length aggressively. Put a 4-word sentence next to a 30-word one.\n"
        "2. Never start two consecutive sentences the same way.\n"
        "3. Use contractions everywhere a human would (don't, won't, it's, they're).\n"
        "4. Replace predictable verbs with concrete ones: 'utilize' -> 'use', 'demonstrate' -> 'show'.\n"
        "5. Delete any sentence that only announces what comes next ('In this section...', 'It is important to note...').\n"
        "6. Break one long sentence into a fragment. Humans do this.\n"
        "7. Keep every fact, number, name, and citation exactly intact.\n"
        "Return only the rewritten text. No preamble, no explanation."
    ),
    "casual": (
        "Rewrite this like a person texting a smart friend. "
        "Short sentences. Fragments allowed. Contractions mandatory. "
        "Cut every formal connector. Vary rhythm hard. "
        "Keep all facts, numbers, names intact. Return only the text."
    ),
    "academic": (
        "Rewrite this as a careful human academic would, but break AI tells. "
        "Vary sentence length. No formulaic transitions ('Furthermore', 'Moreover'). "
        "Use concrete verbs. Keep citations, data, and claims exact. "
        "Return only the text."
    ),
}


@app.post("/api/humanize")
async def humanize(req: HumanizeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")

    system = STYLE_PROMPTS.get(req.style, STYLE_PROMPTS["natural"])
    prompt = f"{system}\n\n---\n{req.text}\n---"

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.75,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_predict": 2048,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Ollama error: {e}")

    return {"humanized": data.get("response", "").strip()}


@app.get("/")
async def root():
    return FileResponse(BASE / "static" / "index.html")


app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=True)
