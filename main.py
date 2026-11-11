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
        "Rewrite the text below so it reads like a person wrote it, not a language model. "
        "Vary sentence length. Cut hedging, throat-clearing, and stock transitions. "
        "Use contractions where natural. Keep every fact, name, number, and link intact. "
        "Return only the rewritten text, no preamble."
    ),
    "casual": (
        "Rewrite the text below in a relaxed, conversational human voice. "
        "Short sentences. Contractions. No corporate filler. "
        "Keep all facts, names, numbers, and links intact. "
        "Return only the rewritten text."
    ),
    "formal": (
        "Rewrite the text below as a careful human editor would: clear, direct, "
        "no AI stock phrases, no over-polishing. Keep all facts, names, numbers, and links intact. "
        "Return only the rewritten text."
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
