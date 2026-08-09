import asyncio
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .training import TrainingManager

BASE_DIR = Path(__file__).parent

app = FastAPI(title="Tiny Shakespeare LM Trainer")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

manager = TrainingManager()


class TrainStartRequest(BaseModel):
    max_iters: Optional[int] = None
    batch_size: Optional[int] = None
    block_size: Optional[int] = None
    learning_rate: Optional[float] = None
    eval_interval: Optional[int] = None
    eval_iters: Optional[int] = None
    warmup_iters: Optional[int] = None


class GenerateRequest(BaseModel):
    prompt: str = "\n"
    max_new_tokens: int = 200
    temperature: float = 1.0


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        request, "index.html", {"status": manager.status()}
    )


@app.get("/api/status")
async def get_status():
    return manager.status()


@app.get("/api/history")
async def get_history():
    return {"history": manager.history}


@app.post("/api/train/start")
async def train_start(req: TrainStartRequest):
    try:
        manager.start(req.model_dump(exclude_none=True))
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    return {"status": "started", "config": manager.status()["config"]}


@app.post("/api/train/stop")
async def train_stop():
    manager.stop()
    return {"status": "stopping"}


@app.post("/api/train/reset")
async def train_reset():
    try:
        manager.reset()
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    return {"status": "reset"}


@app.get("/api/train/stream")
async def train_stream():
    """Server-Sent Events stream of training metrics.

    Replays the full history first (so a page refresh mid-training still
    sees the whole chart), then polls for new events as they land.
    """

    async def event_generator():
        last_sent = 0
        while True:
            hist = manager.history
            while last_sent < len(hist):
                event = hist[last_sent]
                yield f"data: {json.dumps(event)}\n\n"
                last_sent += 1
                if event.get("type") == "done":
                    return
            if not manager.is_training:
                return
            await asyncio.sleep(0.1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    max_new = max(1, min(req.max_new_tokens, 1000))
    temp = max(0.05, min(req.temperature, 3.0))
    result = manager.generate(req.prompt, max_new, temp)
    return result
