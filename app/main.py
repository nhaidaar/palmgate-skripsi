import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path

from app.config import CAMERA_SOURCE, DEVICE_RUNTIME_ENABLED, MODEL_PATH, DB_PATH
from app.database import Database
from app.device_runtime import build_device_runtime
from app.palm_processor import PalmProcessor
from app.routes import recognize, register, users, logs, debug, status, device_registration

log = logging.getLogger("palmgate")

db: Database = None
palm_processor: PalmProcessor = None
device_runtime = None

# Timestamp stamped at process start — appended to static asset URLs so the
# browser fetches fresh CSS/JS on every uvicorn restart without manual cache
# clearing.
_BUILD_TS = str(int(time.time()))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db, palm_processor, device_runtime
    try:
        db = Database(DB_PATH)
        palm_processor = PalmProcessor(MODEL_PATH)
        if DEVICE_RUNTIME_ENABLED and CAMERA_SOURCE == "usb":
            device_runtime = build_device_runtime(palm_processor, db)
            device_runtime.start()
        yield
    finally:
        if device_runtime is not None:
            device_runtime.stop()
            device_runtime = None
        if palm_processor is not None:
            palm_processor.close()
            palm_processor = None
        if db is not None:
            db.close()
            db = None


app = FastAPI(title="Palmprint Recognition Preview", lifespan=lifespan)

app.include_router(recognize.router)
app.include_router(register.router)
app.include_router(users.router)
app.include_router(logs.router)
app.include_router(debug.router)
app.include_router(status.router)
app.include_router(device_registration.router)

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def index():
    html = (static_dir / "index.html").read_text(encoding="utf-8")
    # Rewrite static asset URLs to include the build timestamp so the browser
    # never serves a stale CSS or JS file after a server restart.
    html = html.replace('href="/static/style.css"',
                        f'href="/static/style.css?v={_BUILD_TS}"')
    html = html.replace('href="/static/favicon.svg"',
                        f'href="/static/favicon.svg?v={_BUILD_TS}"')
    html = html.replace('src="/static/app.js"',
                        f'src="/static/app.js?v={_BUILD_TS}"')
    return HTMLResponse(
        content=html,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )
