import asyncio
import mimetypes
import os
import sys
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException

from mikazuki.app.config import app_config
from mikazuki.app.api import load_schemas, load_presets
from mikazuki.app.api import router as api_router
# from mikazuki.app.ipc import router as ipc_router
from mikazuki.app.proxy import router as proxy_router
from mikazuki.utils.devices import check_torch_gpu

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

FRONTEND_DIST_DIR = Path("./frontend/dist")
FRONTEND_ASSETS_DIR = FRONTEND_DIST_DIR / "assets"


class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except HTTPException as ex:
            if ex.status_code == 404:
                return await super().get_response("index.html", scope)
            else:
                raise ex


def _safe_frontend_path(base_dir: Path, relative_path: str) -> Path:
    """Resolve a frontend path while preventing traversal outside the dist directory."""
    base = base_dir.resolve()
    target = (base / relative_path).resolve()
    if target != base and base not in target.parents:
        raise HTTPException(status_code=404)
    return target


def _patch_training_page_js(asset_name: str, content: str) -> str:
    """Expose Anima in the prebuilt VuePress frontend without modifying the submodule.

    The frontend repository only ships compiled assets. The actual training form is
    already loaded dynamically from the backend schema, so all we need here is a
    discoverable name for the existing Flux/Chroma/Anima expert route.
    """
    if asset_name.startswith("app.") and asset_name.endswith(".js"):
        content = content.replace(
            '{"text":"Flux","link":"/lora/flux.md"}',
            '{"text":"Anima / Flux / Chroma","link":"/lora/flux.md"}',
        )
        content = content.replace("Flux LoRA ", "Anima / Flux / Chroma ")
    elif asset_name.startswith("flux.html.") and asset_name.endswith(".js"):
        content = content.replace("Flux LoRA ", "Anima / Flux / Chroma ")
    return content


async def app_startup():
    app_config.load_config()

    await load_schemas()
    await load_presets()
    await asyncio.to_thread(check_torch_gpu)

    if sys.platform == "win32" and os.environ.get("MIKAZUKI_DEV", "0") != "1":
        webbrowser.open(f'http://{os.environ["MIKAZUKI_HOST"]}:{os.environ["MIKAZUKI_PORT"]}')


@asynccontextmanager
async def lifespan(app: FastAPI):
    await app_startup()
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(proxy_router)


cors_config = os.environ.get("MIKAZUKI_APP_CORS", "")
if cors_config != "":
    if cors_config == "1":
        cors_config = ["http://localhost:8004", "*"]
    else:
        cors_config = cors_config.split(";")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def add_cache_control_header(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "max-age=0"
    return response

app.include_router(api_router, prefix="/api")
# app.include_router(ipc_router, prefix="/ipc")


@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIST_DIR / "index.html")


@app.get("/favicon.ico", response_class=FileResponse)
async def favicon():
    return FileResponse("assets/favicon.ico")


@app.get("/assets/{asset_name:path}")
async def frontend_asset(asset_name: str):
    """Serve frontend assets, patching only the compiled training-page navigation text."""
    asset_path = _safe_frontend_path(FRONTEND_ASSETS_DIR, asset_name)
    if not asset_path.is_file():
        raise HTTPException(status_code=404)

    if asset_name.endswith(".js") and (
        asset_name.startswith("app.") or asset_name.startswith("flux.html.")
    ):
        content = asset_path.read_text(encoding="utf-8")
        content = _patch_training_page_js(asset_name, content)
        return Response(content=content, media_type="application/javascript")

    return FileResponse(asset_path)


@app.get("/lora/flux.html")
async def flux_training_page():
    """Rename the shared Flux/Chroma/Anima page in the initial server-rendered HTML."""
    page_path = FRONTEND_DIST_DIR / "lora" / "flux.html"
    if not page_path.is_file():
        raise HTTPException(status_code=404)
    content = page_path.read_text(encoding="utf-8")
    content = content.replace("Flux LoRA 训练 专家模式", "Anima / Flux / Chroma 训练 专家模式")
    return HTMLResponse(content=content)


app.mount("/", SPAStaticFiles(directory="frontend/dist", html=True), name="static")
