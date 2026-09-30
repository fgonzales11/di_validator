from __future__ import annotations

import os
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import store
from .faults.api import router as fault_router
from .forecasting.api import router as forecasting_router
from .meter_lab.api import router as meter_lab_router
from .notebooks import SITE as NOTEBOOK_SITE, router as notebook_router
from .routes import datasets, events, experiments, jobs, labels, workspace


@asynccontextmanager
async def lifespan(app):
    store.initialize()
    yield


app = FastAPI(title="DI Validator", version="0.1.0", lifespan=lifespan)
app.include_router(notebook_router)
app.include_router(fault_router)
app.include_router(forecasting_router)
app.include_router(meter_lab_router)
for route_module in (workspace, datasets, labels, experiments, jobs, events):
    app.include_router(route_module.router)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=os.environ.get("DI_ALLOWED_HOSTS", "localhost,127.0.0.1,testserver").split(","),
)


@app.middleware("http")
async def local_origin(request: Request, call_next):
    gateway_token = os.environ.get("DI_GATEWAY_TOKEN")
    if os.environ.get("DI_HOSTED") == "1":
        if request.url.path == "/api/v1/health":
            return JSONResponse({"status": "ok", "version": "0.1.0", "mode": "hosted"})
        supplied = request.headers.get("x-di-gateway-token", "")
        if not gateway_token or not secrets.compare_digest(supplied, gateway_token):
            return JSONResponse({"detail": "Use the authorized DI Validator site"}, status_code=401)
    origin = request.headers.get("origin")
    if origin:
        from urllib.parse import urlparse

        allowed = {
            "localhost",
            "127.0.0.1",
            *filter(None, os.environ.get("DI_ALLOWED_ORIGINS", "").split(",")),
        }
        if urlparse(origin).hostname not in allowed:
            return JSONResponse({"detail": "Local origins only"}, status_code=403)
    return await call_next(request)


@app.exception_handler(ValueError)
async def value_error(request, error):
    return JSONResponse({"detail": str(error)}, status_code=422)


dist = store.ROOT / "frontend" / "dist"
app.mount("/notebooks", StaticFiles(directory=NOTEBOOK_SITE, html=True, check_dir=False), name="notebooks")
app.mount("/assets", StaticFiles(directory=dist / "assets", check_dir=False), name="assets")


@app.get("/{path:path}", include_in_schema=False)
def frontend(path: str):
    if path.startswith("api/"):
        raise HTTPException(404, "API route not found")
    if (dist / "index.html").exists():
        return FileResponse(dist / "index.html")
    return JSONResponse({"message": "Build the frontend with npm run build in frontend/", "docs": "/docs"})
