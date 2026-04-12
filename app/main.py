"""FastAPI entrypoint for the personal assistant service."""

from fastapi import FastAPI

from app.api.routes import router as api_router


def create_app() -> FastAPI:
    app = FastAPI(title="Personal Assistant API", version="0.1.0")
    app.include_router(api_router)
    return app


app = create_app()
