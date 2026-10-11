from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from zerotrace.api.routes_telemetry import router as telemetry_router
from zerotrace.api.routes_ollama import router as ollama_router
from zerotrace.api.routes_compression import router as compression_router
from zerotrace.api.routes_models import router as models_router
from zerotrace.api.routes_chat import router as chat_router


def create_app() -> FastAPI:
    """Factory function for FastAPI application."""
    app = FastAPI(
        title="Zerotrace AI — Real-Time Hardware Telemetry & Green AI Profiler",
        description="High-frequency hardware profiler, Joule/token energy metrics, Caveman compression, and Semantic Vector Cache.",
        version="2.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(models_router)
    app.include_router(telemetry_router)
    app.include_router(ollama_router)
    app.include_router(compression_router)
    app.include_router(chat_router)

    return app


app = create_app()
