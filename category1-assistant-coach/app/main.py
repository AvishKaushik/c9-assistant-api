"""FastAPI application for Assistant Coach backend."""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .routers import insights, macro_review, what_if

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    yield
    # Shutdown


app = FastAPI(
    title="Assistant Coach API",
    description="Esports coaching insights for LoL and VALORANT",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(insights.router, prefix="/api/v1/insights", tags=["insights"])
app.include_router(macro_review.router, prefix="/api/v1/macro-review", tags=["macro-review"])
app.include_router(what_if.router, prefix="/api/v1/what-if", tags=["what-if"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Assistant Coach API",
        "version": "1.0.0",
        "status": "healthy",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
