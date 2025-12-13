"""
FastAPI Application - Stock Predictor API
API de previsao de precos de acoes usando LSTM
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time

from app.database import init_db
from app.routers import training, inference, monitoring
from app import __version__, __app__, __author__


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa recursos na startup e limpa no shutdown."""
    # Startup
    print("[STARTUP] Inicializando banco de dados...")
    init_db()
    print("[STARTUP] API pronta!")
    yield
    # Shutdown
    print("[SHUTDOWN] Encerrando API...")


# Criar aplicacao FastAPI
app = FastAPI(
    title=__app__,
    description="API de previsao de precos de acoes usando redes neurais LSTM",
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Incluir routers
app.include_router(training.router, prefix="/api/v1", tags=["Treinamento"])
app.include_router(inference.router, prefix="/api/v1", tags=["Inferencia"])
app.include_router(monitoring.router, tags=["Monitoramento"])


# ============== Endpoints Raiz ==============

@app.get("/", tags=["Root"])
async def root():
    """Endpoint raiz com informacoes da API."""
    return {
        "app": __app__,
        "version": __version__,
        "author": __author__,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", tags=["Root"])
async def health_check():
    """Health check para verificar se a API esta funcionando."""
    return {
        "status": "healthy",
        "timestamp": time.time()
    }


@app.get("/ready", tags=["Root"])
async def readiness_check():
    """Readiness check para Kubernetes/Render."""
    return {"status": "ready"}


# ============== Exception Handlers ==============

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": "Bad Request", "detail": str(exc)}
    )


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Not Found", "detail": str(exc)}
    )
