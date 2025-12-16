"""
FastAPI Application - Stock Predictor API
API de previsao de precos de acoes usando LSTM
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time

from fastapi.staticfiles import StaticFiles
from app.database import init_db
from app.routers import training, inference, monitoring, frontend
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


# ============== Middleware para Monitoramento In-App ==============
from starlette.requests import Request
from app.routers.monitoring import save_metric


@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    """
    Middleware para capturar todas as requests e salvar métricas no SQLite.
    Complementa Prometheus para ambientes sem acesso externo.
    """
    start_time = time.time()
    
    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000
        
        # Ignorar rotas estáticas e de métricas para evitar poluição
        path = request.url.path
        if not path.startswith(("/static", "/metrics", "/monitoring")):
            save_metric(
                metric_type="request",
                endpoint=path,
                status="success" if response.status_code < 400 else "error",
                duration_ms=duration_ms
            )
        
        return response
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        save_metric(
            metric_type="request",
            endpoint=request.url.path,
            status="error",
            duration_ms=duration_ms
        )
        raise


# Incluir routers
app.include_router(training.router, prefix="/api/v1", tags=["Treinamento"])
app.include_router(inference.router, prefix="/api/v1", tags=["Inferencia"])
app.include_router(monitoring.router, prefix="/api/v1", tags=["Monitoramento"])
app.include_router(frontend.router, tags=["Frontend"])

# Mount Static Files
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# ============== Endpoints Raiz ==============

# Nota: O endpoint raiz "/" agora é servido pelo frontend.router em app/routers/frontend.py
# O antigo endpoint JSON foi removido para dar lugar ao Dashboard.

@app.get("/api/info", tags=["Root"])
async def api_info():
    """Endpoint informativo da API (antigo root)."""
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
