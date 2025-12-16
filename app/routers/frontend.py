
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from pathlib import Path

router = APIRouter()

# Configurar templates
BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@router.get("/", tags=["Frontend"])
async def dashboard(request: Request):
    """Renderiza o Dashboard."""
    return templates.TemplateResponse("index.html", {"request": request})

@router.get("/ingest", tags=["Frontend"])
async def page_ingest(request: Request):
    """Renderiza página de ingestão."""
    return templates.TemplateResponse("ingest.html", {"request": request})

@router.get("/train", tags=["Frontend"])
async def page_train(request: Request):
    """Renderiza página de treinamento."""
    return templates.TemplateResponse("train.html", {"request": request})

@router.get("/predict", tags=["Frontend"])
async def page_predict(request: Request):
    """Renderiza página de predição."""
    return templates.TemplateResponse("predict.html", {"request": request})

@router.get("/history", tags=["Frontend"])
async def page_history(request: Request):
    """Renderiza página de histórico."""
    return templates.TemplateResponse("history.html", {"request": request})


@router.get("/monitoring", tags=["Frontend"])
async def page_monitoring(request: Request):
    """Renderiza página de monitoramento in-app."""
    return templates.TemplateResponse("monitoring.html", {"request": request})
