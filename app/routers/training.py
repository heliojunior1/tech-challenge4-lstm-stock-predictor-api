"""
Training Router - Endpoints para treinamento do modelo LSTM
"""
from fastapi import APIRouter, BackgroundTasks, HTTPException
from typing import Optional
import time

from app.models.schemas import TrainRequest, TrainResponse, IngestResponse
from app.services.train_service import train_model
from app.database import init_db
from app.routers.monitoring import record_training, update_models_count

# Importar ingest diretamente
import sys
sys.path.insert(0, '.')

router = APIRouter()

# Status de treinamentos em andamento
_training_status = {}


def _background_train(ticker: str, epochs: int, batch_size: int, learning_rate: float, train_ratio: float, features: list):
    """Funcao para executar treinamento em background."""
    global _training_status
    _training_status[ticker] = {"status": "training", "progress": 0}
    start_time = time.time()
    try:
        result = train_model(
            ticker,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            train_ratio=train_ratio,
            features=features
        )
        duration = time.time() - start_time
        record_training(ticker, "success", duration)
        _training_status[ticker] = {"status": "completed", "result": result}
    except Exception as e:
        record_training(ticker, "failed")
        _training_status[ticker] = {"status": "failed", "error": str(e)}


@router.post("/train/{ticker}", response_model=TrainResponse)
async def train_endpoint(
    ticker: str,
    request: TrainRequest = TrainRequest()
):
    """
    Treina um modelo LSTM para o ticker especificado.
    
    - **ticker**: Simbolo da acao (ex: PETR4.SA, AAPL)
    - **epochs**: Numero de epocas de treinamento (default: 100)
    - **batch_size**: Tamanho do batch (default: 32)
    - **learning_rate**: Taxa de aprendizado (default: 0.001)
    - **train_ratio**: Proporcao treino/teste (default: 0.8)
    - **features**: Lista de features (default: ["close"]). Opcoes: close, volume, rsi_14, ema_20
    """
    start_time = time.time()
    try:
        result = train_model(
            ticker,
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            train_ratio=request.train_ratio,
            features=request.features
        )
        
        # Registrar metrica Prometheus
        duration = time.time() - start_time
        record_training(ticker, "success", duration)
        
        return TrainResponse(**result)
    except ValueError as e:
        record_training(ticker, "failed")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        record_training(ticker, "failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train/{ticker}/async")
async def train_async_endpoint(
    ticker: str,
    background_tasks: BackgroundTasks,
    request: TrainRequest = TrainRequest()
):
    """
    Inicia treinamento em background (assincrono).
    Use GET /train/{ticker}/status para verificar o progresso.
    """
    if ticker in _training_status and _training_status[ticker]["status"] == "training":
        raise HTTPException(status_code=409, detail=f"Treinamento ja em andamento para {ticker}")
    
    background_tasks.add_task(
        _background_train,
        ticker,
        request.epochs,
        request.batch_size,
        request.learning_rate,
        request.train_ratio,
        request.features
    )
    
    _training_status[ticker] = {"status": "started"}
    
    return {
        "message": f"Treinamento iniciado para {ticker}",
        "ticker": ticker,
        "epochs": request.epochs,
        "status_url": f"/api/v1/train/{ticker}/status"
    }


@router.get("/train/{ticker}/status")
async def train_status_endpoint(ticker: str):
    """Verifica o status de um treinamento em andamento."""
    if ticker not in _training_status:
        raise HTTPException(status_code=404, detail=f"Nenhum treinamento encontrado para {ticker}")
    
    return _training_status[ticker]


@router.post("/ingest/{ticker}", response_model=IngestResponse)
async def ingest_endpoint(
    ticker: str, 
    start_date: str = None,
    end_date: str = None
):
    """
    Ingere dados do Yahoo Finance para o banco de dados.
    
    - **ticker**: Simbolo da acao (ex: PETR4.SA, AAPL)
    - **start_date**: Data inicial no formato YYYY-MM-DD (ex: 2018-01-01)
    - **end_date**: Data final no formato YYYY-MM-DD (ex: 2024-07-20)
    
    Se nao informar datas, usa os ultimos 2 anos.
    """
    try:
        # Importar funcao de ingestao
        from ingest import ingest_data
        records = ingest_data(ticker, start_date=start_date, end_date=end_date)
        return IngestResponse(
            ticker=ticker, 
            records_inserted=records,
            start_date=start_date,
            end_date=end_date
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

