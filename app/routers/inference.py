"""
Inference Router - Endpoints para previsao/inferencia
"""
from fastapi import APIRouter, HTTPException
from typing import Optional

from app.models.schemas import (
    PredictRequest, 
    PredictResponse, 
    ModelsListResponse, 
    ModelInfo,
    PredictionHistoryResponse,
    PredictionHistory,
    CustomPredictRequest,
    CustomPredictResponse
)
from app.services.predict_service import PredictService, predict_price, predict_from_history
from app.routers.monitoring import record_prediction

router = APIRouter()


# IMPORTANTE: /predict/custom deve vir ANTES de /predict/{ticker}
# para evitar que 'custom' seja interpretado como um ticker
@router.post("/predict/custom", response_model=CustomPredictResponse)
async def predict_custom_endpoint(request: CustomPredictRequest):
    """
    Faz previsao a partir de dados historicos fornecidos pelo usuario.
    
    Este endpoint atende ao requisito do Tech Challenge:
    "A API deve permitir que o usuario forneca dados historicos de precos 
    e receba previsoes dos precos futuros."
    
    - **historical_prices**: Lista de precos historicos (minimo 60 valores)
    - **days**: Numero de dias para prever (1-30)
    - **model_ticker**: Ticker do modelo a ser usado (ex: PETR4.SA, AAPL)
    
    Exemplo de request:
    ```json
    {
        "historical_prices": [30.5, 31.0, 31.5, ..., 36.1],
        "days": 1,
        "model_ticker": "PETR4.SA"
    }
    ```
    """
    try:
        # Validar quantidade minima de precos
        if len(request.historical_prices) < 60:
            raise HTTPException(
                status_code=400, 
                detail=f"Minimo de 60 precos historicos necessarios. Recebido: {len(request.historical_prices)}"
            )
        
        # Validar que todos os precos sao positivos
        if any(p <= 0 for p in request.historical_prices):
            raise HTTPException(
                status_code=400, 
                detail="Todos os precos devem ser valores positivos"
            )
        
        # Fazer previsao
        result = predict_from_history(
            historical_prices=request.historical_prices,
            days=request.days,
            model_ticker=request.model_ticker
        )
        
        # Registrar metrica Prometheus
        for pred in result["predictions"]:
            record_prediction(request.model_ticker, pred["predicted_price"])
        
        return CustomPredictResponse(
            predictions=result["predictions"],
            model_used=result["model_used"],
            input_prices_count=result["input_prices_count"],
            model_ticker=result["model_ticker"]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/{ticker}", response_model=PredictResponse)
async def predict_endpoint(
    ticker: str,
    request: PredictRequest = PredictRequest()
):
    """
    Faz previsao de preco para o ticker especificado.
    
    Usa dados do banco de dados (previamente ingeridos) e modelo treinado para o ticker.
    
    - **ticker**: Simbolo da acao (ex: PETR4.SA, AAPL)
    - **days**: Numero de dias para prever (1-30)
    - **model_path**: Caminho do modelo especifico (opcional)
    """
    try:
        result = predict_price(
            ticker,
            days=request.days,
            model_path=request.model_path
        )
        
        # Registrar metrica Prometheus para cada previsao
        for pred in result["predictions"]:
            record_prediction(ticker, pred["predicted_price"])
        
        return PredictResponse(
            ticker=result["ticker"],
            model=result["model"],
            predictions=[
                {"day": p["day"], "predicted_price": p["predicted_price"]}
                for p in result["predictions"]
            ]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=ModelsListResponse)
async def list_models_endpoint(ticker: Optional[str] = None):
    """
    Lista todos os modelos treinados.
    
    - **ticker**: Filtrar por ticker (opcional)
    """
    try:
        models = PredictService.list_models(ticker)
        return ModelsListResponse(
            models=[ModelInfo(**m) for m in models],
            total=len(models)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{ticker}")
async def get_model_endpoint(ticker: str):
    """Retorna o modelo mais recente para um ticker."""
    try:
        models = PredictService.list_models(ticker)
        if not models:
            raise HTTPException(status_code=404, detail=f"Nenhum modelo encontrado para {ticker}")
        return models[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predictions/history", response_model=PredictionHistoryResponse)
async def prediction_history_endpoint(
    ticker: Optional[str] = None,
    limit: int = 50
):
    """
    Retorna historico de previsoes realizadas.
    
    - **ticker**: Filtrar por ticker (opcional)
    - **limit**: Limite de registros (default: 50)
    """
    try:
        predictions = PredictService.get_prediction_history(ticker, limit)
        return PredictionHistoryResponse(
            predictions=[PredictionHistory(**p) for p in predictions],
            total=len(predictions)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


