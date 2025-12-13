"""
Monitoring Router - Endpoints para metricas Prometheus
"""
from fastapi import APIRouter, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time

router = APIRouter()

# ============== Metricas Prometheus ==============

# Contadores
REQUESTS_TOTAL = Counter(
    'stock_predictor_requests_total',
    'Total de requisicoes da API',
    ['method', 'endpoint', 'status']
)

PREDICTIONS_TOTAL = Counter(
    'stock_predictor_predictions_total',
    'Total de previsoes realizadas',
    ['ticker']
)

TRAININGS_TOTAL = Counter(
    'stock_predictor_trainings_total',
    'Total de treinamentos realizados',
    ['ticker', 'status']
)

# Histogramas
REQUEST_LATENCY = Histogram(
    'stock_predictor_request_latency_seconds',
    'Latencia das requisicoes em segundos',
    ['endpoint']
)

TRAINING_DURATION = Histogram(
    'stock_predictor_training_duration_seconds',
    'Duracao dos treinamentos em segundos',
    ['ticker'],
    buckets=[10, 30, 60, 120, 300, 600, 1200, 3600]
)

# Gauges
MODELS_COUNT = Gauge(
    'stock_predictor_models_count',
    'Numero de modelos treinados',
    ['ticker']
)

LAST_PREDICTION_PRICE = Gauge(
    'stock_predictor_last_prediction_price',
    'Ultimo preco previsto',
    ['ticker']
)


# ============== Funcoes auxiliares para instrumentacao ==============

def record_request(method: str, endpoint: str, status: int):
    """Registra uma requisicao."""
    REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status=str(status)).inc()


def record_prediction(ticker: str, price: float):
    """Registra uma previsao."""
    PREDICTIONS_TOTAL.labels(ticker=ticker).inc()
    LAST_PREDICTION_PRICE.labels(ticker=ticker).set(price)


def record_training(ticker: str, status: str, duration: float = None):
    """Registra um treinamento."""
    TRAININGS_TOTAL.labels(ticker=ticker, status=status).inc()
    if duration:
        TRAINING_DURATION.labels(ticker=ticker).observe(duration)


def update_models_count(ticker: str, count: int):
    """Atualiza contagem de modelos."""
    MODELS_COUNT.labels(ticker=ticker).set(count)


# ============== Endpoints ==============

@router.get("/metrics")
async def metrics():
    """
    Endpoint para expor metricas do Prometheus.
    
    Formato: text/plain compativel com Prometheus scraping.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@router.get("/metrics/json")
async def metrics_json():
    """
    Retorna metricas em formato JSON (para debug).
    """
    from app.services.predict_service import PredictService
    
    # Buscar estatisticas
    models = PredictService.list_models()
    predictions = PredictService.get_prediction_history(limit=100)
    
    # Agrupar por ticker
    models_by_ticker = {}
    for m in models:
        ticker = m["ticker"]
        if ticker not in models_by_ticker:
            models_by_ticker[ticker] = 0
        models_by_ticker[ticker] += 1
    
    predictions_by_ticker = {}
    for p in predictions:
        ticker = p["ticker"]
        if ticker not in predictions_by_ticker:
            predictions_by_ticker[ticker] = 0
        predictions_by_ticker[ticker] += 1
    
    return {
        "total_models": len(models),
        "total_predictions": len(predictions),
        "models_by_ticker": models_by_ticker,
        "predictions_by_ticker": predictions_by_ticker
    }
