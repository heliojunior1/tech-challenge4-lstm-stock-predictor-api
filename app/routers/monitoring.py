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


# ============== In-App Monitoring (SQLite) ==============
# Complementa Prometheus para ambientes como Render

from app.database import SessionLocal, Metric
from datetime import datetime, timedelta
from sqlalchemy import func


def save_metric(
    metric_type: str, 
    ticker: str = None, 
    value: float = None, 
    status: str = None, 
    endpoint: str = None, 
    duration_ms: float = None
):
    """
    Salva métrica no banco SQLite (monitoramento in-app).
    
    Args:
        metric_type: Tipo da métrica ("request", "prediction", "training", "error")
        ticker: Ticker relacionado (opcional)
        value: Valor numérico (latência, preço, etc.)
        status: Status ("success", "error")
        endpoint: Endpoint da requisição
        duration_ms: Duração em milissegundos
    """
    session = SessionLocal()
    try:
        metric = Metric(
            metric_type=metric_type,
            ticker=ticker,
            value=value,
            status=status,
            endpoint=endpoint,
            duration_ms=duration_ms
        )
        session.add(metric)
        session.commit()
    except Exception as e:
        print(f"[WARN] Erro ao salvar métrica: {e}")
    finally:
        session.close()


@router.get("/monitoring/summary")
async def monitoring_summary():
    """
    Retorna resumo das métricas das últimas 24 horas.
    """
    session = SessionLocal()
    try:
        since = datetime.utcnow() - timedelta(hours=24)
        
        # Total de requests
        total_requests = session.query(func.count(Metric.id)).filter(
            Metric.metric_type == "request",
            Metric.timestamp >= since
        ).scalar() or 0
        
        # Total de erros
        total_errors = session.query(func.count(Metric.id)).filter(
            Metric.metric_type == "request",
            Metric.status == "error",
            Metric.timestamp >= since
        ).scalar() or 0
        
        # Total de previsões
        total_predictions = session.query(func.count(Metric.id)).filter(
            Metric.metric_type == "prediction",
            Metric.timestamp >= since
        ).scalar() or 0
        
        # Total de treinamentos
        total_trainings = session.query(func.count(Metric.id)).filter(
            Metric.metric_type == "training",
            Metric.timestamp >= since
        ).scalar() or 0
        
        # Latência média
        avg_latency = session.query(func.avg(Metric.duration_ms)).filter(
            Metric.metric_type == "request",
            Metric.timestamp >= since
        ).scalar() or 0
        
        return {
            "period": "24h",
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": round(total_errors / max(total_requests, 1) * 100, 2),
            "total_predictions": total_predictions,
            "total_trainings": total_trainings,
            "avg_latency_ms": round(avg_latency, 2)
        }
    finally:
        session.close()


@router.get("/monitoring/requests")
async def monitoring_requests():
    """
    Retorna requests por hora (últimas 24h) para gráfico de linha.
    """
    session = SessionLocal()
    try:
        since = datetime.utcnow() - timedelta(hours=24)
        
        # Buscar todas as métricas de request
        metrics = session.query(Metric).filter(
            Metric.metric_type == "request",
            Metric.timestamp >= since
        ).order_by(Metric.timestamp.asc()).all()
        
        # Agrupar por hora
        hourly_data = {}
        for m in metrics:
            hour_key = m.timestamp.strftime("%Y-%m-%d %H:00")
            if hour_key not in hourly_data:
                hourly_data[hour_key] = {"success": 0, "error": 0}
            if m.status == "error":
                hourly_data[hour_key]["error"] += 1
            else:
                hourly_data[hour_key]["success"] += 1
        
        return {
            "labels": list(hourly_data.keys()),
            "success": [v["success"] for v in hourly_data.values()],
            "errors": [v["error"] for v in hourly_data.values()]
        }
    finally:
        session.close()


@router.get("/monitoring/predictions")
async def monitoring_predictions():
    """
    Retorna previsões por ticker para gráfico de barras.
    """
    session = SessionLocal()
    try:
        since = datetime.utcnow() - timedelta(hours=24)
        
        # Agrupar por ticker
        results = session.query(
            Metric.ticker, 
            func.count(Metric.id).label("count")
        ).filter(
            Metric.metric_type == "prediction",
            Metric.timestamp >= since
        ).group_by(Metric.ticker).all()
        
        return {
            "labels": [r[0] for r in results],
            "data": [r[1] for r in results]
        }
    finally:
        session.close()


@router.get("/monitoring/events")
async def monitoring_events(limit: int = 50):
    """
    Retorna últimos eventos (logs) para tabela.
    """
    session = SessionLocal()
    try:
        metrics = session.query(Metric).order_by(
            Metric.timestamp.desc()
        ).limit(limit).all()
        
        return {
            "events": [{
                "id": m.id,
                "type": m.metric_type,
                "ticker": m.ticker,
                "endpoint": m.endpoint,
                "status": m.status,
                "duration_ms": m.duration_ms,
                "timestamp": m.timestamp.isoformat() if m.timestamp else None
            } for m in metrics]
        }
    finally:
        session.close()


@router.get("/monitoring/config")
async def monitoring_config():
    """
    Retorna configuração do monitoramento (intervalo de refresh).
    """
    return {
        "refresh_interval": 30000,  # 30 segundos em ms
        "available_intervals": [10000, 30000, 60000]
    }
