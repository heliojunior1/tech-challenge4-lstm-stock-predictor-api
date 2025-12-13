"""
Models package - Contém o modelo LSTM e schemas Pydantic
"""
from app.models.lstm import StockLSTM, create_model
from app.models.schemas import (
    TrainRequest,
    TrainResponse,
    PredictRequest,
    PredictResponse,
    HealthResponse,
    ErrorResponse,
    ModelInfo,
    ModelsListResponse,
    PredictionHistory,
    PredictionHistoryResponse,
    IngestResponse
)

__all__ = [
    "StockLSTM",
    "create_model",
    "TrainRequest",
    "TrainResponse", 
    "PredictRequest",
    "PredictResponse",
    "HealthResponse",
    "ErrorResponse",
    "ModelInfo",
    "ModelsListResponse",
    "PredictionHistory",
    "PredictionHistoryResponse",
    "IngestResponse"
]
