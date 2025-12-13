"""
Pydantic Schemas - Modelos de validação para requests/responses da API
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from datetime import datetime


# ============== Request Schemas ==============

class TrainRequest(BaseModel):
    """Request para iniciar treinamento."""
    model_config = ConfigDict(protected_namespaces=())
    
    epochs: Optional[int] = Field(default=100, ge=1, le=1000, description="Numero de epocas")
    batch_size: Optional[int] = Field(default=32, ge=8, le=256, description="Tamanho do batch")
    learning_rate: Optional[float] = Field(default=0.001, gt=0, lt=1, description="Taxa de aprendizado")
    train_ratio: Optional[float] = Field(default=0.8, gt=0.5, lt=1, description="Proporcao treino/teste")


class PredictRequest(BaseModel):
    """Request para fazer previsao."""
    model_config = ConfigDict(protected_namespaces=())
    
    days: Optional[int] = Field(default=1, ge=1, le=30, description="Dias para prever")
    model_path: Optional[str] = Field(default=None, description="Caminho do modelo especifico")


# ============== Response Schemas ==============

class HealthResponse(BaseModel):
    """Response do health check."""
    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TrainResponse(BaseModel):
    """Response do treinamento."""
    model_config = ConfigDict(protected_namespaces=())
    
    ticker: str
    epochs: int
    final_train_loss: float
    final_val_loss: float
    rmse: float
    mae: float
    model_path: str
    message: str = "Treinamento concluído com sucesso"


class PredictionItem(BaseModel):
    """Item de previsão."""
    day: int
    predicted_price: float


class PredictResponse(BaseModel):
    """Response da previsão."""
    ticker: str
    model: str
    predictions: List[PredictionItem]


class ModelInfo(BaseModel):
    """Informacoes de um modelo treinado."""
    model_config = ConfigDict(protected_namespaces=())
    
    id: int
    ticker: str
    model_path: str
    rmse: Optional[float]
    mae: Optional[float]
    epochs: int
    created_at: Optional[str]


class ModelsListResponse(BaseModel):
    """Response da lista de modelos."""
    models: List[ModelInfo]
    total: int


class PredictionHistory(BaseModel):
    """Item do historico de previsoes."""
    model_config = ConfigDict(protected_namespaces=())
    
    id: int
    ticker: str
    predicted_price: float
    actual_price: Optional[float]
    prediction_date: Optional[str]
    model_version: str


class PredictionHistoryResponse(BaseModel):
    """Response do histórico de previsões."""
    predictions: List[PredictionHistory]
    total: int


class ErrorResponse(BaseModel):
    """Response de erro."""
    error: str
    detail: Optional[str] = None
    status_code: int


class IngestResponse(BaseModel):
    """Response da ingestão de dados."""
    ticker: str
    records_inserted: int
    message: str = "Dados ingeridos com sucesso"
