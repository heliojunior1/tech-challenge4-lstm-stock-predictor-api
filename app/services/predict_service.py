"""
Predict Service - Serviço de inferência/previsão

Carrega modelo treinado e faz previsões de preço
"""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from sklearn.preprocessing import MinMaxScaler

from app.models.lstm import StockLSTM
from app.services.data_service import DataService
from app.database import SessionLocal, TrainedModel, Prediction
from app.config import MODELS_DIR
from datetime import datetime


class PredictService:
    """
    Serviço para fazer previsões usando modelo LSTM treinado.
    """
    
    def __init__(self, ticker: str):
        """
        Inicializa o serviço de previsão.
        
        Args:
            ticker: Símbolo da ação
        """
        self.ticker = ticker
        self.model: Optional[StockLSTM] = None
        self.scaler: Optional[MinMaxScaler] = None
        self.model_path: Optional[Path] = None
    
    def load_model(self, model_path: str = None) -> bool:
        """
        Carrega modelo treinado do disco.
        
        Args:
            model_path: Caminho do modelo (opcional, usa o mais recente)
        
        Returns:
            True se carregou com sucesso
        """
        if model_path:
            self.model_path = Path(model_path)
        else:
            # Buscar modelo mais recente do banco
            session = SessionLocal()
            try:
                trained = session.query(TrainedModel).filter(
                    TrainedModel.ticker == self.ticker
                ).order_by(TrainedModel.created_at.desc()).first()
                
                if not trained:
                    raise ValueError(f"Nenhum modelo treinado para {self.ticker}")
                
                self.model_path = Path(trained.model_path)
            finally:
                session.close()
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {self.model_path}")
        
        # Carregar checkpoint
        checkpoint = torch.load(self.model_path, weights_only=False)
        
        # Recriar modelo com mesma configuração
        self.model = StockLSTM.from_config(checkpoint["model_config"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        # Recriar scaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.data_min_ = np.array([checkpoint["scaler_min"]])
        self.scaler.data_max_ = np.array([checkpoint["scaler_max"]])
        self.scaler.scale_ = 1 / (self.scaler.data_max_ - self.scaler.data_min_)
        self.scaler.min_ = -self.scaler.data_min_ * self.scaler.scale_
        
        print(f"[OK] Modelo carregado: {self.model_path.name}")
        return True
    
    def predict(self, days: int = 1) -> Dict:
        """
        Faz previsão do preço para os próximos dias.
        
        Args:
            days: Número de dias para prever (default: 1)
        
        Returns:
            Dicionário com previsões
        """
        if self.model is None:
            self.load_model()
        
        # Preparar dados para inferência
        data_service = DataService(self.ticker)
        X, scaler = data_service.prepare_inference_data()
        
        # Usar o scaler do modelo treinado para consistência
        predictions = []
        current_sequence = X.clone()
        
        for day in range(days):
            # Fazer previsão
            with torch.no_grad():
                pred_scaled = self.model(current_sequence)
            
            # Inverter escala
            pred_value = self.scaler.inverse_transform(pred_scaled.numpy())[0, 0]
            predictions.append({
                "day": day + 1,
                "predicted_price": float(pred_value)
            })
            
            # Atualizar sequência para próxima previsão (rolling window)
            if days > 1:
                new_value = pred_scaled.view(1, 1, 1)
                current_sequence = torch.cat([current_sequence[:, 1:, :], new_value], dim=1)
        
        # Salvar previsão no banco
        self._save_prediction(predictions)
        
        return {
            "ticker": self.ticker,
            "model": self.model_path.name if self.model_path else "unknown",
            "predictions": predictions
        }
    
    def _save_prediction(self, predictions: List[Dict]):
        """Salva previsões no banco de dados."""
        session = SessionLocal()
        try:
            for pred in predictions:
                prediction = Prediction(
                    ticker=self.ticker,
                    predicted_price=pred["predicted_price"],
                    model_version=self.model_path.name if self.model_path else "unknown"
                )
                session.add(prediction)
            session.commit()
        finally:
            session.close()
    
    @staticmethod
    def list_models(ticker: str = None) -> List[Dict]:
        """
        Lista modelos treinados.
        
        Args:
            ticker: Filtrar por ticker (opcional)
        
        Returns:
            Lista de modelos
        """
        session = SessionLocal()
        try:
            query = session.query(TrainedModel)
            if ticker:
                query = query.filter(TrainedModel.ticker == ticker)
            
            models = query.order_by(TrainedModel.created_at.desc()).all()
            
            return [{
                "id": m.id,
                "ticker": m.ticker,
                "model_path": m.model_path,
                "rmse": m.rmse,
                "mae": m.mae,
                "epochs": m.epochs,
                "created_at": m.created_at.isoformat() if m.created_at else None
            } for m in models]
        finally:
            session.close()
    
    @staticmethod
    def get_prediction_history(ticker: str = None, limit: int = 50) -> List[Dict]:
        """
        Retorna histórico de previsões.
        
        Args:
            ticker: Filtrar por ticker (opcional)
            limit: Limite de registros
        
        Returns:
            Lista de previsões
        """
        session = SessionLocal()
        try:
            query = session.query(Prediction)
            if ticker:
                query = query.filter(Prediction.ticker == ticker)
            
            predictions = query.order_by(
                Prediction.prediction_date.desc()
            ).limit(limit).all()
            
            return [{
                "id": p.id,
                "ticker": p.ticker,
                "predicted_price": p.predicted_price,
                "actual_price": p.actual_price,
                "prediction_date": p.prediction_date.isoformat() if p.prediction_date else None,
                "model_version": p.model_version
            } for p in predictions]
        finally:
            session.close()


def predict_price(ticker: str, days: int = 1, model_path: str = None) -> Dict:
    """
    Função de conveniência para fazer previsão.
    
    Args:
        ticker: Símbolo da ação
        days: Dias para prever
        model_path: Caminho do modelo (opcional)
    
    Returns:
        Resultado da previsão
    """
    service = PredictService(ticker)
    if model_path:
        service.load_model(model_path)
    return service.predict(days)
