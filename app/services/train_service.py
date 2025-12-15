"""
Train Service - Serviço de treinamento do modelo LSTM

Inclui:
- Loop de treinamento com MSELoss
- Cálculo de métricas (RMSE, MAE)
- Salvamento do modelo treinado
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

from app.models.lstm import StockLSTM, create_model
from app.services.data_service import DataService
from app.database import SessionLocal, TrainedModel
from app.config import TRAINING_CONFIG, MODELS_DIR


class TrainService:
    """
    Serviço para treinamento de modelos LSTM.
    """
    
    def __init__(self, ticker: str):
        """
        Inicializa o serviço de treinamento.
        
        Args:
            ticker: Símbolo da ação para treinar
        """
        self.ticker = ticker
        self.data_service = DataService(ticker)
        self.model: Optional[StockLSTM] = None
        self.scaler = None
        self.history: Dict = {"train_loss": [], "val_loss": []}
    
    def train(
        self,
        epochs: int = None,
        batch_size: int = None,
        learning_rate: float = None,
        train_ratio: float = None
    ) -> Dict:
        """
        Treina o modelo LSTM.
        
        Args:
            epochs: Número de épocas (default: 100)
            batch_size: Tamanho do batch (default: 32)
            learning_rate: Taxa de aprendizado (default: 0.001)
            train_ratio: Proporção treino/teste (default: 0.8)
        
        Returns:
            Dicionário com métricas e caminho do modelo
        """
        # Usar config padrão se não especificado
        epochs = epochs or TRAINING_CONFIG["epochs"]
        batch_size = batch_size or TRAINING_CONFIG["batch_size"]
        learning_rate = learning_rate or TRAINING_CONFIG["learning_rate"]
        train_ratio = train_ratio or TRAINING_CONFIG["train_ratio"]
        
        print(f"\n[START] Iniciando treinamento para {self.ticker}")
        print(f"   Epochs: {epochs} | Batch: {batch_size} | LR: {learning_rate}")
        
        # 1. Preparar dados
        X_train, X_test, y_train, y_test, self.scaler = self.data_service.prepare_training_data(
            train_ratio=train_ratio
        )
        
        # Criar DataLoaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 2. Criar modelo
        self.model = create_model()
        print(f"   Modelo: {self.model.hidden_size} unidades, {self.model.num_layers} camadas")
        
        # 3. Configurar treinamento
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 4. Loop de treinamento
        print("\n[TRAINING] Treinando...")
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            self.history["train_loss"].append(avg_train_loss)
            
            # Validação
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_test)
                val_loss = criterion(val_pred, y_test).item()
                self.history["val_loss"].append(val_loss)
            
            # Log a cada 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"   Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # 5. Avaliar modelo
        metrics = self.evaluate(X_test, y_test)
        
        # 6. Salvar modelo
        model_path = self.save_model(metrics)
        
        return {
            "ticker": self.ticker,
            "epochs": epochs,
            "final_train_loss": self.history["train_loss"][-1],
            "final_val_loss": self.history["val_loss"][-1],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "mape": metrics["mape"],
            "model_path": str(model_path)
        }
    
    def evaluate(self, X_test: torch.Tensor, y_test: torch.Tensor) -> Dict:
        """
        Avalia o modelo com RMSE e MAE.
        
        Args:
            X_test: Features de teste
            y_test: Target de teste
        
        Returns:
            Dicionário com métricas
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test)
        
        # Inverter normalização
        y_pred = self.scaler.inverse_transform(predictions.numpy())
        y_true = self.scaler.inverse_transform(y_test.numpy())
        
        # Calcular métricas
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        
        # Calcular MAPE (Mean Absolute Percentage Error)
        # Evitar divisão por zero usando máscara
        mask = y_true != 0
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        
        print(f"\n[METRICS] Metricas de Avaliacao:")
        print(f"   RMSE: R$ {rmse:.2f}")
        print(f"   MAE:  R$ {mae:.2f}")
        print(f"   MAPE: {mape:.2f}%")
        
        return {
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "predictions": y_pred.flatten().tolist(),
            "actual": y_true.flatten().tolist()
        }
    
    def save_model(self, metrics: Dict) -> Path:
        """
        Salva o modelo treinado e registra no banco.
        
        Args:
            metrics: Métricas do modelo
        
        Returns:
            Caminho do arquivo .pt
        """
        # Garantir que diretório existe
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Nome do arquivo com timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{self.ticker}_{timestamp}.pt"
        model_path = MODELS_DIR / model_filename
        
        # Salvar pesos do modelo
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_config": self.model.get_config(),
            "scaler_min": float(self.scaler.data_min_[0]),
            "scaler_max": float(self.scaler.data_max_[0]),
            "metrics": metrics,
            "ticker": self.ticker,
            "timestamp": timestamp
        }, model_path)
        
        print(f"\n[SAVE] Modelo salvo: {model_path}")
        
        # Registrar no banco
        session = SessionLocal()
        try:
            trained_model = TrainedModel(
                ticker=self.ticker,
                model_path=str(model_path),
                train_loss=self.history["train_loss"][-1],
                val_loss=self.history["val_loss"][-1],
                rmse=metrics["rmse"],
                mae=metrics["mae"],
                mape=metrics["mape"],
                epochs=len(self.history["train_loss"])
            )
            session.add(trained_model)
            session.commit()
        finally:
            session.close()
        
        return model_path


def train_model(ticker: str, **kwargs) -> Dict:
    """
    Função de conveniência para treinar modelo.
    
    Args:
        ticker: Símbolo da ação
        **kwargs: Parâmetros de treinamento
    
    Returns:
        Resultado do treinamento
    """
    service = TrainService(ticker)
    return service.train(**kwargs)
