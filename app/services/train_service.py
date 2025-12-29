"""
Train Service - Serviço de treinamento do modelo LSTM

Inclui:
- Loop de treinamento com MSELoss
- Cálculo de métricas (RMSE, MAE)
- Salvamento do modelo treinado
- Early Stopping para evitar overfitting
- Learning Rate Scheduler para otimização adaptativa
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List
import copy

from app.models.lstm import StockLSTM, create_model
from app.services.data_service import DataService
from app.database import SessionLocal, TrainedModel
from app.config import TRAINING_CONFIG, MODELS_DIR


class EarlyStopping:
    """
    Early Stopping para evitar overfitting.
    
    Para o treinamento quando val_loss não melhora por 'patience' epochs consecutivos.
    Salva o melhor modelo encontrado durante o treinamento.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0001, verbose: bool = True):
        """
        Args:
            patience: Número de epochs sem melhora antes de parar
            min_delta: Mínima melhora para considerar como progresso
            verbose: Se True, imprime mensagens de status
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Verifica se deve parar o treinamento.
        
        Args:
            val_loss: Loss de validação atual
            model: Modelo atual
            
        Returns:
            True se deve parar, False caso contrário
        """
        if self.best_loss is None:
            # Primeira época
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            return False
        
        if val_loss < self.best_loss - self.min_delta:
            # Melhorou! Resetar contador e salvar modelo
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            if self.verbose:
                print(f"   ✓ Novo melhor modelo! Val Loss: {val_loss:.6f}")
            return False
        else:
            # Não melhorou
            self.counter += 1
            if self.verbose and self.counter % 5 == 0:
                print(f"   ⚠ Sem melhora há {self.counter} epochs (patience: {self.patience})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"   🛑 Early Stopping ativado! Melhor val_loss: {self.best_loss:.6f}")
                return True
            return False
    
    def restore_best_model(self, model: nn.Module):
        """Restaura os pesos do melhor modelo encontrado."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


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
        self.scalers = None
        self.features_used = None
        self.history: Dict = {"train_loss": [], "val_loss": []}
    
    def train(
        self,
        epochs: int = None,
        batch_size: int = None,
        learning_rate: float = None,
        train_ratio: float = None,
        features: List[str] = None
    ) -> Dict:
        """
        Treina o modelo LSTM.
        
        Args:
            epochs: Número de épocas (default: 100)
            batch_size: Tamanho do batch (default: 32)
            learning_rate: Taxa de aprendizado (default: 0.001)
            train_ratio: Proporção treino/teste (default: 0.8)
        
            features: Lista de features para usar (default: ["close"])
        
        Returns:
            Dicionário com métricas e caminho do modelo
        """
        # Usar config padrão se não especificado
        epochs = epochs or TRAINING_CONFIG["epochs"]
        batch_size = batch_size or TRAINING_CONFIG["batch_size"]
        learning_rate = learning_rate or TRAINING_CONFIG["learning_rate"]
        train_ratio = train_ratio or TRAINING_CONFIG["train_ratio"]
        if features is None:
            features = ["close"]
        
        print(f"\n[START] Iniciando treinamento para {self.ticker}")
        print(f"   Epochs: {epochs} | Batch: {batch_size} | LR: {learning_rate}")
        print(f"   Features: {features}")
        
        # 1. Preparar dados (MULTIVARIADO)
        X_train, X_test, y_train, y_test, self.scalers, self.features_used = self.data_service.prepare_training_data(
            train_ratio=train_ratio,
            features=features
        )
        
        # Determinar input_size baseado no número de features
        n_features = X_train.shape[2]
        
        # Criar DataLoaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 2. Criar modelo com input_size correto para multi-feature
        self.model = create_model(input_size=n_features)
        print(f"   Modelo: {self.model.hidden_size} unidades, {self.model.num_layers} camadas, {n_features} features")
        
        # 3. Configurar treinamento
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 3.1 Configurar Learning Rate Scheduler
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min',           # Minimizar val_loss
            factor=0.5,           # Reduz LR pela metade
            patience=5,           # Espera 5 epochs sem melhora
            min_lr=1e-6           # LR mínimo
        )
        
        # 3.2 Configurar Early Stopping
        early_stopping = EarlyStopping(
            patience=TRAINING_CONFIG.get("early_stopping_patience", 15),
            min_delta=0.0001,
            verbose=True
        )
        
        # 4. Loop de treinamento com Early Stopping e LR Scheduler
        print("\n[TRAINING] Treinando com Early Stopping e LR Scheduler...")
        final_epoch = epochs
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                
                # Gradient Clipping para evitar gradientes explosivos
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
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
            
            # Atualizar LR Scheduler baseado no val_loss
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            
            # Log a cada 10 epochs ou quando LR muda
            if (epoch + 1) % 10 == 0 or epoch == 0:
                lr_info = f" | LR: {new_lr:.2e}" if new_lr != old_lr else ""
                print(f"   Epoch {epoch+1:3d}/{epochs} | Train: {avg_train_loss:.6f} | Val: {val_loss:.6f}{lr_info}")
            elif new_lr < old_lr:
                print(f"   📉 LR reduzido: {old_lr:.2e} -> {new_lr:.2e}")
            
            # Verificar Early Stopping
            if early_stopping(val_loss, self.model):
                final_epoch = epoch + 1
                print(f"\n   ⏹️ Treinamento parado na epoch {final_epoch} de {epochs}")
                break
        
        # 4.1 Restaurar melhor modelo encontrado
        if early_stopping.best_model_state is not None:
            early_stopping.restore_best_model(self.model)
            print(f"   ✅ Restaurado melhor modelo (val_loss: {early_stopping.best_loss:.6f})")
        
        # 5. Avaliar modelo
        metrics = self.evaluate(X_test, y_test)
        
        # 6. Salvar modelo
        model_path = self.save_model(metrics)
        
        return {
            "ticker": self.ticker,
            "epochs": len(self.history["train_loss"]),  # Campo obrigatório para TrainResponse
            "epochs_configured": epochs,
            "epochs_trained": len(self.history["train_loss"]),
            "early_stopped": early_stopping.early_stop,
            "best_val_loss": early_stopping.best_loss,
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
        
        # Inverter normalização usando scaler do close (target é sempre close)
        close_scaler = self.scalers["close"]
        y_pred = close_scaler.inverse_transform(predictions.numpy())
        y_true = close_scaler.inverse_transform(y_test.numpy())
        
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
        
        # Salvar pesos do modelo com scalers e features para multi-feature
        # Serializar scalers para o checkpoint
        scalers_data = {}
        for name, scaler in self.scalers.items():
            scalers_data[name] = {
                "data_min": float(scaler.data_min_[0]),
                "data_max": float(scaler.data_max_[0]),
                "feature_range": scaler.feature_range
            }
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_config": self.model.get_config(),
            "scalers": scalers_data,  # Dict com dados de todos os scalers
            "features": self.features_used,  # Lista de features usadas
            # Retrocompatibilidade: manter scaler_min/max do close
            "scaler_min": scalers_data["close"]["data_min"],
            "scaler_max": scalers_data["close"]["data_max"],
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
