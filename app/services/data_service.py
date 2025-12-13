"""
Data Service - Serviço para fetch e processamento de dados do yfinance
Inclui scaling com MinMaxScaler e windowing para LSTM
"""
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy.orm import Session
from typing import Tuple, Optional

from app.database import DadosMercado, SessionLocal
from app.config import LSTM_CONFIG


class DataService:
    """
    Serviço para processamento de dados de séries temporais.
    Inclui normalização (MinMaxScaler) e criação de sequências (windowing).
    """
    
    def __init__(self, ticker: str):
        """
        Inicializa o serviço de dados.
        
        Args:
            ticker: Símbolo da ação (ex: PETR4.SA)
        """
        self.ticker = ticker
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.window_size = LSTM_CONFIG["window_size"]
    
    def fetch_from_db(self, session: Optional[Session] = None) -> pd.DataFrame:
        """
        Busca dados históricos do banco SQLite.
        
        Returns:
            DataFrame com colunas: data, open, high, low, close, volume
        """
        close_session = False
        if session is None:
            session = SessionLocal()
            close_session = True
        
        try:
            dados = session.query(DadosMercado).filter(
                DadosMercado.ticker == self.ticker
            ).order_by(DadosMercado.data.asc()).all()
            
            if not dados:
                raise ValueError(f"Nenhum dado encontrado para {self.ticker}")
            
            df = pd.DataFrame([{
                'data': d.data,
                'open': d.open,
                'high': d.high,
                'low': d.low,
                'close': d.close,
                'volume': d.volume
            } for d in dados])
            
            return df
        finally:
            if close_session:
                session.close()
    
    def scale_data(self, close_prices: np.ndarray) -> Tuple[np.ndarray, MinMaxScaler]:
        """
        Normaliza os preços de fechamento para o intervalo [0, 1].
        
        LSTMs são sensíveis à escala dos dados!
        
        Args:
            close_prices: Array de preços de fechamento
        
        Returns:
            scaled_data: Dados normalizados
            scaler: Objeto scaler para inverter a transformação
        """
        scaled_data = self.scaler.fit_transform(close_prices.reshape(-1, 1))
        return scaled_data, self.scaler
    
    def create_sequences(self, data: np.ndarray, window_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cria sequências temporais para o LSTM (windowing).
        
        Usa os últimos N dias para prever o dia N+1.
        Exemplo: window_size=60 usa 60 dias para prever o dia 61.
        
        Args:
            data: Dados normalizados
            window_size: Tamanho da janela (default: 60 dias)
        
        Returns:
            X: Features (sequências de entrada) - shape: (samples, window_size, 1)
            y: Target (valor a prever) - shape: (samples,)
        """
        if window_size is None:
            window_size = self.window_size
        
        X, y = [], []
        for i in range(window_size, len(data)):
            X.append(data[i-window_size:i, 0])  # Últimos N dias
            y.append(data[i, 0])                 # Dia N+1
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape para LSTM: (samples, timesteps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def train_test_split_temporal(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        train_ratio: float = 0.8
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Divide os dados respeitando a ordem temporal.
        
        IMPORTANTE: Em séries temporais, nunca use split aleatório!
        
        Args:
            X: Features
            y: Target
            train_ratio: Proporção para treino (default: 80%)
        
        Returns:
            X_train, X_test, y_train, y_test (como tensores PyTorch)
        """
        split_idx = int(len(X) * train_ratio)
        
        X_train = torch.tensor(X[:split_idx], dtype=torch.float32)
        X_test = torch.tensor(X[split_idx:], dtype=torch.float32)
        y_train = torch.tensor(y[:split_idx], dtype=torch.float32).unsqueeze(1)
        y_test = torch.tensor(y[split_idx:], dtype=torch.float32).unsqueeze(1)
        
        return X_train, X_test, y_train, y_test
    
    def prepare_training_data(
        self, 
        train_ratio: float = 0.8
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, MinMaxScaler]:
        """
        Pipeline completo de preparação de dados para treinamento.
        
        1. Busca dados do banco
        2. Aplica MinMaxScaler
        3. Cria sequências (windowing)
        4. Divide em treino/teste (temporal)
        
        Returns:
            X_train, X_test, y_train, y_test, scaler
        """
        # 1. Buscar dados
        df = self.fetch_from_db()
        close_prices = df['close'].values
        
        print(f"[DATA] {self.ticker}: {len(close_prices)} registros")
        
        # 2. Normalizar (MinMaxScaler)
        scaled_data, scaler = self.scale_data(close_prices)
        
        # 3. Criar sequências (windowing)
        X, y = self.create_sequences(scaled_data)
        
        print(f"   -> {len(X)} sequencias criadas (janela: {self.window_size} dias)")
        
        # 4. Split temporal
        X_train, X_test, y_train, y_test = self.train_test_split_temporal(X, y, train_ratio)
        
        print(f"   -> Treino: {len(X_train)} | Teste: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test, scaler
    
    def prepare_inference_data(self) -> Tuple[torch.Tensor, MinMaxScaler]:
        """
        Prepara os últimos N dias para fazer uma previsão.
        
        Returns:
            X: Tensor com os últimos window_size dias
            scaler: Scaler para inverter a previsão
        """
        # Buscar dados
        df = self.fetch_from_db()
        close_prices = df['close'].values
        
        # Normalizar
        scaled_data, scaler = self.scale_data(close_prices)
        
        # Pegar últimos window_size dias
        last_sequence = scaled_data[-self.window_size:]
        
        # Reshape para LSTM: (1, window_size, 1)
        X = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0)
        
        return X, scaler
