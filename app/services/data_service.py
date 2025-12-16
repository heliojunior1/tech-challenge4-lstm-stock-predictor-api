"""
Data Service - Serviço para fetch e processamento de dados do yfinance
Inclui scaling com MinMaxScaler e windowing para LSTM
Suporta múltiplas features: close, volume, rsi_14, ema_20
"""
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy.orm import Session
from typing import Tuple, Optional, List, Dict
import ta

from app.database import DadosMercado, SessionLocal
from app.config import LSTM_CONFIG, AVAILABLE_FEATURES, DEFAULT_FEATURES


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
    
    def calculate_features(self, df: pd.DataFrame, features: List[str] = None) -> pd.DataFrame:
        """
        Calcula indicadores técnicos selecionados.
        
        Args:
            df: DataFrame com dados OHLCV
            features: Lista de features a calcular (default: ["close"])
        
        Returns:
            DataFrame com as features calculadas
        """
        if features is None:
            features = DEFAULT_FEATURES
        
        # Garantir que close está incluído
        if "close" not in features:
            features = ["close"] + features
        
        result = pd.DataFrame(index=df.index)
        
        # Close (sempre incluído)
        result['close'] = df['close']
        
        # Volume
        if 'volume' in features:
            result['volume'] = df['volume']
        
        # RSI (14 períodos)
        if 'rsi_14' in features:
            result['rsi_14'] = ta.momentum.RSIIndicator(
                df['close'], window=14
            ).rsi()
        
        # EMA (20 períodos)
        if 'ema_20' in features:
            result['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        
        # Remover linhas com NaN (primeiros dias para indicadores)
        initial_len = len(result)
        result = result.dropna()
        dropped = initial_len - len(result)
        
        if dropped > 0:
            print(f"   -> Removidos {dropped} registros iniciais (NaN de indicadores)")
        
        return result
    
    def scale_data(self, close_prices: np.ndarray, scaler: Optional[MinMaxScaler] = None) -> Tuple[np.ndarray, MinMaxScaler]:
        """
        Normaliza os preços de fechamento para o intervalo [0, 1].
        
        LSTMs são sensíveis à escala dos dados!
        
        Args:
            close_prices: Array de preços de fechamento
            scaler: Scaler opcional (usar durante inferência para manter consistência)
        
        Returns:
            scaled_data: Dados normalizados
            scaler: Objeto scaler usado (o passado ou um novo fitado)
        """
        if scaler:
            # Inferência: Usar scaler já treinado (transform apenas)
            scaled_data = scaler.transform(close_prices.reshape(-1, 1))
            return scaled_data, scaler
        else:
            # Treinamento: Fitar novo scaler (fit_transform)
            scaled_data = self.scaler.fit_transform(close_prices.reshape(-1, 1))
            return scaled_data, self.scaler
    
    def create_sequences(self, data: np.ndarray, window_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cria sequências temporais para o LSTM (windowing).
        
        Suporta dados univariados e multivariados.
        Usa os últimos N dias para prever o dia N+1 (close).
        
        Args:
            data: Dados normalizados - shape: (samples,) ou (samples, features)
            window_size: Tamanho da janela (default: 60 dias)
        
        Returns:
            X: Features - shape: (samples, window_size, n_features)
            y: Target (close do dia N+1) - shape: (samples,)
        """
        if window_size is None:
            window_size = self.window_size
        
        # Garantir que data é 2D
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        n_features = data.shape[1]
        X, y = [], []
        
        for i in range(window_size, len(data)):
            X.append(data[i-window_size:i, :])  # Todas as features dos últimos N dias
            y.append(data[i, 0])                 # Apenas close do dia N+1
        
        X = np.array(X)  # Shape: (samples, window_size, n_features)
        y = np.array(y)  # Shape: (samples,)
        
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
        train_ratio: float = 0.8,
        features: List[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, MinMaxScaler], List[str]]:
        """
        Pipeline completo de preparação de dados para treinamento MULTIVARIADO.
        
        ORDEM CORRETA (evita data leakage):
        1. Busca dados do banco
        2. Calcula features técnicas (RSI, EMA, etc.)
        3. PRIMEIRO divide em treino/teste (temporal)
        4. Fita MinMaxScaler APENAS nos dados de treino (por feature)
        5. Aplica transform nos dados de teste
        6. Cria sequências (windowing) separadamente
        
        Args:
            train_ratio: Proporção para treino (default: 0.8)
            features: Lista de features a usar (default: ["close"])
        
        Returns:
            X_train, X_test, y_train, y_test, scalers (dict), features_used (list)
        """
        if features is None:
            features = DEFAULT_FEATURES.copy()
        
        # Garantir close está incluído
        if "close" not in features:
            features = ["close"] + features
        
        # 1. Buscar dados
        df = self.fetch_from_db()
        print(f"[DATA] {self.ticker}: {len(df)} registros brutos")
        
        # 2. Calcular features técnicas
        df_features = self.calculate_features(df, features)
        features_used = list(df_features.columns)
        n_features = len(features_used)
        
        print(f"   -> Features: {features_used}")
        print(f"   -> {len(df_features)} registros após calcular indicadores")
        
        # 3. PRIMEIRO: Split temporal nos dados BRUTOS
        split_idx = int(len(df_features) * train_ratio)
        train_df = df_features.iloc[:split_idx]
        test_df = df_features.iloc[split_idx:]
        
        print(f"   -> Split: Treino={len(train_df)} | Teste={len(test_df)}")
        
        # 4. Fitar scaler por feature APENAS nos dados de treino (evita data leakage!)
        scalers = {}
        train_scaled = np.zeros((len(train_df), n_features))
        test_scaled = np.zeros((len(test_df), n_features))
        
        for i, col in enumerate(features_used):
            scaler = MinMaxScaler(feature_range=(0, 1))
            train_scaled[:, i] = scaler.fit_transform(train_df[[col]]).flatten()
            test_scaled[:, i] = scaler.transform(test_df[[col]]).flatten()
            scalers[col] = scaler
        
        # 5. Criar sequências separadamente
        X_train, y_train = self.create_sequences(train_scaled)
        
        # Para teste, precisamos incluir os últimos window_size dias do treino
        # para criar a primeira sequência de teste corretamente
        test_with_context = np.concatenate([
            train_scaled[-self.window_size:],  # Contexto do fim do treino
            test_scaled
        ])
        X_test, y_test = self.create_sequences(test_with_context)
        
        print(f"   -> Sequências: Treino={len(X_train)} | Teste={len(X_test)} (janela: {self.window_size})")
        print(f"   -> Shape X: {X_train.shape} (samples, window, features)")
        
        # Converter para tensores PyTorch
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
        
        return X_train, X_test, y_train, y_test, scalers, features_used
    
    def prepare_inference_data(
        self, 
        scalers: Dict[str, MinMaxScaler] = None,
        features: List[str] = None
    ) -> torch.Tensor:
        """
        Prepara os últimos N dias para fazer uma previsão MULTIVARIADA.
        
        Args:
            scalers: Dict de scalers treinados (um por feature)
            features: Lista de features a usar (mesmas do treino)
            
        Returns:
            X: Tensor com os últimos window_size dias - shape: (1, window_size, n_features)
        """
        if features is None:
            features = DEFAULT_FEATURES.copy()
        if scalers is None:
            scalers = {}
        
        # Buscar dados
        df = self.fetch_from_db()
        
        # Calcular features técnicas
        df_features = self.calculate_features(df, features)
        features_used = list(df_features.columns)
        n_features = len(features_used)
        
        # Normalizar cada feature usando seu scaler
        scaled_data = np.zeros((len(df_features), n_features))
        
        for i, col in enumerate(features_used):
            if col in scalers:
                scaled_data[:, i] = scalers[col].transform(df_features[[col]]).flatten()
            else:
                # Fallback: criar novo scaler se não existir (não ideal, mas evita erro)
                temp_scaler = MinMaxScaler()
                scaled_data[:, i] = temp_scaler.fit_transform(df_features[[col]]).flatten()
        
        # Pegar últimos window_size dias
        last_sequence = scaled_data[-self.window_size:]
        
        # Reshape para LSTM: (1, window_size, n_features)
        X = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0)
        
        return X
