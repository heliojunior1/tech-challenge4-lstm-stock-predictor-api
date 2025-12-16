"""
Configurações da aplicação
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Carregar variaveis de ambiente
load_dotenv()

# Diretórios
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"

# Criar diretórios se não existirem
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Banco de dados
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/dados_mercado.db")

# API Keys para fontes de dados
# Alpha Vantage - Obter chave gratuita em: https://www.alphavantage.co/support/#api-key
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "demo")

# Configurações do modelo LSTM
LSTM_CONFIG = {
    "input_size": 1,
    "hidden_size": 50,
    "num_layers": 2,
    "dropout": 0.2,
    "output_size": 1,
    "window_size": 60,  # Dias para previsão
}

# Configurações de treinamento
TRAINING_CONFIG = {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "train_ratio": 0.8,
}

# Tickers padrão
DEFAULT_TICKERS = ["PETR4.SA", "DIS"]

# Features disponíveis para treinamento multivariado
AVAILABLE_FEATURES = {
    "close": {
        "name": "Preço de Fechamento",
        "required": True,
        "description": "Preço de fechamento do dia (obrigatório)"
    },
    "volume": {
        "name": "Volume",
        "required": False,
        "description": "Volume de negociação (confirma tendências)"
    },
    "rsi_14": {
        "name": "RSI (14)",
        "required": False,
        "description": "Índice de Força Relativa - 14 períodos (detecta reversões)"
    },
    "ema_20": {
        "name": "EMA (20)",
        "required": False,
        "description": "Média Móvel Exponencial - 20 períodos (direção da tendência)"
    },
}

DEFAULT_FEATURES = ["close"]  # Apenas close por padrão (retrocompatível)
