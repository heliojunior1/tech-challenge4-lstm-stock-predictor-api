"""
LSTM Model - Modelo de rede neural para previsão de preços de ações

Arquitetura:
- Camada LSTM com 50 unidades (configurável)
- Dropout de 0.2 para evitar overfitting
- Camada Densa com 1 saída (regressão)
"""
import torch
import torch.nn as nn
from app.config import LSTM_CONFIG


class StockLSTM(nn.Module):
    """
    Modelo LSTM para previsão de preços de ações.
    
    Arquitetura padrão:
    - Input: (batch, seq_len=60, features=1)
    - LSTM: 50 unidades, 2 camadas
    - Dropout: 0.2
    - Output: 1 (preço previsto)
    """
    
    def __init__(
        self, 
        input_size: int = None,
        hidden_size: int = None,
        num_layers: int = None,
        dropout: float = None,
        output_size: int = None
    ):
        """
        Inicializa o modelo LSTM.
        
        Args:
            input_size: Features por timestep (default: 1, apenas Close)
            hidden_size: Unidades LSTM (default: 50)
            num_layers: Camadas LSTM empilhadas (default: 2)
            dropout: Taxa de dropout (default: 0.2)
            output_size: Saída (default: 1, regressão)
        """
        super().__init__()
        
        # Usar config padrão se não especificado
        self.input_size = input_size or LSTM_CONFIG["input_size"]
        self.hidden_size = hidden_size or LSTM_CONFIG["hidden_size"]
        self.num_layers = num_layers or LSTM_CONFIG["num_layers"]
        self.dropout = dropout or LSTM_CONFIG["dropout"]
        self.output_size = output_size or LSTM_CONFIG["output_size"]
        
        # Camadas LSTM
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        # Camada de Dropout adicional
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Camada Densa de Saída
        self.fc = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do modelo.
        
        Args:
            x: Input tensor - shape: (batch, seq_len, features)
        
        Returns:
            prediction: Previsão - shape: (batch, 1)
        """
        # Passar pela LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Pegar apenas o último timestep
        last_output = lstm_out[:, -1, :]
        
        # Aplicar dropout
        dropped = self.dropout_layer(last_output)
        
        # Camada densa final
        prediction = self.fc(dropped)
        
        return prediction
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Faz previsão em modo de avaliação (sem gradientes).
        
        Args:
            x: Input tensor
        
        Returns:
            prediction: Previsão
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    
    def get_config(self) -> dict:
        """Retorna configuração do modelo."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "output_size": self.output_size
        }
    
    @classmethod
    def from_config(cls, config: dict) -> "StockLSTM":
        """Cria modelo a partir de configuração."""
        return cls(**config)


def create_model(
    input_size: int = 1,
    hidden_size: int = 50,
    num_layers: int = 2,
    dropout: float = 0.2,
    output_size: int = 1
) -> StockLSTM:
    """
    Factory function para criar modelo LSTM.
    
    Args:
        input_size: Features por timestep
        hidden_size: Unidades LSTM
        num_layers: Camadas empilhadas
        dropout: Taxa de dropout
        output_size: Tamanho da saída
    
    Returns:
        Modelo StockLSTM inicializado
    """
    return StockLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        output_size=output_size
    )
