# 📈 Stock Predictor API

API de previsão de preços de ações usando redes neurais LSTM, desenvolvida para o **Tech Challenge Fase 4** da Pós-Tech FIAP.

## 🎯 Objetivo

Desenvolver uma API que utiliza modelos LSTM para prever preços de ações da bolsa de valores, com monitoramento via Prometheus e deploy na nuvem.

## 🛠️ Tecnologias

| Tecnologia | Uso |
|------------|-----|
| **FastAPI** | Framework web para a API REST |
| **PyTorch** | Deep Learning (modelo LSTM) |
| **SQLite** | Banco de dados local |
| **yfinance** | Coleta de dados de mercado |
| **Prometheus** | Monitoramento e métricas |
| **Docker** | Containerização |
| **Render** | Deploy na nuvem |

## 📁 Estrutura do Projeto

```
stock-predictor-api/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app principal
│   ├── config.py            # Configurações
│   ├── database.py          # Conexão SQLite
│   ├── models/
│   │   ├── lstm.py          # Modelo LSTM
│   │   └── schemas.py       # Pydantic schemas
│   ├── services/
│   │   ├── data_service.py  # Processamento de dados
│   │   ├── train_service.py # Treinamento
│   │   └── predict_service.py
│   └── routers/
│       ├── training.py      # Endpoints de treino
│       ├── inference.py     # Endpoints de previsão
│       └── monitoring.py    # Métricas Prometheus
├── data/models/             # Modelos treinados (.pt)
├── tests/
├── ingest.py                # Script de ingestão
├── train_test.py            # Script de teste
├── Dockerfile
├── docker-compose.yml
├── prometheus.yml
├── render.yaml
└── requirements.txt
```

## 🚀 Quick Start

### Pré-requisitos

- Python 3.11+
- Anaconda (recomendado) ou pip

### 1. Ativar Ambiente

```powershell
# Windows com Anaconda
& C:\Users\junio\anaconda3\shell\condabin\conda-hook.ps1
conda activate base
cd c:\Users\junio\tech-challenge4-lstm-stock-predictor-api
```

### 2. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 3. Ingerir Dados

```bash
# Baixar dados de PETR4.SA e AAPL (2 anos)
python ingest.py PETR4.SA AAPL

# Ou com período customizado
python ingest.py VALE3.SA --period 5y
```

### 4. Treinar e Testar

```bash
# Treinar modelo e fazer previsão
python train_test.py PETR4.SA --epochs 50
```

### 5. Iniciar a API

```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 6. Acessar a API

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health
- **Métricas**: http://localhost:8000/metrics

## 📡 Endpoints da API

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/health` | Health check |
| GET | `/metrics` | Métricas Prometheus |
| POST | `/api/v1/ingest/{ticker}` | Ingerir dados |
| POST | `/api/v1/train/{ticker}` | Treinar modelo |
| POST | `/api/v1/predict/{ticker}` | Fazer previsão |
| GET | `/api/v1/models` | Listar modelos |
| GET | `/api/v1/predictions/history` | Histórico |

### Exemplos de Uso

#### Ingerir Dados
```bash
curl -X POST "http://localhost:8000/api/v1/ingest/PETR4.SA?period=2y"
```

#### Treinar Modelo
```bash
curl -X POST "http://localhost:8000/api/v1/train/PETR4.SA" \
  -H "Content-Type: application/json" \
  -d '{"epochs": 50, "batch_size": 32}'
```

#### Fazer Previsão
```bash
curl -X POST "http://localhost:8000/api/v1/predict/PETR4.SA" \
  -H "Content-Type: application/json" \
  -d '{"days": 1}'
```

## 🧠 Modelo LSTM

### Arquitetura

- **Input**: Janela de 60 dias (preço de fechamento)
- **LSTM**: 50 unidades, 2 camadas, dropout 0.2
- **Output**: 1 valor (preço previsto)

### Pré-processamento

1. **Scaling**: MinMaxScaler (normalização 0-1)
2. **Windowing**: Janela deslizante de 60 dias
3. **Split**: 80% treino, 20% teste (temporal)

### Métricas de Avaliação

- **RMSE**: Root Mean Squared Error (em R$)
- **MAE**: Mean Absolute Error (em R$)

## 🐳 Docker

### Build e Run

```bash
# Build
docker build -t stock-predictor-api .

# Run
docker run -p 8000:8000 stock-predictor-api
```

### Docker Compose (API + Prometheus + Grafana)

```bash
docker compose up -d
```

Serviços:
- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

## ☁️ Deploy no Render

1. Faça push do projeto para GitHub
2. Acesse https://render.com
3. New → Blueprint
4. Conecte o repositório
5. Render detectará o `render.yaml` automaticamente
6. Clique em Apply

> **Nota**: Plano free tem 512MB RAM e entra em sleep após 15min.

## 📊 Monitoramento

### Métricas Prometheus

```
stock_predictor_requests_total
stock_predictor_predictions_total
stock_predictor_trainings_total
stock_predictor_request_latency_seconds
stock_predictor_training_duration_seconds
```

### Dashboard Grafana

1. Acesse http://localhost:3000
2. Add data source → Prometheus → URL: http://prometheus:9090
3. Import dashboard ou crie painéis personalizados

## 🧪 Testes

```bash
# Executar testes
pytest tests/ -v

# Com cobertura
pytest tests/ --cov=app --cov-report=html
```

## 📝 Licença

MIT License

## 👨‍💻 Autor

**Tech Challenge Fase 4** - Pós-Tech FIAP Machine Learning Engineering
