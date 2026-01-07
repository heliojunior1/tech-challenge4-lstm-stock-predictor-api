# 📈 Stock Predictor API

API de previsão de preços de ações usando redes neurais LSTM, desenvolvida para o **Tech Challenge Fase 4** da Pós-Tech FIAP.

# 📈 Acesso em produção

https://stock-predictor-api-x2o4.onrender.com

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


## 🖥️ Frontend (Dashboard)

A API agora acompanha uma interface web simples para facilitar o uso.

### Acesso

- **Dashboard**: [http://localhost:8000/](http://localhost:8000/)

### Funcionalidades do Frontend

1.  **Dashboard**: Visão geral de modelos treinados e previsões recentes.
2.  **Ingestão**: Formulário para baixar dados históricos (com suporte a datas).
3.  **Treinamento**: Interface para treinar novos modelos (síncrono ou assíncrono). A criação do assíncrono é em virtude que alguns treinamentos o periodo pode ser longo e o servidor render tem uma limitacao de 30s para uma chamada feita pelo gateway na versão gratuita . 
4.  **Predição**:
    *   **Padrão**: Prever usando ações cadastradas.
    *   **Custom**: Prever usando dados históricos colados manualmente.
5.  **Histórico**: Visualizar todas as previsões realizadas.

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
│       ├── monitoring.py    # Métricas Prometheus
│       └── frontend.py      # [NEW] Rotas do Frontend
│   ├── templates/           # [NEW] Arquivos HTML (Jinja2)
│   └── static/              # [NEW] Arquivos estáticos (CSS/JS)
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


# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual (PowerShell)
.\venv\Scripts\Activate.ps1

# OU ativar ambiente virtual (CMD)
venv\Scripts\activate.bat
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
| POST | `/api/v1/ingest/{ticker}` | Ingerir dados (start_date, end_date) |
| POST | `/api/v1/train/{ticker}` | Treinar modelo |
| POST | `/api/v1/predict/{ticker}` | Previsão por ticker (dados do banco) |
| POST | `/api/v1/predict/custom` | **Previsão com dados do usuário** |
| GET | `/api/v1/models` | Listar modelos |
| GET | `/api/v1/predictions/history` | Histórico |

### Exemplos de Uso

#### Ingerir Dados (com datas específicas)
```bash
# Usando datas específicas (formato YYYY-MM-DD)
curl -X POST "http://localhost:8000/api/v1/ingest/PETR4.SA?start_date=2018-01-01&end_date=2024-07-20"

# Sem datas (usa últimos 2 anos por padrão)
curl -X POST "http://localhost:8000/api/v1/ingest/PETR4.SA"
```

#### Treinar Modelo
```bash
curl -X POST "http://localhost:8000/api/v1/train/PETR4.SA" \
  -H "Content-Type: application/json" \
  -d '{"epochs": 50, "batch_size": 32}'
```

#### Fazer Previsão (por ticker)
```bash
curl -X POST "http://localhost:8000/api/v1/predict/PETR4.SA" \
  -H "Content-Type: application/json" \
  -d '{"days": 1}'
```

#### Fazer Previsão (com dados do usuário)

Este endpoint atende ao requisito do Tech Challenge:
> "A API deve permitir que o usuário forneça dados históricos de preços e receba previsões"

```bash
curl -X POST "http://localhost:8000/api/v1/predict/custom" \
  -H "Content-Type: application/json" \
  -d '{
    "historical_prices": [30.5, 30.7, 30.9, 31.1, 31.3, 31.5, 31.7, 31.9, 32.1, 32.3,
                          32.5, 32.7, 32.9, 33.1, 33.3, 33.5, 33.7, 33.9, 34.1, 34.3,
                          34.5, 34.7, 34.9, 35.1, 35.3, 35.5, 35.7, 35.9, 36.1, 36.3,
                          36.5, 36.7, 36.9, 37.1, 37.3, 37.5, 37.7, 37.9, 38.1, 38.3,
                          38.5, 38.7, 38.9, 39.1, 39.3, 39.5, 39.7, 39.9, 40.1, 40.3,
                          40.5, 40.7, 40.9, 41.1, 41.3, 41.5, 41.7, 41.9, 42.1, 42.3],
    "days": 3,
    "model_ticker": "PETR4.SA"
  }'
```

**Parâmetros:**
- `historical_prices`: Lista de preços históricos (mínimo 60 valores)
- `days`: Número de dias para prever (1-30)
- `model_ticker`: Ticker do modelo a ser usado

## 🧠 Modelo LSTM

### Arquitetura

```
Input (batch, 60, n_features)
         ▼
┌────────────────────────────┐
│  nn.LSTM                   │
│  • hidden_size: 50         │
│  • num_layers: 2           │
│  • dropout: 0.2            │
│  • bias: True              │
└────────────────────────────┘
         ▼
    Último timestep
         ▼
┌────────────────────────────┐
│  nn.Dropout(0.2)           │
└────────────────────────────┘
         ▼
┌────────────────────────────┐
│  nn.Linear(50, 1)          │
│  • bias: True              │
│  • ativação: Nenhuma       │
└────────────────────────────┘
         ▼
Output (batch, 1) → preço previsto
```

### Funções de Ativação (Internas da LSTM)

A LSTM usa **4 gates** com ativações específicas (implementação PyTorch):

| Gate | Ativação | Fórmula | Propósito |
|------|----------|---------|-----------|
| **Forget Gate** | Sigmoid | `σ(Wf·[ht-1, xt] + bf)` | Decide o que esquecer |
| **Input Gate** | Sigmoid | `σ(Wi·[ht-1, xt] + bi)` | Decide o que atualizar |
| **Candidate** | Tanh | `tanh(Wc·[ht-1, xt] + bc)` | Cria novos candidatos |
| **Output Gate** | Sigmoid | `σ(Wo·[ht-1, xt] + bo)` | Decide a saída |

- **Sigmoid (0-1)**: Atua como "porta" - 0 = bloqueia, 1 = permite
- **Tanh (-1 a 1)**: Permite ajustes bidirecionais

### Parâmetros do Modelo

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `input_size` | 1-4 | Features por timestep (adjclose, volume, rsi, ema) |
| `hidden_size` | 50 | Neurônios LSTM por camada |
| `num_layers` | 2 | Camadas empilhadas |
| `dropout` | 0.2 | 20% regularização |
| `output_size` | 1 | Preço previsto |
| `window_size` | 60 | Dias de input |
| `bias` | True | Offset aprendível em cada gate |
Obs: Foi utilizado como padrao o adjclose, visto que é o close normalizado considerando os dividendos. 


### Loss e Otimizador

| Componente | Implementação | Motivo |
|------------|---------------|--------|
| **Loss** | `MSELoss` | Regressão - penaliza erros quadráticos |
| **Otimizador** | `Adam` | Converge rápido, adapta LR por parâmetro |
| **Learning Rate** | 0.001 | Padrão conservador |

### Fluxo de Treinamento (Forward + Backpropagation)

```
┌─────────────────────────────────────────────────────────────────┐
│                    1. FORWARD PASS                              │
├─────────────────────────────────────────────────────────────────┤
│   Input X (batch, 60, n_features)                               │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐                    │
│   │  LSTM   │ →  │ Dropout │ →  │ Linear  │ →  Previsão ŷ      │
│   └─────────┘    └─────────┘    └─────────┘                    │
│                                       │                         │
│   Valor Real y ──────────────────────►│                         │
│                                       ▼                         │
│                                   MSELoss                       │
│                               Loss = (ŷ - y)²                   │
└─────────────────────────────────────────────────────────────────┘
                             │
                             │ loss.backward()
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 2. BACKPROPAGATION                              │
├─────────────────────────────────────────────────────────────────┤
│   Gradientes calculados de trás para frente (Chain Rule):      │
│                                                                 │
│   Loss → ∂Loss/∂w_linear → ∂Loss/∂w_lstm2 → ∂Loss/∂w_lstm1     │
│                                                                 │
│   Cada peso sabe "quanto contribuiu" para o erro               │
└─────────────────────────────────────────────────────────────────┘
                             │
                             │ optimizer.step()
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│               3. ATUALIZAÇÃO DE PESOS                           │
├─────────────────────────────────────────────────────────────────┤
│   w_novo = w_antigo - learning_rate × gradiente                 │
│                                                                 │
│   Adam: ajusta LR adaptativamente por parâmetro                 │
└─────────────────────────────────────────────────────────────────┘
```

### Pré-processamento

1. **Scaling**: MinMaxScaler (normalização 0-1)
2. **Windowing**: Janela deslizante de 60 dias
3. **Split**: 80% treino, 20% teste (temporal)

#### ⚠️ Prevenção de Data Leakage

O pipeline de dados foi cuidadosamente projetado para **evitar data leakage** na normalização:

```
❌ Errado: Normalizar → Dividir (scaler "vê" dados de teste)
✅ Correto: Dividir → Normalizar treino → Aplicar no teste
```

**Implementação em `data_service.py`:**

1. **Split primeiro**: Dados brutos são divididos em 80/20 **antes** de qualquer processamento
2. **Fit apenas no treino**: `scaler.fit()` é chamado **apenas** nos dados de treino
3. **Transform no teste**: Dados de teste usam `scaler.transform()` (não refit)
4. **Contexto preservado**: Últimos 60 dias do treino são usados como contexto inicial para sequências de teste

Isso garante que o modelo nunca tenha acesso a informações do futuro durante o treinamento.

### Métricas de Avaliação

- **RMSE**: Root Mean Squared Error (em R$)
- **MAE**: Mean Absolute Error (em R$)
- **MAPE**: Mean Absolute Percentage Error (%)

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
- Swagger UI: http://localhost:8000/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

### Ver Logs dos Containers

```powershell
# Logs da API (tempo real)
docker logs stock-predictor-api -f

# Logs do Prometheus
docker logs stock-predictor-prometheus -f

# Logs do Grafana
docker logs stock-predictor-grafana -f

# Logs de TODOS os serviços
docker compose logs -f

# Últimas 50 linhas
docker compose logs --tail 50
```

> **Dica**: Use `Ctrl+C` para sair do modo de logs em tempo real.

### Comandos Úteis Docker

| Comando | Descrição |
|---------|-----------|
| `docker compose up -d` | Iniciar todos os serviços |
| `docker compose down` | Parar todos os serviços |
| `docker compose restart` | Reiniciar serviços |
| `docker compose logs -f` | Ver logs em tempo real |
| `docker ps` | Listar containers rodando |
| `docker compose up -d --build` | Rebuild e reiniciar |

### Executar Endpoints via Docker

Com os containers rodando (`docker compose up -d`), use o Swagger UI ou os comandos abaixo:

**Via Swagger (Recomendado):**
1. Acesse http://localhost:8000/docs
2. Clique no endpoint desejado
3. Clique em "Try it out"
4. Preencha os parâmetros
5. Clique em "Execute"

**Via PowerShell:**
```powershell
# Verificar se API está rodando
Invoke-RestMethod -Uri "http://localhost:8000/health"

# Ingerir dados
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/ingest/AAPL?period=1y" -Method Post

# Treinar modelo
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/train/AAPL" -Method Post -ContentType "application/json" -Body '{"epochs": 5}'

# Fazer previsão
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/predict/AAPL" -Method Post -ContentType "application/json" -Body '{"days": 1}'

# Listar modelos
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/models"
```


## ☁️ Deploy no Render

1. Faça push do projeto para GitHub
2. Acesse https://render.com
3. New → Blueprint
4. Conecte o repositório
5. Render detectará o `render.yaml` automaticamente
6. Clique em Apply

> **Nota**: Plano free tem 512MB RAM e entra em sleep após 15min.

## 📊 Monitoramento

### Métricas Prometheus Disponíveis

```
stock_predictor_requests_total        # Total de requisições
stock_predictor_predictions_total     # Total de previsões realizadas
stock_predictor_trainings_total       # Total de treinamentos
stock_predictor_request_latency_seconds   # Latência das requisições
stock_predictor_training_duration_seconds # Duração dos treinamentos
stock_predictor_models_count          # Número de modelos treinados
stock_predictor_last_prediction_price # Último preço previsto
```

### Configurar Prometheus

1. Acesse http://localhost:9090
2. Vá em **Status → Targets** para verificar se os targets estão UP
3. Na aba **Graph**, digite uma query:

**Queries que sempre funcionam:**
```promql
# Informações do Python
python_info

# Memória do processo
process_resident_memory_bytes

# CPU utilizada
process_cpu_seconds_total

# Garbage Collector
python_gc_collections_total
```

**Queries personalizadas (após usar a API):**
```promql
# Total de previsões
stock_predictor_predictions_total

# Último preço previsto por ticker
stock_predictor_last_prediction_price

# Total de treinamentos
stock_predictor_trainings_total

# Duração dos treinamentos
stock_predictor_training_duration_seconds_sum
```

> **Nota**: As métricas `stock_predictor_*` só aparecem após a primeira utilização da API (previsão, treino, etc.)

### Gerar Métricas via Aplicação

Para que as métricas apareçam no Prometheus, você precisa **usar a API**. Siga estes passos:

**Opção 1: Via Swagger UI (Interface Gráfica)**

1. Acesse http://localhost:8000/docs
2. **Ingerir dados** (necessário antes de treinar):
   - Clique em `POST /api/v1/ingest/{ticker}`
   - Clique em "Try it out"
   - Digite o ticker: `AAPL` (ou `PETR4.SA`)
   - Clique em "Execute"
3. **Treinar modelo**:
   - Clique em `POST /api/v1/train/{ticker}`
   - Clique em "Try it out"
   - Digite o ticker: `AAPL`
   - No body, use: `{"epochs": 5}`
   - Clique em "Execute"
4. **Fazer previsão**:
   - Clique em `POST /api/v1/predict/{ticker}`
   - Clique em "Try it out"
   - Digite o ticker: `AAPL`
   - No body, use: `{"days": 1}`
   - Clique em "Execute"

**Opção 2: Via Linha de Comando (PowerShell)**

```powershell
# 1. Ingerir dados (baixar 1 ano de histórico)
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/ingest/AAPL?period=1y" -Method Post

# 2. Treinar modelo (5 epochs para teste rápido)
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/train/AAPL" -Method Post -ContentType "application/json" -Body '{"epochs": 5}'

# 3. Fazer previsão
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/predict/AAPL" -Method Post -ContentType "application/json" -Body '{"days": 1}'
```

**Opção 3: Via cURL (Linux/Mac)**

```bash
# 1. Ingerir dados
curl -X POST "http://localhost:8000/api/v1/ingest/AAPL?period=1y"

# 2. Treinar modelo
curl -X POST "http://localhost:8000/api/v1/train/AAPL" \
  -H "Content-Type: application/json" \
  -d '{"epochs": 5}'

# 3. Fazer previsão
curl -X POST "http://localhost:8000/api/v1/predict/AAPL" \
  -H "Content-Type: application/json" \
  -d '{"days": 1}'
```

Após executar esses comandos, acesse http://localhost:9090 e verifique as métricas!

### Configurar Grafana

1. Acesse http://localhost:3000
2. Login: `admin` / `admin`

**Adicionar Data Source:**
1. Clique em ⚙️ → **Data Sources**
2. Clique em **Add data source**
3. Selecione **Prometheus**
4. Em URL digite: `http://prometheus:9090`
5. Clique em **Save & Test**

**Criar Dashboard:**
1. Clique em **+** → **Dashboard**
2. Clique em **Add visualization**
3. Selecione **Prometheus** como data source
4. Digite a query (ex: `process_resident_memory_bytes`)
5. Clique em **Apply**

**Painéis sugeridos:**
| Métrica | Tipo | Descrição |
|---------|------|-----------|
| `process_resident_memory_bytes` | Gauge | Memória RAM usada |
| `process_cpu_seconds_total` | Counter | CPU acumulada |
| `stock_predictor_predictions_total` | Counter | Previsões realizadas |
| `stock_predictor_last_prediction_price` | Gauge | Último preço previsto |

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

---

## 🔬 Otimizações do Modelo (Data Science)

Para garantir a melhor performance e evitar problemas comuns em séries temporais (como overfitting e instabilidade), o projeto implementa as seguintes técnicas:

### 1. Parada Antecipada (Early Stopping)
O modelo monitora o erro nos dados de validação. Se o erro parar de cair por **15 épocas** consecutivas, o treinamento é interrompido. Isso evita que o modelo "decore" o passado (overfitting) e economiza recursos computacionais.

### 2. Agendador de Taxa de Aprendizado (LR Scheduler)
Utilizamos o `ReduceLROnPlateau`. Se o modelo atingir um platô (estacionar), o agendador reduz a taxa de aprendizado pela metade. Isso permite que o modelo faça ajustes mais finos conforme se aproxima do ponto ideal.

### 3. Corte de Gradiente (Gradient Clipping)
Em LSTMs, os gradientes podem explodir em sequências longas. O projeto limita a norma dos gradientes em **1.0**, garantindo que as atualizações dos pesos sejam sempre estáveis.

### 4. Monitoramento Automático de Métricas
Toda execução de treinamento calcula e armazena:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error) - *Ideal para comparar erros entre diferentes preços de ações.*
