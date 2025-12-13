# ==============================================================================
# Dockerfile para Stock Predictor API
# Build: docker build -t stock-predictor-api .
# Run: docker run -p 8000:8000 stock-predictor-api
# ==============================================================================

FROM python:3.11-slim

# Argumentos de build
ARG APP_ENV=production

# Labels
LABEL maintainer="Tech Challenge Fase 4"
LABEL description="API de previsao de precos de acoes usando LSTM"
LABEL version="1.0.0"

# Variáveis de ambiente
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_ENV=${APP_ENV}

# Criar usuario nao-root para seguranca
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Diretorio de trabalho
WORKDIR /app

# Instalar dependencias do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copiar e instalar dependencias Python (cache de layers)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar codigo da aplicacao
COPY app/ ./app/
COPY ingest.py .

# Criar diretorios necessarios com permissoes
RUN mkdir -p /app/data/models \
    && chown -R appuser:appgroup /app

# Mudar para usuario nao-root
USER appuser

# Expor porta
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando de inicializacao
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
