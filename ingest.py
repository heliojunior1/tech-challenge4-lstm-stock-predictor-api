"""
ingest.py - Script de ingestão de dados do Yahoo Finance para SQLite

Uso:
    python ingest.py                    # Baixa tickers padrão (PETR4.SA, DIS)
    python ingest.py AAPL MSFT GOOGL    # Baixa tickers específicos
    python ingest.py --period 5y VALE3.SA  # Baixa com período customizado
"""
import sys
import argparse
import yfinance as yf
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Carregar variaveis de ambiente do .env
load_dotenv()

# Adicionar diretório raiz ao path para imports
sys.path.insert(0, str(__file__).rsplit('\\', 1)[0] if '\\' in str(__file__) else '.')

from app.database import SessionLocal, DadosMercado, init_db
import requests
import time
from app.config import DEFAULT_TICKERS, ALPHAVANTAGE_API_KEY


def download_alphavantage(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Baixa dados da API Alpha Vantage como fallback.
    """
    print(f"   [FALLBACK] Tentando Alpha Vantage para {ticker}...")
    
    if not ALPHAVANTAGE_API_KEY or ALPHAVANTAGE_API_KEY == "demo":
        print("   [AVISO] API Key do Alpha Vantage não configurada (usando 'demo').")
        if ticker != "IBM": # Demo só funciona bem com IBM
            print("   [ERRO] Alpha Vantage Demo só suporta ticker IBM.")
            return pd.DataFrame()
            
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "apikey": ALPHAVANTAGE_API_KEY,
        "outputsize": "compact", # Free tier limitação: compact=100 dados
        "datatype": "json"
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if "Error Message" in data:
            if "premium" in data.get("Error Message", "").lower():
                 print("   [AVISO] Alpha Vantage Premium requerido para histórico completo.")
            raise ValueError(f"Alpha Vantage Error: {data['Error Message']}")
        if "Information" in data:
             if "premium" in data.get("Information", "").lower():
                 print(f"   [AVISO] Alpha Vantage Free Tier: Retornando apenas 100 últimos registros (Compact).")
        if "Note" in data:
            print(f"   [AVISO] Alpha Vantage Limite: {data['Note']}")
            
        time_series = data.get("Time Series (Daily)", {})
        if not time_series:
            print(f"   [WARN] Alpha Vantage: Serie temporal nao encontrada. Resposta: {data}")
            return pd.DataFrame()
            
        # Converter JSON para DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df.index.name = 'Date'
        df = df.sort_index()
        
        # Filtrar por data
        mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
        df_filtered = df.loc[mask]
        
        if df_filtered.empty:
             min_date = df.index.min().date()
             max_date = df.index.max().date()
             print(f"   [WARN] Dados filtrados resultaram vazio (Free Tier limita a 100 ultimos dias).")
             print(f"          Solicitado: {start_date} a {end_date}")
             print(f"          Disponivel: {min_date} a {max_date}")
        
        df = df_filtered
        
        # Renomear colunas para formato yfinance/interno
        df = df.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume"
        })
        
        # Converter colunas para numeric
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col])
            
        return df
        
    except Exception as e:
        print(f"   [ERRO] Falha no Alpha Vantage: {e}")
        return pd.DataFrame()


def ingest_data(ticker: str, start_date: str = None, end_date: str = None) -> int:
    """
    Baixa dados do Yahoo Finance e salva no banco SQLite.
    Tenta Alpha Vantage se Yahoo falhar.
    
    Args:
        ticker: Símbolo da ação (ex: PETR4.SA, DIS, AAPL)
        start_date: Data inicial no formato YYYY-MM-DD (ex: 2018-01-01)
        end_date: Data final no formato YYYY-MM-DD (ex: 2024-07-20)
    
    Returns:
        Quantidade de registros inseridos
    """
    from datetime import timedelta
    
    # Se não passar datas, usar últimos 2 anos como padrão
    if not start_date or not end_date:
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=730)  # 2 anos
        start_date = start_dt.strftime('%Y-%m-%d')
        end_date = end_dt.strftime('%Y-%m-%d')
    
    print(f"[INGEST] Baixando dados de {ticker} ({start_date} a {end_date})...")
    
    # 1. Tentar baixar dados do yfinance (Prioridade 1)
    df = pd.DataFrame()
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    except Exception as e:
        print(f"   [ERRO] Yahoo Finance falhou: {e}")
    
    # 2. Se falhar ou vazio, tentar Alpha Vantage (Prioridade 2)
    if df.empty:
        print(f"   [INFO] Yahoo Finance retornou vazio. Tentando fallback...")
        df = download_alphavantage(ticker, start_date, end_date)
    
    if df.empty:
        raise ValueError(f"Nenhum dado encontrado para {ticker} (Yahoo e Alpha Vantage falharam)")

    
    print(f"   -> {len(df)} registros baixados")
    
    # 2. Limpeza de dados
    df = df.reset_index()
    df = df.rename(columns={
        'Date': 'data',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    
    # Remover timezone e converter para date
    df['data'] = pd.to_datetime(df['data']).dt.date
    
    # Remover valores nulos
    initial_count = len(df)
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    if len(df) < initial_count:
        print(f"   -> {initial_count - len(df)} registros removidos (nulos)")
    
    # Selecionar apenas colunas necessárias
    df = df[['data', 'open', 'high', 'low', 'close', 'volume']]
    df['ticker'] = ticker
    
    # 3. Salvar no banco de dados
    session = SessionLocal()
    try:
        # Remover dados antigos do mesmo ticker
        deleted = session.query(DadosMercado).filter(
            DadosMercado.ticker == ticker
        ).delete()
        if deleted > 0:
            print(f"   -> {deleted} registros antigos removidos")
        
        # Inserir novos dados
        for _, row in df.iterrows():
            dado = DadosMercado(
                ticker=row['ticker'],
                data=row['data'],
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume'])
            )
            session.add(dado)
        
        session.commit()
        print(f"[OK] {len(df)} registros salvos para {ticker}")
        return len(df)
        
    except Exception as e:
        session.rollback()
        print(f"[ERRO] Erro ao salvar dados: {e}")
        raise e
    finally:
        session.close()


def show_stats(ticker: str = None):
    """Mostra estatísticas dos dados armazenados."""
    session = SessionLocal()
    try:
        if ticker:
            count = session.query(DadosMercado).filter(
                DadosMercado.ticker == ticker
            ).count()
            print(f"\n📊 {ticker}: {count} registros")
        else:
            from sqlalchemy import func
            results = session.query(
                DadosMercado.ticker,
                func.count(DadosMercado.id),
                func.min(DadosMercado.data),
                func.max(DadosMercado.data)
            ).group_by(DadosMercado.ticker).all()
            
            print("\n📊 Dados armazenados:")
            print("-" * 50)
            for ticker, count, min_date, max_date in results:
                print(f"   {ticker}: {count} registros ({min_date} a {max_date})")
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(
        description="Ingestao de dados do Yahoo Finance para SQLite"
    )
    parser.add_argument(
        'tickers', 
        nargs='*', 
        default=DEFAULT_TICKERS,
        help='Tickers para baixar (ex: PETR4.SA DIS AAPL)'
    )
    parser.add_argument(
        '--start', '-s',
        default=None,
        help='Data inicial no formato YYYY-MM-DD (ex: 2018-01-01)'
    )
    parser.add_argument(
        '--end', '-e',
        default=None,
        help='Data final no formato YYYY-MM-DD (ex: 2024-07-20)'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Mostrar estatisticas dos dados armazenados'
    )
    
    args = parser.parse_args()
    
    # Inicializar banco de dados
    print("[INIT] Inicializando banco de dados...")
    init_db()
    
    if args.stats:
        show_stats()
        return
    
    # Processar cada ticker
    total_records = 0
    for ticker in args.tickers:
        try:
            records = ingest_data(ticker, start_date=args.start, end_date=args.end)
            total_records += records
        except Exception as e:
            print(f"[ERRO] Erro ao processar {ticker}: {e}")
    
    print(f"\n[DONE] Ingestao concluida! Total: {total_records} registros")
    show_stats()


if __name__ == "__main__":
    main()
