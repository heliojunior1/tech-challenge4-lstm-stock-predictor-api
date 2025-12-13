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

# Adicionar diretório raiz ao path para imports
sys.path.insert(0, str(__file__).rsplit('\\', 1)[0] if '\\' in str(__file__) else '.')

from app.database import SessionLocal, DadosMercado, init_db
from app.config import DEFAULT_TICKERS


def ingest_data(ticker: str, period: str = "2y") -> int:
    """
    Baixa dados do Yahoo Finance e salva no banco SQLite.
    
    Args:
        ticker: Símbolo da ação (ex: PETR4.SA, DIS, AAPL)
        period: Período de histórico (ex: 1y, 2y, 5y, max)
    
    Returns:
        Quantidade de registros inseridos
    """
    print(f"📥 Baixando dados de {ticker} (período: {period})...")
    
    # 1. Baixar dados do yfinance
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    
    if df.empty:
        raise ValueError(f"Nenhum dado encontrado para {ticker}")
    
    print(f"   ↳ {len(df)} registros baixados")
    
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
        print(f"   ↳ {initial_count - len(df)} registros removidos (nulos)")
    
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
            print(f"   ↳ {deleted} registros antigos removidos")
        
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
        print(f"✅ {len(df)} registros salvos para {ticker}")
        return len(df)
        
    except Exception as e:
        session.rollback()
        print(f"❌ Erro ao salvar dados: {e}")
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
        description="Ingestão de dados do Yahoo Finance para SQLite"
    )
    parser.add_argument(
        'tickers', 
        nargs='*', 
        default=DEFAULT_TICKERS,
        help='Tickers para baixar (ex: PETR4.SA DIS AAPL)'
    )
    parser.add_argument(
        '--period', '-p',
        default='2y',
        help='Período de histórico: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max'
    )
    parser.add_argument(
        '--stats', '-s',
        action='store_true',
        help='Mostrar estatísticas dos dados armazenados'
    )
    
    args = parser.parse_args()
    
    # Inicializar banco de dados
    print("🔧 Inicializando banco de dados...")
    init_db()
    
    if args.stats:
        show_stats()
        return
    
    # Processar cada ticker
    total_records = 0
    for ticker in args.tickers:
        try:
            records = ingest_data(ticker, args.period)
            total_records += records
        except Exception as e:
            print(f"❌ Erro ao processar {ticker}: {e}")
    
    print(f"\n🎉 Ingestão concluída! Total: {total_records} registros")
    show_stats()


if __name__ == "__main__":
    main()
