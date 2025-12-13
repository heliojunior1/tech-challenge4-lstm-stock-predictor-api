"""
train_test.py - Script para testar treinamento e previsao

Uso:
    python train_test.py                    # Treina e preve PETR4.SA
    python train_test.py VALE3.SA           # Treina ticker especifico
    python train_test.py PETR4.SA --epochs 100  # Customiza epochs
"""
import argparse
import sys

# Garantir que o diretorio raiz esta no path
sys.path.insert(0, '.')

from app.database import init_db
from app.services import train_model, predict_price


def main():
    parser = argparse.ArgumentParser(description="Testar treinamento e previsao")
    parser.add_argument('ticker', nargs='?', default='PETR4.SA', help='Ticker para treinar')
    parser.add_argument('--epochs', '-e', type=int, default=50, help='Numero de epochs')
    parser.add_argument('--days', '-d', type=int, default=1, help='Dias para prever')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("[TEST] TESTE DE TREINAMENTO E PREVISAO")
    print("=" * 50)
    
    # Inicializar banco
    print("\n[INFO] Inicializando banco de dados...")
    init_db()
    
    # 1. Treinar modelo
    print(f"\n[TRAIN] Treinando modelo para {args.ticker}...")
    result = train_model(args.ticker, epochs=args.epochs)
    
    print("\n" + "=" * 50)
    print("[RESULT] RESULTADO DO TREINAMENTO")
    print("=" * 50)
    print(f"   Ticker: {result['ticker']}")
    print(f"   Epochs: {result['epochs']}")
    print(f"   Train Loss: {result['final_train_loss']:.6f}")
    print(f"   Val Loss: {result['final_val_loss']:.6f}")
    print(f"   RMSE: R$ {result['rmse']:.2f}")
    print(f"   MAE: R$ {result['mae']:.2f}")
    print(f"   Modelo: {result['model_path']}")
    
    # 2. Fazer previsao
    print(f"\n[PREDICT] Fazendo previsao para {args.days} dia(s)...")
    pred = predict_price(args.ticker, days=args.days)
    
    print("\n" + "=" * 50)
    print("[RESULT] PREVISAO")
    print("=" * 50)
    for p in pred['predictions']:
        print(f"   Dia {p['day']}: R$ {p['predicted_price']:.2f}")
    
    print("\n[OK] Teste concluido com sucesso!")
    print("=" * 50)


if __name__ == "__main__":
    main()
