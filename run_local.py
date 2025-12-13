"""
run_local.py - Script para executar a API e Prometheus localmente

Uso:
    python run_local.py           # Inicia apenas a API
    python run_local.py --all     # Inicia API + Prometheus (requer Docker)
"""
import argparse
import subprocess
import sys
import time
import webbrowser


def run_api(host="127.0.0.1", port=8000, reload=True):
    """Inicia a API com uvicorn."""
    print("=" * 50)
    print("[API] Iniciando Stock Predictor API")
    print("=" * 50)
    print(f"   URL: http://{host}:{port}")
    print(f"   Docs: http://{host}:{port}/docs")
    print(f"   Metrics: http://{host}:{port}/metrics")
    print("=" * 50)
    print("\n[INFO] Pressione Ctrl+C para parar\n")
    
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "app.main:app",
        "--host", host,
        "--port", str(port)
    ]
    
    if reload:
        cmd.append("--reload")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n[INFO] API encerrada.")


def run_prometheus():
    """Inicia Prometheus via Docker."""
    print("\n[PROMETHEUS] Iniciando Prometheus via Docker...")
    
    cmd = [
        "docker", "run", "-d",
        "--name", "prometheus-local",
        "-p", "9090:9090",
        "-v", f"{__file__}/../prometheus.yml:/etc/prometheus/prometheus.yml",
        "prom/prometheus:latest"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("[PROMETHEUS] Prometheus iniciado em http://localhost:9090")
    except subprocess.CalledProcessError:
        print("[PROMETHEUS] Erro ao iniciar. Verifique se Docker esta instalado.")
    except FileNotFoundError:
        print("[PROMETHEUS] Docker nao encontrado. Instale Docker para usar Prometheus.")


def run_docker_compose():
    """Inicia todos os servicos via docker-compose."""
    print("\n[DOCKER] Iniciando todos os servicos...")
    
    try:
        subprocess.run(["docker-compose", "up", "-d"], check=True)
        print("\n[OK] Servicos iniciados:")
        print("   - API: http://localhost:8000")
        print("   - Prometheus: http://localhost:9090")
        print("   - Grafana: http://localhost:3000 (admin/admin)")
    except subprocess.CalledProcessError as e:
        print(f"[ERRO] Falha ao iniciar: {e}")
    except FileNotFoundError:
        print("[ERRO] docker-compose nao encontrado.")


def main():
    parser = argparse.ArgumentParser(description="Executar Stock Predictor API localmente")
    parser.add_argument('--all', action='store_true', help='Iniciar API + Prometheus + Grafana (Docker)')
    parser.add_argument('--docker', action='store_true', help='Usar docker-compose')
    parser.add_argument('--host', default='127.0.0.1', help='Host da API')
    parser.add_argument('--port', type=int, default=8000, help='Porta da API')
    parser.add_argument('--no-reload', action='store_true', help='Desabilitar auto-reload')
    parser.add_argument('--open', action='store_true', help='Abrir docs no navegador')
    
    args = parser.parse_args()
    
    if args.docker or args.all:
        run_docker_compose()
    else:
        if args.open:
            # Abrir docs apos 2 segundos
            import threading
            def open_browser():
                time.sleep(2)
                webbrowser.open(f"http://{args.host}:{args.port}/docs")
            threading.Thread(target=open_browser, daemon=True).start()
        
        run_api(args.host, args.port, not args.no_reload)


if __name__ == "__main__":
    main()
