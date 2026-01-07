"""
Resource Service - Serviço para coleta de métricas de recursos do sistema

Coleta métricas de CPU e memória usando psutil.
Atualiza tanto métricas Prometheus quanto banco SQLite.
"""
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from app.database import SessionLocal, SystemMetric


class ResourceService:
    """Serviço para monitoramento de recursos do sistema."""
    
    @staticmethod
    def get_current() -> Dict:
        """
        Retorna uso atual de CPU e memória.
        
        Returns:
            Dict com cpu_percent, memory_mb, memory_percent
        """
        # Uso de CPU do sistema
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memória do processo atual
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = round(memory_info.rss / (1024 * 1024), 2)
        memory_percent = round(process.memory_percent(), 2)
        
        return {
            "cpu_percent": cpu_percent,
            "memory_mb": memory_mb,
            "memory_percent": memory_percent,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @staticmethod
    def save_snapshot() -> None:
        """Salva snapshot das métricas no banco SQLite."""
        metrics = ResourceService.get_current()
        
        session = SessionLocal()
        try:
            system_metric = SystemMetric(
                cpu_percent=metrics["cpu_percent"],
                memory_mb=metrics["memory_mb"],
                memory_percent=metrics["memory_percent"]
            )
            session.add(system_metric)
            session.commit()
        except Exception as e:
            print(f"[WARN] Erro ao salvar métrica de sistema: {e}")
        finally:
            session.close()
    
    @staticmethod
    def get_history(hours: int = 1) -> List[Dict]:
        """
        Retorna histórico de métricas de recursos.
        
        Args:
            hours: Número de horas para buscar (default: 1)
        
        Returns:
            Lista de métricas ordenadas por timestamp
        """
        session = SessionLocal()
        try:
            since = datetime.utcnow() - timedelta(hours=hours)
            
            metrics = session.query(SystemMetric).filter(
                SystemMetric.timestamp >= since
            ).order_by(SystemMetric.timestamp.asc()).all()
            
            return [{
                "cpu_percent": m.cpu_percent,
                "memory_mb": m.memory_mb,
                "memory_percent": m.memory_percent,
                "timestamp": m.timestamp.isoformat()
            } for m in metrics]
        finally:
            session.close()
    
    @staticmethod
    def cleanup_old_metrics(days: int = 7) -> int:
        """
        Remove métricas antigas do banco.
        
        Args:
            days: Métricas mais antigas que X dias serão removidas
        
        Returns:
            Número de registros removidos
        """
        session = SessionLocal()
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            
            deleted = session.query(SystemMetric).filter(
                SystemMetric.timestamp < cutoff
            ).delete()
            
            session.commit()
            return deleted
        finally:
            session.close()
