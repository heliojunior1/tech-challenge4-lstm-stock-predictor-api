"""
Routers package - Contém endpoints da API organizados por funcionalidade
"""
from app.routers import training, inference, monitoring

__all__ = ["training", "inference", "monitoring"]
