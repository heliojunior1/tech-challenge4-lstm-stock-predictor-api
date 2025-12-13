"""
Database - Conexão SQLite com SQLAlchemy
Configuração do banco de dados e modelos ORM
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from app.config import DATABASE_URL

# Engine SQLite
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class DadosMercado(Base):
    """
    Modelo para armazenar dados históricos do mercado.
    Campos: ticker, data, open, high, low, close, volume
    """
    __tablename__ = "dados_mercado"
    
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True)
    data = Column(Date, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    
    def __repr__(self):
        return f"<DadosMercado(ticker={self.ticker}, data={self.data}, close={self.close})>"


class Prediction(Base):
    """
    Modelo para armazenar histórico de previsões realizadas.
    """
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True)
    predicted_price = Column(Float)
    actual_price = Column(Float, nullable=True)
    prediction_date = Column(DateTime, default=datetime.utcnow)
    model_version = Column(String)
    
    def __repr__(self):
        return f"<Prediction(ticker={self.ticker}, predicted={self.predicted_price})>"


class TrainedModel(Base):
    """
    Modelo para tracking de modelos treinados.
    """
    __tablename__ = "trained_models"
    
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True)
    model_path = Column(String)
    train_loss = Column(Float)
    val_loss = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)
    epochs = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<TrainedModel(ticker={self.ticker}, rmse={self.rmse})>"


def init_db():
    """Inicializa o banco de dados criando todas as tabelas."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency para injetar sessão do banco nas rotas FastAPI."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
