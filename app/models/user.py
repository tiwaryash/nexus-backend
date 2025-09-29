from sqlalchemy import Boolean, Column, Integer, String
from sqlalchemy.orm import relationship
from app.db.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    name = Column(String)
    password = Column(String)
    is_active = Column(Boolean, default=True)
    
    documents = relationship("Document", back_populates="user")
    shared_documents = relationship("ShareAccess", back_populates="shared_by")
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
