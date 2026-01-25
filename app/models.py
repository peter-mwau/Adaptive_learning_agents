from datetime import datetime
from sys import displayhook

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class UserProfile(Base):
    __tablename__ = "user_profiles"

    walletaddress = Column(String, primary_key=True)
    created_at = Column(DateTime, default=datetime.now(datetime.UTC))
    updated_at = Column(
        DateTime,
        default=datetime.now(datetime.UTC),
        onupdate=datetime.now(datetime.UTC),
    )

    email = Column(String(255), nullable=False, unique=True)
    display_name = Column(String(100), nullable=False)

    career_context = Column(JSON, default=dict, nullable=True)
    skill_profile = Column(JSON, default=dict, nullable=True)
    learning_preferences = Column(JSON, default=dict, nullable=True)
    learning_challenges = Column(JSON, default=list, nullable=True)

    conversations = relationship(
        "Conversation", back_populates="user", cascade="all, delete-orphan"
    )
    recommendations = relationship(
        "CourseRecommendation", back_populates="user", cascade="all, delete-orphan"
    )
