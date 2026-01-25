from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Optional, List, Dict
import json

from app.config import settings
from app.models import (
    Base, 
    UserProfile, 
    Conversation, 
    CourseRecommendation,
    AgentAnalytics
)

#create engine
if "sqlite" in settings.DATABASE_URL:
    # SQLite for development
    engine = create_engine(
        settings.DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
else:
    # PostgreSQL for production
    engine = create_engine(
        settings.DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,  # Verify connections before using
        echo=settings.ENVIRONMENT == "development"
    )

# Create all tables
Base.metadata.create_all(bind=engine)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ==================== DEPENDENCY ====================

def get_db():
    """FastAPI dependency for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_context():
    """Context manager for non-FastAPI usage."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ==================== USER PROFILE OPERATIONS ====================

def get_user_profile(db: Session, wallet_address: str) -> Optional[Dict]:
    """Get user profile as dict."""
    profile = db.query(UserProfile).filter_by(wallet_address=wallet_address).first()
    
    if not profile:
        return None
    
    return {
        'wallet_address': profile.wallet_address,
        'email': profile.email,
        'display_name': profile.display_name,
        'career_context': profile.career_context or {},
        'skill_profile': profile.skill_profile or {},
        'learning_preferences': profile.learning_preferences or {},
        'learning_challenges': profile.learning_challenges or [],
        'total_conversations': profile.total_conversations,
        'last_active': profile.last_active
    }

def create_user_profile(db: Session, wallet_address: str, **kwargs) -> UserProfile:
    """Create new user profile."""
    profile = UserProfile(
        wallet_address=wallet_address,
        **kwargs
    )
    db.add(profile)
    db.commit()
    db.refresh(profile)
    return profile

def update_user_profile(db: Session, wallet_address: str, updates: Dict) -> UserProfile:
    """Update user profile with new data."""
    profile = db.query(UserProfile).filter_by(wallet_address=wallet_address).first()
    
    if not profile:
        # Create if doesn't exist
        profile = create_user_profile(db, wallet_address)
    
    # Update career context
    if 'career_context' in updates:
        current_career = profile.career_context or {}
        current_career.update(updates['career_context'])
        profile.career_context = current_career
    
    # Update skill profile
    if 'skill_profile' in updates:
        current_skills = profile.skill_profile or {}
        current_skills.update(updates['skill_profile'])
        profile.skill_profile = current_skills
    
    # Update learning preferences
    if 'learning_preferences' in updates:
        current_prefs = profile.learning_preferences or {}
        current_prefs.update(updates['learning_preferences'])
        profile.learning_preferences = current_prefs
    
    # Update learning challenges
    if 'learning_challenges' in updates:
        current_challenges = profile.learning_challenges or []
        new_challenges = updates['learning_challenges']
        # Merge without duplicates
        profile.learning_challenges = list(set(current_challenges + new_challenges))
    
    # Update other fields
    for key in ['email', 'display_name']:
        if key in updates:
            setattr(profile, key, updates[key])
    
    profile.last_active = func.now()
    
    db.commit()
    db.refresh(profile)
    return profile

# ==================== CONVERSATION OPERATIONS ====================

def save_conversation(
    db: Session,
    wallet_address: str,
    role: str,
    content: str,
    agent_type: str = 'student',
    mode: Optional[str] = None,
    course_id: Optional[int] = None,
    chapter_id: Optional[int] = None,
    tokens_used: int = 0
) -> Conversation:
    """Save a conversation message."""
    
    conversation = Conversation(
        wallet_address=wallet_address,
        agent_type=agent_type,
        mode=mode,
        role=role,
        content=content,
        course_id=course_id,
        chapter_id=chapter_id,
        tokens_used=tokens_used
    )
    
    db.add(conversation)
    
    # Update user's total conversation count
    profile = db.query(UserProfile).filter_by(wallet_address=wallet_address).first()
    if profile:
        profile.total_conversations += 1
        profile.last_active = func.now()
    
    db.commit()
    db.refresh(conversation)
    return conversation

def get_conversation_history(
    db: Session,
    wallet_address: str,
    limit: int = 10,
    agent_type: Optional[str] = None,
    course_id: Optional[int] = None
) -> List[Dict]:
    """Get recent conversation history."""
    
    query = db.query(Conversation).filter_by(wallet_address=wallet_address)
    
    if agent_type:
        query = query.filter_by(agent_type=agent_type)
    
    if course_id:
        query = query.filter_by(course_id=course_id)
    
    messages = query.order_by(Conversation.created_at.desc()).limit(limit).all()
    
    return [
        {
            'role': msg.role,
            'content': msg.content,
            'created_at': msg.created_at
        }
        for msg in reversed(messages)
    ]

# ==================== COURSE RECOMMENDATION OPERATIONS ====================

def create_recommendation(
    db: Session,
    wallet_address: str,
    course_id: int,
    reason: str,
    priority: int = 3
) -> CourseRecommendation:
    """Create a course recommendation."""
    
    # Check if already exists
    existing = db.query(CourseRecommendation).filter_by(
        wallet_address=wallet_address,
        course_id=course_id
    ).first()
    
    if existing:
        # Update existing
        existing.reason = reason
        existing.priority = priority
        db.commit()
        return existing
    
    recommendation = CourseRecommendation(
        wallet_address=wallet_address,
        course_id=course_id,
        reason=reason,
        priority=priority
    )
    
    db.add(recommendation)
    db.commit()
    db.refresh(recommendation)
    return recommendation

def get_user_recommendations(
    db: Session,
    wallet_address: str,
    unviewed_only: bool = False
) -> List[CourseRecommendation]:
    """Get user's course recommendations."""
    
    query = db.query(CourseRecommendation).filter_by(wallet_address=wallet_address)
    
    if unviewed_only:
        query = query.filter_by(is_viewed=False)
    
    return query.order_by(CourseRecommendation.priority).all()

def log_agent_event(
    db: Session,
    agent_type: str,
    event_type: str,
    execution_time_ms: int,
    tokens_used: int,
    success: bool = True,
    **kwargs
) -> AgentAnalytics:
    """Log an agent event for analytics."""
    
    analytics = AgentAnalytics(
        agent_type=agent_type,
        event_type=event_type,
        execution_time_ms=execution_time_ms,
        tokens_used=tokens_used,
        success=success,
        **kwargs
    )
    
    db.add(analytics)
    db.commit()
    return analytics

def get_agent_stats(db: Session, agent_type: Optional[str] = None, days: int = 7) -> Dict:
    """Get agent performance statistics."""
    from datetime import datetime, timedelta
    
    since = datetime.utcnow() - timedelta(days=days)
    
    query = db.query(AgentAnalytics).filter(AgentAnalytics.created_at >= since)
    
    if agent_type:
        query = query.filter_by(agent_type=agent_type)
    
    events = query.all()
    
    if not events:
        return {
            'total_requests': 0,
            'avg_execution_time_ms': 0,
            'avg_tokens_per_request': 0,
            'success_rate': 0
        }
    
    total_requests = len(events)
    successful = sum(1 for e in events if e.success)
    total_time = sum(e.execution_time_ms for e in events)
    total_tokens = sum(e.tokens_used for e in events)
    
    return {
        'total_requests': total_requests,
        'avg_execution_time_ms': total_time / total_requests,
        'avg_tokens_per_request': total_tokens / total_requests,
        'success_rate': successful / total_requests
    }

# ==================== UTILITY FUNCTIONS ====================

def delete_user_data(db: Session, wallet_address: str) -> bool:
    """Delete all user data (GDPR compliance)."""
    profile = db.query(UserProfile).filter_by(wallet_address=wallet_address).first()
    
    if profile:
        db.delete(profile)  # Cascade will delete conversations and recommendations
        db.commit()
        return True
        
    return False

def get_active_users_count(db: Session, days: int = 30) -> int:
    """Count active users in the last N days."""
    from datetime import datetime, timedelta
    
    since = datetime.utcnow() - timedelta(days=days)
    
    return db.query(UserProfile).filter(
        UserProfile.last_active >= since
    ).count()