from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime

# ==================== STUDENT AGENT ====================


class LearningContext(BaseModel):
    current_course_id: Optional[int] = None
    current_chapter: Optional[int] = None
    current_chapter_title: Optional[str] = None
    current_chapter_summary: Optional[str] = None
    # Use default_factory to avoid shared mutable default
    completed_courses: List[Dict] = Field(default_factory=list)


class StudentChatRequest(BaseModel):
    wallet_address: str = Field(
        ...,
        min_length=42,
        max_length=42,
        description="Ethereum wallet address",
    )
    message: str = Field(..., min_length=1, max_length=2000)
    # Optional top-level course id for convenience/backwards-compat
    current_course_id: Optional[int] = None
    # Frontend-supplied learning context snapshot
    learning_context: Optional[LearningContext] = None

    class Config:
        json_schema_extra = {
            "example": {
                "wallet_address": "0x1234567890123456789012345678901234567890",
                "message": "Help me understand smart contract security",
                "current_course_id": 5,
                "learning_context": {
                    "current_course_id": 5,
                    "current_chapter": 2,
                    "current_chapter_title": "Smart contract security basics",
                    "current_chapter_summary": "Overview of key vulnerabilities and best practices.",
                    "completed_courses": [
                        {"course_id": 1, "title": "Intro to Web3"}
                    ],
                },
            }
        }


class StudentChatResponse(BaseModel):
    response: str
    mode: str  # 'career', 'learning', 'progress', 'recommendation', 'general'
    profile_updated: bool
    recommendations: Optional[List[Dict]] = None  # If mode='recommendation'

    class Config:
        json_schema_extra = {
            "example": {
                "response": "Smart contract security is crucial! Let me explain the key concepts...",
                "mode": "learning",
                "profile_updated": True,
                "recommendations": None,
            }
        }

# ==================== USER PROFILE ====================

class UserProfileResponse(BaseModel):
    wallet_address: str
    career_context: Dict
    skill_profile: Dict
    learning_preferences: Dict
    learning_challenges: List[str]
    total_conversations: int
    last_active: datetime

class UserProfileUpdate(BaseModel):
    career_context: Optional[Dict] = None
    skill_profile: Optional[Dict] = None
    learning_preferences: Optional[Dict] = None
    learning_challenges: Optional[List[str]] = None
    
# ==================== COURSE RECOMMENDATIONS ====================

class CourseRecommendationCreate(BaseModel):
    wallet_address: str
    course_id: int
    reason: str
    priority: int = Field(default=3, ge=1, le=5)

class CourseRecommendationResponse(BaseModel):
    id: int
    course_id: int
    reason: str
    priority: int
    is_viewed: bool
    is_enrolled: bool
    created_at: datetime

# ==================== ANALYTICS ====================

class AgentAnalyticsCreate(BaseModel):
    agent_type: str
    event_type: str
    execution_time_ms: int
    tokens_used: int
    success: bool
    error_message: Optional[str] = None
    wallet_address: Optional[str] = None
    course_id: Optional[int] = None

class AgentAnalyticsResponse(BaseModel):
    total_requests: int
    avg_execution_time_ms: float
    avg_tokens_per_request: float
    success_rate: float
    most_common_errors: List[Dict]