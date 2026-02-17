from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .agents.student_agent import StudentCompanionAgent
from .database import get_db
from .schemas import (
    AgentAnalyticsCreate,
    CareerOnboardingRequest,
    CareerOnboardingResponse,
    CourseRecommendationCreate,
    CourseRecommendationResponse,
    StudentChatRequest,
    StudentChatResponse,
    UserProfileResponse,
    UserProfileUpdate,
)

app = FastAPI(
    title="Adaptive Learning Agents", description="API for Adaptive Learning Agents"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/career-onboarding", response_model=CareerOnboardingResponse)
async def career_onboarding(
    form_data: CareerOnboardingRequest, db: Session = Depends(get_db)
):
    """
    Handle career onboarding form submission.
    Creates user profile and generates personalized learning recommendations.
    """
    print("Career onboarding form received", form_data)
    if not form_data.agreeToTerms:
        raise HTTPException(
            status_code=400, detail="Must agree to terms to submit onboarding"
        )

    # Initialize agent
    agent = StudentCompanionAgent(db)

    # Convert form data to onboarding dict
    onboarding_dict = form_data.model_dump()

    # Prepare initial state for agent
    initial_state = {
        "wallet_address": form_data.walletAddress,
        "last_message": "Career onboarding form submitted",
        "onboarding_data": onboarding_dict,
        "user_profile": {},
        "conversation_history": [],
        "conversation_summary": "",
        "completed_courses": None,
        "current_course_id": None,
        "current_chapter": None,
        "current_chapter_title": None,
        "current_chapter_summary": None,
        "mode": "onboarding",
        "response": "",
        "profile_updates": {},
    }

    # Execute agent workflow
    result = agent.invoke(initial_state)
    onboarding_results = result.get("onboarding_results", {})
    print("Onboarding Results:", onboarding_results)

    return CareerOnboardingResponse(
        careerProfile=onboarding_results.get("careerProfile"),
        courseMatchAnalysis=onboarding_results.get("courseMatchAnalysis"),
        suggestedCourses=onboarding_results.get("suggestedCourses"),
        additionalNotes=onboarding_results.get("additionalNotes", ""),
    )


@app.post("/student/chat", response_model=StudentChatResponse)
async def student_chat(payload: StudentChatRequest, db: Session = Depends(get_db)):
    """
    Main chat endpoint for student-agent interactions.
    Handles all modes: career, learning, progress, recommendations, general.
    """
    agent = StudentCompanionAgent(db)

    # Extract learning context if provided
    lc = payload.learning_context

    # Prepare initial state
    initial_state = {
        "wallet_address": payload.wallet_address,
        "last_message": payload.message,
        "onboarding_data": None,  # Only set for onboarding form submissions
        "user_profile": {},
        "conversation_history": [],
        "conversation_summary": "",
        "completed_courses": lc.completed_courses if lc else None,
        "current_course_id": lc.current_course_id if lc else payload.current_course_id,
        "current_chapter": lc.current_chapter if lc else None,
        "current_chapter_title": lc.current_chapter_title if lc else None,
        "current_chapter_summary": lc.current_chapter_summary if lc else None,
        "mode": "general",
        "response": "",
        "profile_updates": {},
    }

    # Execute agent
    result = agent.invoke(initial_state)

    return StudentChatResponse(
        response=result["response"],
        mode=result["mode"],
        profile_updated=bool(result.get("profile_updates")),
        recommendations=None,  # Can be populated if needed
    )


@app.post("/api/student/learning-mode", response_model=StudentChatResponse)
async def student_learning_mode(
    payload: StudentChatRequest, db: Session = Depends(get_db)
):
    """
    Dedicated endpoint for learning assistance.
    Used when student is actively in a course and needs help with content.

    Requires:
    - current_course_id or learning_context with course details
    - current_chapter_title and current_chapter_summary (recommended)
    """
    print("Payload:", payload)
    # if payload.current_course_id is not None:
    #     raise HTTPException(
    #         status_code=400,
    #         detail="current_course_id is required for learning assistance",
    #     )

    agent = StudentCompanionAgent(db)
    lc = payload.learning_context

    # Prepare initial state with learning context
    initial_state = {
        "wallet_address": payload.wallet_address,
        "last_message": payload.message,
        "onboarding_data": None,
        "user_profile": {},
        "conversation_history": [],
        "conversation_summary": "",
        "completed_courses": lc.completed_courses if lc else [],
        "current_course_id": lc.current_course_id if lc else payload.current_course_id,
        "current_chapter": lc.current_chapter if lc else None,
        "current_chapter_title": lc.current_chapter_title if lc else None,
        "current_chapter_summary": lc.current_chapter_summary if lc else None,
        "mode": "learning",  # Force learning mode
        "response": "",
        "profile_updates": {},
    }

    # Execute agent
    result = agent.invoke(initial_state)

    return StudentChatResponse(
        response=result["response"],
        mode=result["mode"],
        profile_updated=bool(result.get("profile_updates")),
        recommendations=None,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
