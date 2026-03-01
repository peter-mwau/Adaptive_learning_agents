from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .agents.course_agent import (
    COURSE_CLUSTERS,
    EVALUATION_ELEMENTS,
    PASS_MARK,
    CourseEvaluationAgent,
)
from .agents.student_agent import StudentCompanionAgent
from .database import get_db
from .schemas import (
    AgentAnalyticsCreate,
    CareerOnboardingRequest,
    CareerOnboardingResponse,
    CourseEvaluationRequest,
    CourseEvaluationResponse,
    CourseRecommendationCreate,
    CourseRecommendationResponse,
    StudentChatRequest,
    StudentChatResponse,
    UserProfileResponse,
    UserProfileUpdate,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Adaptive Learning Agents",
    description="API for Adaptive Learning Agents",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared stateless course agent instance (no DB dependency)
_course_agent = CourseEvaluationAgent()

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

student_router = APIRouter(prefix="/api/student", tags=["Student"])
career_router = APIRouter(prefix="/api", tags=["Career"])
course_router = APIRouter(prefix="/course", tags=["Course Evaluation"])


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Career
# ---------------------------------------------------------------------------


@career_router.post("/career-onboarding", response_model=CareerOnboardingResponse)
async def career_onboarding(
    form_data: CareerOnboardingRequest, db: Session = Depends(get_db)
):
    """
    Handle career onboarding form submission.
    Creates user profile and generates personalised learning recommendations.
    """
    if not form_data.agreeToTerms:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must agree to terms to submit onboarding.",
        )

    agent = StudentCompanionAgent(db)

    initial_state = {
        "wallet_address": form_data.walletAddress,
        "last_message": "Career onboarding form submitted",
        "onboarding_data": form_data.model_dump(),
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

    result = agent.invoke(initial_state)
    onboarding_results = result.get("onboarding_results", {})

    return CareerOnboardingResponse(
        careerProfile=onboarding_results.get("careerProfile"),
        courseMatchAnalysis=onboarding_results.get("courseMatchAnalysis"),
        suggestedCourses=onboarding_results.get("suggestedCourses"),
        additionalNotes=onboarding_results.get("additionalNotes", ""),
    )


# ---------------------------------------------------------------------------
# Student
# ---------------------------------------------------------------------------


@student_router.post("/chat", response_model=StudentChatResponse)
async def student_chat(payload: StudentChatRequest, db: Session = Depends(get_db)):
    """
    Main chat endpoint for student-agent interactions.
    Handles all modes: career, learning, progress, recommendations, general.
    """
    agent = StudentCompanionAgent(db)
    lc = payload.learning_context

    initial_state = {
        "wallet_address": payload.wallet_address,
        "last_message": payload.message,
        "onboarding_data": None,
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

    result = agent.invoke(initial_state)

    return StudentChatResponse(
        response=result["response"],
        mode=result["mode"],
        profile_updated=bool(result.get("profile_updates")),
        recommendations=None,
    )


@student_router.post("/learning-mode", response_model=StudentChatResponse)
async def student_learning_mode(
    payload: StudentChatRequest, db: Session = Depends(get_db)
):
    """
    Dedicated endpoint for learning assistance.
    Used when a student is actively in a course and needs help with content.

    Requires current_course_id or a learning_context with course details.
    Providing current_chapter_title and current_chapter_summary is recommended.
    """
    agent = StudentCompanionAgent(db)
    lc = payload.learning_context

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

    result = agent.invoke(initial_state)

    return StudentChatResponse(
        response=result["response"],
        mode=result["mode"],
        profile_updated=bool(result.get("profile_updates")),
        recommendations=None,
    )


# ---------------------------------------------------------------------------
# Course Evaluation
# ---------------------------------------------------------------------------
#
@course_router.post(
    "/evaluate",
    response_model=CourseEvaluationResponse,
    summary="Evaluate course content (JSON)",
    description=(
        "Submit raw course text and receive a full ABYA rubric evaluation: "
        "cluster category, per-element grades, weighted final score, and pass/fail result."
    ),
)
async def evaluate_course_json(
    request: CourseEvaluationRequest,
) -> CourseEvaluationResponse:
    result = _course_agent.evaluate(request.course_content)
    return CourseEvaluationResponse(**result)


@course_router.post(
    "/evaluate/upload",
    response_model=CourseEvaluationResponse,
    summary="Evaluate course content (file upload)",
    description=(
        "Upload a plain-text (.txt) or Markdown (.md) file containing the course content. "
        "Maximum file size: 5 MB."
    ),
)
async def evaluate_course_upload(
    file: UploadFile = File(..., description="Plain-text or Markdown course file"),
) -> CourseEvaluationResponse:
    allowed_types = {"text/plain", "text/markdown", "application/octet-stream"}
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                "Please upload a plain-text (.txt) or Markdown (.md) file."
            ),
        )

    raw_bytes = await file.read()

    if len(raw_bytes) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds the 5 MB limit.",
        )

    try:
        course_content = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File could not be decoded as UTF-8 text.",
        )

    if len(course_content.strip()) < 50:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Course content is too short (minimum 50 characters).",
        )

    result = _course_agent.evaluate(course_content)
    return CourseEvaluationResponse(**result)


@course_router.get(
    "/clusters",
    response_model=dict,
    summary="List available ABYA rubric clusters",
)
async def list_clusters() -> dict:
    return {
        "clusters": COURSE_CLUSTERS,
        "pass_mark": PASS_MARK,
        "evaluation_elements": EVALUATION_ELEMENTS,
    }


# ---------------------------------------------------------------------------
# Register routers
# ---------------------------------------------------------------------------

app.include_router(career_router)
app.include_router(student_router)
app.include_router(course_router)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
