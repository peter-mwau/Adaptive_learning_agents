import pytest

from app.database import Base, SessionLocal, engine


@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database session for each test."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def test_wallet():
    """Provide a test wallet address."""
    return "0xTestWallet123456789"


@pytest.fixture
def sample_completed_courses():
    """Provide sample completed courses data."""
    return [
        {"title": "Blockchain Fundamentals", "course_id": 1},
        {"title": "Smart Contract Development", "course_id": 2},
        {"title": "Web3 Architecture", "course_id": 3},
    ]


@pytest.fixture
def sample_onboarding_data():
    """Provide sample onboarding form data."""
    return {
        "currentStatus": "student",
        "currentRole": "Software Developer",
        "yearsOfExperience": 3,
        "industryBackground": "Finance",
        "technicalLevel": "intermediate",
        "programmingLanguages": ["Python", "JavaScript"],
        "hasBlockchainExp": False,
        "hasAIExp": True,
        "targetRole": ["Smart Contract Developer"],
        "careerTimeline": 12,
        "geographicPreference": "Remote",
        "primaryMotivation": ["Career advancement"],
        "webThreeInterest": "high",
        "aiInterest": "medium",
        "learningStyle": "hands-on",
        "timeCommitment": 10,
        "shortTermGoal": "Build first dApp",
        "concerns": "Time management",
        "submittedAt": "2024-01-01T00:00:00Z",
    }
