import pytest
from app.agents.student_agent import StudentCompanionAgent


def test_onboarding_mode(db_session, test_wallet):
    """Test onboarding flow."""
    agent = StudentCompanionAgent(db_session)
    
    state = {
        "wallet_address": test_wallet,
        "last_message": "I want to become a blockchain developer",
        "completed_courses": [],
        "current_course_id": None,
        "current_chapter": None,
        "current_chapter_title": None,
        "current_chapter_summary": None,
        "onboarding_data": None,
    }
    
    result = agent.invoke(state)
    
    assert result["mode"] in ["career", "onboarding"]
    assert "response" in result
    assert isinstance(result["response"], str)
    assert len(result["response"]) > 0