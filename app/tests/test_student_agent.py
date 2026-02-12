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
    
def test_learning_mode(db_session, test_wallet):
    """Test learning assistance mode."""
    agent = StudentCompanionAgent(db_session)
    
    # First create a user profile (simulate existing user)
    from app.database import create_user_profile
    create_user_profile(db_session, test_wallet)
    
    state = {
        "wallet_address": test_wallet,
        "last_message": "Can you explain what a smart contract is?",
        "completed_courses": [],
        "current_course_id": 1,  # User is in a course
        "current_chapter": 2,
        "current_chapter_title": "Introduction to Smart Contracts",
        "current_chapter_summary": "This chapter covers the basics of smart contracts, including their definition, use cases, and how they work on blockchain platforms.",
        "onboarding_data": None,
    }
    
    result = agent.invoke(state)
    
    # Assertions
    assert result["mode"] == "learning"
    assert "response" in result
    assert isinstance(result["response"], str)
    assert len(result["response"]) > 0
    # Check that response mentions relevant concepts
    assert any(keyword in result["response"].lower() for keyword in ["smart contract", "blockchain", "code"])
