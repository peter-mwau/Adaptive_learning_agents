from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_anthropic import ChatAnthropic
import operator
import json

from app.database import (
    get_user_profile,
    update_user_profile,
    save_conversation,
    get_conversation_history,
)
from app.config import settings


class StudentState(TypedDict):
    wallet_address: str
    # Loaded context from DB
    user_profile: dict
    conversation_history: Annotated[list, operator.add]

    # Frontend-supplied context (no Web3 calls needed)
    completed_courses: list
    current_course_id: int | None
    current_chapter: int | None
    current_chapter_title: str | None
    current_chapter_summary: str | None

    # Current interaction
    last_message: str
    mode: str  # 'career', 'learning', 'progress', 'recommendation', 'general'

    # Response data
    response: str
    profile_updates: dict


class StudentCompanionAgent:
    """Unified agent for all learner interactions."""

    def __init__(self, db_session):
        self.db = db_session
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-flash", api_key=settings.GEMINI_API_KEY
        )
        self.graph = self.build_graph()

    def build_graph(self):
        """Build and compile the student companion workflow graph."""
        workflow = StateGraph(StudentState)

        # Add nodes
        workflow.add_node("load_context", self.load_context)
        workflow.add_node("determine_mode", self.determine_mode)
        workflow.add_node("career_mode", self.career_guidance)
        workflow.add_node("learning_mode", self.learning_assistance)
        workflow.add_node("progress_mode", self.progress_review)
        workflow.add_node("recommend_mode", self.course_recommendation)
        workflow.add_node("general_mode", self.general_conversation)
        workflow.add_node("update_profile", self.update_user_profile_node)

        # Entry point
        workflow.set_entry("load_context")

        # Sequential flow
        workflow.add_edge("load_context", "determine_mode")

        # Conditional routing based on detected mode
        workflow.add_conditional_edges(
            "determine_mode",
            lambda s: s["mode"],
            {
                "career": "career_mode",
                "learning": "learning_mode",
                "progress": "progress_mode",
                "recommendation": "recommend_mode",
                "general": "general_mode",
            },
        )

        # All modes converge to profile update
        workflow.add_edge("career_mode", "update_profile")
        workflow.add_edge("learning_mode", "update_profile")
        workflow.add_edge("progress_mode", "update_profile")
        workflow.add_edge("recommend_mode", "update_profile")
        workflow.add_edge("general_mode", "update_profile")

        workflow.add_edge("update_profile", END)

        return workflow.compile()

    def load_context(self, state: StudentState) -> StudentState:
        """
        Load user context from the agent database (profile and recent history).

        All course/progress information is expected to be already present
        in the incoming state from the frontend.
        """
        wallet = state["wallet_address"]

        profile = get_user_profile(self.db, wallet) or {}
        history = get_conversation_history(self.db, wallet, limit=10)

        return {
            **state,
            "user_profile": profile,
            "conversation_history": history,
            "profile_updates": {},
        }
            
    def determine_mode(self, state: StudentState) -> StudentState:
        """Decide what mode to operate in."""

        message = state["last_message"].lower()

        # Career onboarding for new users
        if not state["completed_courses"] and not state["user_profile"].get(
            "career_context"
        ):
            mode = "career"

        # Learning help if user is in a course
        elif state["current_course_id"]:
            # Check if they're asking about progress or recommendations
            if any(kw in message for kw in ["progress", "how am i doing", "stats"]):
                mode = "progress"
            elif any(
                kw in message for kw in ["what next", "recommend", "what should i"]
            ):
                mode = "recommendation"
            elif any(kw in message for kw in ["what should i learn", "what course"]):
                mode = "recommendation"

            else:
                mode = "learning"

        # Progress review
        elif any(kw in message for kw in ["progress", "completed", "achievement"]):
            mode = "progress"

        # Career guidance
        elif any(kw in message for kw in ["career", "job", "become a", "work as"]):
            mode = "career"

        # Default
        else:
            mode = "general"

        return {**state, "mode": mode}
    def career_guidance(self, state: StudentState) -> StudentState:
        """Provide career guidance to the student."""
        profile = state["user_profile"]
        completed = state["completed_courses"]

        system_prompt = f"""You are a career guidance AI for a Web3 learning platform.
User has completed: {[c.get('title', 'Unknown course') for c in completed] if completed else 'No courses yet'}
Current career context: {profile.get('career_context', 'Unknown')}

Your goal: Help user discover their career path in Web3/AI/tech.

If new user:
- Ask about current situation (student, employed, career change)
- Discover target role (smart contract dev, blockchain analyst, etc.)
- Understand motivation and timeline

If returning user:
- Review progress toward career goal
- Provide job market insights
- Recommend next steps

Be conversational and encouraging."""
        messages = [
            {"role": "system", "content": system_prompt},
            *state["conversation_history"],
            {"role": "user", "content": state["last_message"]},
        ]

        response = self.llm.invoke(messages)

        # Extract career insights
        extraction_prompt = f"""From this conversation, extract career context as JSON:

Conversation: {state['last_message']}
Response: {response.content}

Return ONLY valid JSON (or empty object if no new info):
{{
    "current_role": "student|employed|unemployed|...",
    "target_role": "smart contract developer|...",
    "industries": ["defi", "nfts", ...],
    "timeline": "3-6 months|...",
    "motivation": "career change|upskill|..."
}}"""

        extraction = self.llm.invoke(extraction_prompt)

        try:
            career_context = json.loads(
                extraction.content.strip("```json").strip("```")
            )
        except Exception:
            career_context = {}

        return {
            **state,
            "response": response.content,
            "profile_updates": (
                {"career_context": career_context} if career_context else {}
            ),
        }

    def learning_assistance(self, state: StudentState) -> StudentState:
        """Help user while they're in a course, using frontend-supplied chapter context."""

        chapter_title = state.get("current_chapter_title")
        chapter_summary = state.get("current_chapter_summary")

        system_prompt = f"""You are a learning assistant helping a student in a Web3 course.

Current Chapter: {chapter_title or "Unknown"}
Chapter content summary:
{chapter_summary or "No summary available"}

User's skill level: {state['user_profile'].get('skill_level', 'Unknown')}

Your role:
- Answer questions about the current chapter
- Explain concepts clearly
- Provide examples and analogies
- Give hints for exercises (don't give full solutions)
- Encourage and motivate

Be patient and adaptive to their level."""

        messages = [
            {"role": "system", "content": system_prompt},
            *state["conversation_history"],
            {"role": "user", "content": state["last_message"]},
        ]

        response = self.llm.invoke(messages)

        # Track learning challenges based on difficulty signals
        if any(
            kw in state["last_message"].lower()
            for kw in ["don't understand", "confused", "stuck", "difficult"]
        ):
            challenges = state["user_profile"].get("learning_challenges", [])
            topic = chapter_title or "Current chapter"
            if topic not in challenges:
                challenges.append(topic)

            profile_updates = {"learning_challenges": challenges}
        else:
            profile_updates = {}

        return {**state, "response": response.content, "profile_updates": profile_updates}
    
    def progress_review(self, state: StudentState) -> StudentState:
        """Show user their learning progress."""

        completed = state["completed_courses"]
        current = state["current_course_id"]

        completed_lines = "\n".join(
            [f"✓ {c.get('title', 'Unknown course')}" for c in completed]
        )
        current_course_label = (
            f"Course ID {current}" if current is not None else "No active course"
        )

        progress_summary = f"""You've completed {len(completed)} courses:
{completed_lines}

Currently learning: {current_course_label}
"""

        system_prompt = f"""You are showing a student their learning progress.

{progress_summary}

Career goal: {state['user_profile'].get('career_context', {}).get('target_role', 'Not set')}

Provide:
- Celebration of achievements
- Progress toward career goal
- Encouragement
- Next recommended steps"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["last_message"]},
        ]

        response = self.llm.invoke(messages)

        return {**state, "response": response.content}

    def course_recommendation(self, state: StudentState) -> StudentState:
        """Recommend next courses based on goals and progress.

        This version does not call Web3 directly; it relies on the agent's
        general knowledge and the completed courses context.
        """

        career_goal = state["user_profile"].get("career_context", {}).get(
            "target_role", "Web3 developer"
        )

        completed_titles = [
            c.get("title", f"Course ID {c.get('course_id', '?')}")
            for c in state["completed_courses"]
        ]

        system_prompt = f"""You are a course recommendation assistant for a Web3 learning platform.

User wants to become: {career_goal}
They've completed: {completed_titles if completed_titles else "No courses yet"}

Based on this, suggest the top 3 next course topics or learning modules they should take,
and explain briefly why each is important for their goal. You don't know the exact
course catalog; focus on topics and learning objectives."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["last_message"]},
        ]

        response = self.llm.invoke(messages)

        return {**state, "response": response.content}
    
    def general_conversation(self, state: StudentState) -> StudentState:
        """General helpful conversation."""

        system_prompt = f"""You are a friendly learning companion for a Web3 education platform.

User's goal: {state['user_profile'].get('career_context', {}).get('target_role', 'learning Web3')}

Be helpful, encouraging, and guide them toward their learning goals."""

        messages = [
            {"role": "system", "content": system_prompt},
            *state["conversation_history"],
            {"role": "user", "content": state["last_message"]},
        ]

        response = self.llm.invoke(messages)

        return {**state, "response": response.content}

    def update_user_profile_node(self, state: StudentState) -> StudentState:
        """Save any profile updates to DB and log the conversation."""

        if state.get("profile_updates"):
            update_user_profile(
                self.db, state["wallet_address"], state["profile_updates"]
            )

        # Save conversation
        save_conversation(
            self.db, state["wallet_address"], "user", state["last_message"]
        )
        save_conversation(
            self.db, state["wallet_address"], "assistant", state["response"]
        )

        return state

    def invoke(self, initial_state: dict) -> dict:
        """Execute the compiled agent graph."""
        return self.graph.invoke(initial_state)
