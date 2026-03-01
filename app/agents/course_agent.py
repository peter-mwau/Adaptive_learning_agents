"""
Course Evaluation Agent
-----------------------
Stateless LangGraph-powered agent that evaluates course text against the
ABYA University rubric.

Consistent with StudentCompanionAgent:
  - Uses LangChain ChatGoogleGenerativeAI
  - Uses LangGraph StateGraph
  - Same _extract_text / _safe_parse_json helpers

Pipeline (3 nodes):
    categorise_node → grade_node → score_node → END

No database writes – results are returned directly to the caller.
"""

import json
import operator
import re
from typing import Annotated, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from app.config import settings

# ---------------------------------------------------------------------------
# Rubric configuration
# ---------------------------------------------------------------------------

RUBRIC_WEIGHTS: dict[str, dict[str, int]] = {
    "Blockchain Technology and Development": {
        "Learner Agency": 10,
        "Critical Thinking": 19,
        "Collaborative Learning": 10,
        "Reflective Practice": 5,
        "Adaptive Learning": 9,
        "Authentic Learning": 14,
        "Technology Integration": 19,
        "Learner Support": 5,
        "Assessment for Learning": 5,
        "Engagement and Motivation": 4,
    },
    "Web3 Development and Design": {
        "Learner Agency": 15,
        "Critical Thinking": 15,
        "Collaborative Learning": 15,
        "Reflective Practice": 10,
        "Adaptive Learning": 10,
        "Authentic Learning": 10,
        "Technology Integration": 10,
        "Learner Support": 5,
        "Assessment for Learning": 5,
        "Engagement and Motivation": 5,
    },
    "Blockchain Applications and Business": {
        "Learner Agency": 10,
        "Critical Thinking": 20,
        "Collaborative Learning": 15,
        "Reflective Practice": 10,
        "Adaptive Learning": 10,
        "Authentic Learning": 15,
        "Technology Integration": 5,
        "Learner Support": 5,
        "Assessment for Learning": 5,
        "Engagement and Motivation": 5,
    },
    "Web3 Ecosystem and Operations": {
        "Learner Agency": 16,
        "Critical Thinking": 16,
        "Collaborative Learning": 16,
        "Reflective Practice": 10,
        "Adaptive Learning": 11,
        "Authentic Learning": 10,
        "Technology Integration": 5,
        "Learner Support": 5,
        "Assessment for Learning": 5,
        "Engagement and Motivation": 6,
    },
    "Emerging Technologies and Intersections": {
        "Learner Agency": 14,
        "Critical Thinking": 19,
        "Collaborative Learning": 14,
        "Reflective Practice": 10,
        "Adaptive Learning": 10,
        "Authentic Learning": 14,
        "Technology Integration": 5,
        "Learner Support": 5,
        "Assessment for Learning": 4,
        "Engagement and Motivation": 5,
    },
}

COURSE_CLUSTERS: list[str] = list(RUBRIC_WEIGHTS.keys())
EVALUATION_ELEMENTS: list[str] = list(
    RUBRIC_WEIGHTS["Blockchain Technology and Development"].keys()
)
PASS_MARK: int = 80


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class EvaluationState(TypedDict):
    # Input
    course_content: str

    # Intermediate / output
    category: str | None
    grades: dict | None
    final_score: float | None
    passed: bool | None
    pass_mark: int
    error: str | None

    # Internal message log (mirrors StudentCompanionAgent pattern)
    messages: Annotated[list, operator.add]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class CourseEvaluationAgent:
    """
    LangGraph-based stateless course evaluation agent.

    Usage::

        agent = CourseEvaluationAgent()
        result = agent.evaluate("Full course text here...")
    """

    def __init__(self):
        # Same model + client as StudentCompanionAgent
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            api_key=settings.GEMINI_API_KEY,
            temperature=0.4,
        )
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Shared helpers (same pattern as StudentCompanionAgent)
    # ------------------------------------------------------------------

    def _extract_text(self, response) -> str:
        """Extract plain text from a LangChain LLM response."""
        if isinstance(response.content, str):
            return response.content
        elif isinstance(response.content, list):
            parts = []
            for block in response.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(parts)
        return str(response.content)

    def _safe_parse_json(self, raw: str) -> dict | None:
        """Strip markdown fences and parse the first JSON object found."""
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
        match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1)
        try:
            return json.loads(cleaned)
        except Exception as exc:
            print(f"[CourseEvaluationAgent] JSON parse error: {exc}")
            return None

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------

    def _categorise_node(self, state: EvaluationState) -> EvaluationState:
        """Node 1 – classify the course into one of the rubric clusters."""
        clusters_list = "\n".join(f"- {c}" for c in COURSE_CLUSTERS)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a course classification engine. "
                    "Respond ONLY with the exact cluster name from the provided list. "
                    "No extra text, no punctuation."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Categorise this course into EXACTLY ONE of these clusters:\n"
                    f"{clusters_list}\n\n"
                    f"Course Content (first 2000 chars):\n---\n"
                    f"{state['course_content'][:2000]}\n---"
                ),
            },
        ]

        response = self.llm.invoke(messages)
        raw = self._extract_text(response).strip()

        # Fuzzy match to handle minor model embellishments
        matched = None
        for cluster in COURSE_CLUSTERS:
            if cluster.lower() in raw.lower():
                matched = cluster
                break

        if not matched:
            return {
                **state,
                "category": None,
                "error": f"Could not categorise the course. Model replied: '{raw[:120]}'",
                "messages": [{"role": "assistant", "content": raw}],
            }

        return {
            **state,
            "category": matched,
            "error": None,
            "messages": [{"role": "assistant", "content": raw}],
        }

    def _grade_node(self, state: EvaluationState) -> EvaluationState:
        """Node 2 – score each rubric element for the detected category."""
        # Propagate failure from previous node
        if state.get("error") or not state.get("category"):
            return state

        category = state["category"]
        weights = RUBRIC_WEIGHTS[category]
        elements_str = "\n".join(
            f"- {el} (Weight: {weights[el]}%)" for el in EVALUATION_ELEMENTS
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert course evaluator using the ABYA University rubric. "
                    "Output ONLY valid JSON. No conversational text, no markdown backticks."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Evaluate the course content below for the '{category}' cluster.\n"
                    f"For each element give an integer score from 0 to 100.\n\n"
                    f"Course Content (first 5000 chars):\n---\n"
                    f"{state['course_content'][:5000]}\n---\n\n"
                    f"Rubric elements:\n{elements_str}\n\n"
                    "Return ONLY a valid JSON object — element names as keys, integer scores as values.\n"
                    f'Example: {{"Learner Agency": 85, "Critical Thinking": 90}}'
                ),
            },
        ]

        response = self.llm.invoke(messages)
        raw = self._extract_text(response)
        parsed = self._safe_parse_json(raw)

        if not parsed:
            return {
                **state,
                "grades": None,
                "error": "Could not parse grading response as JSON.",
                "messages": [{"role": "assistant", "content": raw}],
            }

        missing = [el for el in EVALUATION_ELEMENTS if el not in parsed]
        if missing:
            return {
                **state,
                "grades": None,
                "error": f"Grading response missing elements: {missing}",
                "messages": [{"role": "assistant", "content": raw}],
            }

        if not all(isinstance(parsed[el], (int, float)) for el in EVALUATION_ELEMENTS):
            return {
                **state,
                "grades": None,
                "error": "Grading response contains non-numeric scores.",
                "messages": [{"role": "assistant", "content": raw}],
            }

        grades = {el: max(0, min(100, int(parsed[el]))) for el in EVALUATION_ELEMENTS}

        return {
            **state,
            "grades": grades,
            "error": None,
            "messages": [{"role": "assistant", "content": raw}],
        }

    def _score_node(self, state: EvaluationState) -> EvaluationState:
        """Node 3 – compute the weighted final score and pass/fail result."""
        if state.get("error") or not state.get("grades") or not state.get("category"):
            return {**state, "final_score": None, "passed": None}

        weights = RUBRIC_WEIGHTS[state["category"]]
        final_score = round(
            sum(
                state["grades"][el] * (weights[el] / 100.0)
                for el in EVALUATION_ELEMENTS
            ),
            2,
        )
        passed = final_score >= state["pass_mark"]

        return {**state, "final_score": final_score, "passed": passed}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        """Build and compile the evaluation workflow graph."""
        workflow = StateGraph(EvaluationState)

        workflow.add_node("categorise", self._categorise_node)
        workflow.add_node("grade", self._grade_node)
        workflow.add_node("score", self._score_node)

        workflow.set_entry_point("categorise")

        # Linear pipeline – errors propagate through state fields
        workflow.add_edge("categorise", "grade")
        workflow.add_edge("grade", "score")
        workflow.add_edge("score", END)

        return workflow.compile()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, course_content: str) -> dict:
        """
        Run the full evaluation pipeline on *course_content*.

        Returns a dict matching ``CourseEvaluationResponse``:
        {
            "category": str | None,
            "grades": dict | None,
            "final_score": float | None,
            "passed": bool | None,
            "pass_mark": int,
            "error": str | None,
        }
        """
        if not course_content or not course_content.strip():
            return {
                "category": None,
                "grades": None,
                "final_score": None,
                "passed": None,
                "pass_mark": PASS_MARK,
                "error": "Course content is empty.",
            }

        initial_state: EvaluationState = {
            "course_content": course_content,
            "category": None,
            "grades": None,
            "final_score": None,
            "passed": None,
            "pass_mark": PASS_MARK,
            "error": None,
            "messages": [],
        }

        final_state = self.graph.invoke(initial_state)

        return {
            "category": final_state.get("category"),
            "grades": final_state.get("grades"),
            "final_score": final_state.get("final_score"),
            "passed": final_state.get("passed"),
            "pass_mark": final_state.get("pass_mark", PASS_MARK),
            "error": final_state.get("error"),
        }
