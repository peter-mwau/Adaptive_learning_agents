from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_anthropic import ChatAnthropic
import operator
import json

from app.database import get_user_profile, update_user_profile, save_conversation, get_conversation_history
from app.config import settings

class StudentState(TypedDict):
    wallet_address: str
    # Loaded context
    user_profile: dict
    conversation_history: Annotated[list, operator.add]
        
    # Blockchain data
    completed_courses: list
    current_course_id: int
    current_chapter: int
        
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
            model="gemini-flash",
            api_key=settings.GEMINI_API_KEY
        )
        self.graph = self.build_graph()
    def build_graph(self):
        # Build the graph here
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
        
        #set_entry
        workflow.set_entry("load_context")
        
        #sequential flow
        workflow.add_edge("load_context", "determine_mode")
        # Lets add conditional routing based on node
        workflow.add_conditional_edges(
                "determine_mode",
                lambda s: s['mode'],
                {
                    "career": "career_mode",
                    "learning": "learning_mode",
                    "progress": "progress_mode",
                    "recommendation": "recommend_mode",
                    "general": "general_mode"
                }
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
            """Load all user context from blockchain, IPFS, and DB."""
            
            wallet = state['wallet_address']
            
            # Load from blockchain
            try:
                enrollments = web3_client.get_user_enrollments(wallet)
                
                completed_courses = []
                current_course = None
                current_chapter = None
                
                for course_id in enrollments:
                    progress = web3_client.get_user_progress(wallet, course_id)
                    course_meta = web3_client.get_course_metadata(course_id)
                    
                    if progress['completed_chapters'] == progress['total_chapters']:
                        completed_courses.append(course_meta)
                    else:
                        # User is currently in this course
                        current_course = course_id
                        current_chapter = progress['completed_chapters']
            
            except Exception as e:
                print(f"Blockchain error: {e}")
                enrollments = []
                completed_courses = []
                current_course = None
                current_chapter = None
            
            # Load from agent DB
            profile = get_user_profile(self.db, wallet) or {}
            history = get_conversation_history(self.db, wallet, limit=10)
            
            return {**state,
                'user_profile': profile,
                'conversation_history': history,
                'completed_courses': completed_courses,
                'current_course_id': current_course,
                'current_chapter': current_chapter,
                'profile_updates': {}
            }
            
    def determine_mode(self, state: StudentState) -> StudentState:
        """Decide what mode to operate in."""
            
        message = state['last_message'].lower()
            
        # Career onboarding for new users
        if not state['completed_courses'] and not state['user_profile'].get('career_context'):
                mode = 'career'
            
        # Learning help if user is in a course
        elif state['current_course_id']:
            # Check if they're asking about progress or recommendations
            if any(kw in message for kw in ['progress', 'how am i doing', 'stats']):
                mode = 'progress'
            elif any(kw in message for kw in ['what next', 'recommend', 'what should i']):
                mode = 'recommendation'
            elif any(kw in message for kw in ['what should i learn', 'what course']):
                mode = 'recommendation'
            
        # Progress review
        elif any(kw in message for kw in ['progress', 'completed', 'achievement']):
            mode = 'progress'
            
        # Career guidance
        elif any(kw in message for kw in ['career', 'job', 'become a', 'work as']):
            mode = 'career'
            
        # Default
        else:
            mode = 'general'
            
        return {**state, 'mode': mode}
    def career_guidance(self, state: StudentState) -> StudentState:
        """Provide career guidance to the student."""
        profile = state['user_profile']
        completed = state['completed_courses']
                
        system_prompt = f"""You are a career guidance AI for a Web3 learning platform.
        User has completed: {[c['title'] for c in completed] if completed else 'No courses yet'}
        Current career context: {profile.get('career_context', 'Unknown')}
        
        Your goal: Help user discover their career path in Web3/tech.
        
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
                    {'role': 'system', 'content': system_prompt},
                    *state['conversation_history'],
                    {'role': 'user', 'content': state['last_message']}
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
            career_context = json.loads(extraction.content.strip('```json').strip('```'))
        except:
            career_context = {}
                
        return {
            **state,
            'response': response.content,
            'profile_updates': {'career_context': career_context} if career_context else {}
        }

    def learning_assistance(self, state: StudentState) -> StudentState:
        """Help user while they're in a course."""
        
        course_id = state['current_course_id']
        chapter = state['current_chapter']
        
        # Get current chapter content from IPFS
        course_content = web3_client.get_course_content(course_meta['ipfs_hash'])
        current_chapter_content = course_content['chapters'][chapter]
        
        system_prompt = 
        f"""You are a learning assistant helping a student in a Web3 course.

        Current Chapter: {chapter + 1} - {current_chapter_content['title']}

        Chapter content summary:
        {current_chapter_content.get('summary', 'No summary available')}

        User's skill level: {state['user_profile'].get('skill_level', 'Unknown')}
                    
        Your role:
        - Answer questions about the current chapter
        - Explain concepts clearly
        - Provide examples and analogies
        - Give hints for exercises (don't give full solutions)
        - Encourage and motivate
                    
        Be patient and adaptive to their level."""

        messages = [
            {'role': 'system', 'content': system_prompt},
            *state['conversation_history'],
            {'role': 'user', 'content': state['last_message']}
        ]
        
        response = self.llm.invoke(messages)
        
        # Track learning challenges
        if any(kw in state['last_message'].lower() for kw in ["don't understand", "confused", "stuck", "difficult"]):
            challenges = state['user_profile'].get('learning_challenges', [])
            topic = current_chapter_content['title']
            if topic not in challenges:
                challenges.append(topic)
            
            profile_updates = {'learning_challenges': challenges}
        else:
            profile_updates = {}
        
        return {
            **state,
            'response': response.content,
            'profile_updates': profile_updates
        }
    
    def progress_review(self, state: StudentState) -> StudentState:
        """Show user their learning progress."""
        
        completed = state['completed_courses']
        current = state['current_course_id']
        
        progress_summary = f"""You've completed {len(completed)} courses:
            {chr(10).join([f"✓ {c['title']}" for c in completed])}

            Currently learning: {web3_client.get_course_metadata(current)['title'] if current else 'No active course'}
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
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': state['last_message']}
        ]
        
        response = self.llm.invoke(messages)
        
        return {
            **state,
            'response': response.content
        }
    
    def course_recommendation(self, state: StudentState) -> StudentState:
        """Recommend next courses based on goals and progress."""
        
        # Get all courses
        all_courses = web3_client.get_all_courses()
        
        # Filter out completed
        completed_ids = [c['course_id'] for c in state['completed_courses']]
        available = [c for c in all_courses if c['course_id'] not in completed_ids]
        
        career_goal = state['user_profile'].get('career_context', {}).get('target_role', 'Web3 developer')
        
        system_prompt = f"""User wants to become: {career_goal}

        They've completed: {[c['title'] for c in state['completed_courses']]}

        Available courses:
            {json.dumps(available, indent=2)}

        Recommend the top 3 most relevant courses and explain why each is important for their goal."""

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': state['last_message']}
        ]
        
        response = self.llm.invoke(messages)
        
        return {
            **state,
            'response': response.content
        }
    
    def general_conversation(self, state: StudentState) -> StudentState:
        """General helpful conversation."""
        
        system_prompt = f"""You are a friendly learning companion for a Web3 education platform.

        User's goal: {state['user_profile'].get('career_context', {}).get('target_role', 'learning Web3')}

        Be helpful, encouraging, and guide them toward their learning goals."""

        messages = [
            {'role': 'system', 'content': system_prompt},
            *state['conversation_history'],
            {'role': 'user', 'content': state['last_message']}
        ]
        
        response = self.llm.invoke(messages)
        
        return {
            **state,
            'response': response.content
        }
    
    def update_user_profile_node(self, state: StudentState) -> StudentState:
        """Save any profile updates to DB."""
        
        if state.get('profile_updates'):
            update_user_profile(
                self.db,
                state['wallet_address'],
                state['profile_updates']
            )
        
        # Save conversation
        save_conversation(self.db, state['wallet_address'], 'user', state['last_message'])
        save_conversation(self.db, state['wallet_address'], 'assistant', state['response'])
        
        return state    
    
    def invoke(self, initial_state: dict) -> dict:
        "Agent Execution"
        return self.graph.invoke(initial_state)
