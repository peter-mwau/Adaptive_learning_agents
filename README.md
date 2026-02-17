# Adaptive Learning Agents

An intelligent AI-powered learning platform that uses autonomous agents to personalize education and career guidance for students and support course creation and evaluation for instructors.

## 🎯 Overview

This project implements a multi-agent system leveraging **LangChain** and **LLMs** (Anthropic Claude, Google Gemini) to create adaptive learning experiences. The platform serves two primary user groups:

### For Students (Learners)
- **Career Guidance Agent**: Provides onboarding and ongoing career advice
- **Learning Assistant**: Offers support while taking courses
- Browse and enroll in curated courses

### For Instructors (Moderators)
- **Course Evaluation Agent**: Analyzes course quality and provides grading
- **Content Generation Agent**: Creates quizzes and improves educational materials
- Create and manage courses

## 🚀 Features

- **Adaptive Learning Management**: Personalized course recommendations based on student profiles
- **Multi-Agent Architecture**: Independent agents for different learning contexts
- **LLM Integration**: Support for Claude and Google Gemini LLMs
- **Database Persistence**: SQLAlchemy ORM with PostgreSQL backend
- **RESTful API**: FastAPI for seamless integration
- **Career Profiling**: Build comprehensive student skill and career profiles
- **Conversation History**: Track all agent-student interactions

## 📦 Tech Stack

- **Backend**: FastAPI, SQLAlchemy, Alembic
- **AI/ML**: LangChain, LangGraph, Claude (Anthropic), Gemini (Google)
- **Database**: PostgreSQL
- **ORM**: SQLAlchemy 2.0+
- **Testing**: pytest, pytest-asyncio, pytest-cov
- **Server**: Uvicorn

## 🔧 Installation

### Prerequisites
- Python 3.13+
- PostgreSQL
- [uv](https://docs.astral.sh/uv/) (modern Python package installer)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Adaptive_learning_agents.git
   cd Adaptive_learning_agents
   ```

2. **Install dependencies**
   ```bash
   For Macos/Linux:
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   For Windows:
   powershell -c "irm https://astral.sh/uv/install.ps1 | more"

   uv sync
   ```

3. **Configure environment variables**
   Create a `.env` file with:
   ```
   DATABASE_URL=postgresql://user:password@localhost/adaptive_learning
   ANTHROPIC_API_KEY=your_anthropic_key
   GOOGLE_API_KEY=your_google_key
   ```

4. **Initialize the database**
   ```bash
   alembic upgrade head
   ```

5. **Run the server**
   ```bash
   uv run fastapi dev app/main.py
   ```

   The API will be available at `http://localhost:8000`

## 📚 Project Structure

```
adaptive-learning-agents/
├── app/
│   ├── agents/
│   │   └── student_agent.py        # Student companion agent
│   ├── models.py                   # SQLAlchemy database models
│   ├── schemas.py                  # Pydantic request/response schemas
│   ├── database.py                 # Database configuration
│   ├── config.py                   # Application settings
│   ├── main.py                     # FastAPI application
│   └── tests/
│       ├── conftest.py             # Test configuration
│       └── test_student_agent.py   # Agent tests
├── alembic/                        # Database migrations
├── pyproject.toml                  # Project dependencies
└── pytest.ini                      # Test configuration
```

## 🔌 API Endpoints

### Health Check
```
GET /health
```

### Career Onboarding
```
POST /api/career-onboarding
Content-Type: application/json

{
  "name": "John Doe",
  "email": "john@example.com",
  "career_context": {...},
  "agreeToTerms": true
}
```

### User Profile Management
```
GET /api/users/{wallet_address}
PUT /api/users/{wallet_address}
```

### Chat Interface
```
POST /api/student-chat
Content-Type: application/json

{
  "wallet_address": "0x...",
  "message": "How can I improve my Python skills?"
}
```

## 🗄️ Database Models

- **UserProfile**: Core user data and learning preferences
- **Conversation**: Chat history between agents and students
- **CourseRecommendation**: Personalized course recommendations
- **AgentAnalytics**: Agent interaction metrics and insights

## 🧪 Testing

Run the test suite with coverage:
```bash
uv run pytest --cov=app tests/
```

Run specific test files:
```bash
uv run pytest app/tests/test_student_agent.py -v
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Create a feature branch (`git checkout -b feature/amazing-feature`)
2. Commit your changes (`git commit -m 'Add amazing feature'`)
3. Push to the branch (`git push origin feature/amazing-feature`)
4. Open a Pull Request

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📧 Support

For issues and questions, please open an issue on GitHub.

---