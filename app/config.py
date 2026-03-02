from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    # DATABASE_URL: str = "postgresql://user:password@localhost:5432/web3_lms_agents"
    # For development:
    DATABASE_URL: str = "sqlite:///./web3_lms.db"

    # LLM
    ANTHROPIC_API_KEY: str
    GEMINI_API_KEY: str
    # App
    ENVIRONMENT: str = "development"
    DEV_LENIENT_GRADING: bool = True
    

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
