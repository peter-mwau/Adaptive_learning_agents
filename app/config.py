from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./learning_agents.db"
    ANTHROPIC_API_KEY: str
    ENVIRONMENT: str = "development"
    SECRET_KEY: str = "dev-secret-key"
    
    class Config:
        env_file = ".env"

settings = Settings()