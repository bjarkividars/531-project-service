from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    azure_ai_endpoint: str
    azure_ai_key: str
    azure_speech_key: str
    gemini_api_key: str
    openai_api_key: str
    class Config:
        env_file = ".env"

settings = Settings() 