from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    database_url: str = (
        "postgresql+psycopg2://aiops_user:aiops_password@localhost:5432/aiops_db"
    )
    embedding_dim: int = 64

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
