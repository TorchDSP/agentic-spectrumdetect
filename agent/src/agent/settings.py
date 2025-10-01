from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    SPECTRUM_SERVER_MCP_SERVER: str = Field(
        description="Base address for the Spectrum_Server MCP", default="http://spectrum_server:8000"
    )
    SPECTRUM_SERVER_MCP_ROUTE: str = Field(
        description="URL route to reach the Spectrum_Server MCP", default="/llm/mcp"
    )
    LLM_API_KEY: str = Field(description="The API key for authenticating to the LLM", default="my-key")
    LLM_API: str = Field(
        description="The URL for the LLM API",
        default="http://vllm:8888/v1",
    )
    LLM_MODEL: str = Field(
        description="The model to use on the LLM API", default="gpt-oss-20b"
    )
    SYSTEM_PROMPT: str = Field(
        description="System prompt for the agent",
        default="You are an expert spectrum analyzer scientific measurement assistant. You can tune a receiver to different parts of the spectrum and detect intermittent or continuous energy. You can determine the RSSI measurement of the energy as well as the spectrogram_diff score of the overall captured bandwidth.",
    )
    CA_CHAIN: str | None = Field(
        description="File path to the CA certificate chain",
        default="/etc/ssl/certs/ca-certificates.crt",
    )
    OTEL_EXPORTER_OTLP_ENDPOINT: str | None = Field(
        description="OpenTelemetry collector endpoint",
        default=None, #"http://lgtm:4318"
    )
    REASONING_LEVEL: str | None = Field(
        description="Level of reasoning to have the model perform. (low, medium, high)",
        default="high",
    )
    model_config = SettingsConfigDict(
        env_prefix="", env_file="env.example", env_file_encoding="utf-8"
    )


config = Settings()
