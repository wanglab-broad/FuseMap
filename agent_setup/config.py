import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Load environment variables from .env file
# Find the project root and load .env from there
project_root = Path(__file__).parent.parent
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)


# Supported model configurations.
# "temperature" is the value passed to the LLM constructor. Use None for models
# that only accept their default temperature (e.g. GPT-5 reasoning models reject
# temperature=0); None means the parameter is omitted and the provider default is used.
SUPPORTED_MODELS = {
    "gpt-4o": {"provider": "openai", "model_name": "gpt-4o", "temperature": 0},
    "gpt-5": {"provider": "openai", "model_name": "gpt-5", "temperature": None},
    "claude-sonnet-4": {"provider": "anthropic", "model_name": "claude-sonnet-4-20250514", "temperature": 0},
    # Legacy alias for backward compatibility
    "claude-3.7": {"provider": "anthropic", "model_name": "claude-sonnet-4-20250514", "temperature": 0},
}


def is_anthropic_model(model_choice: str) -> bool:
    """Check if the given model choice is an Anthropic model."""
    model_info = SUPPORTED_MODELS.get(model_choice, {})
    return model_info.get("provider") == "anthropic"


def is_openai_model(model_choice: str) -> bool:
    """Check if the given model choice is an OpenAI model."""
    model_info = SUPPORTED_MODELS.get(model_choice, {})
    return model_info.get("provider") == "openai"


def create_llm(model_choice="gpt-4o", api_key=None, base_url=None):
    """
    Create and return an LLM instance based on the provided parameters.

    Args:
        model_choice (str): The model to use (see SUPPORTED_MODELS for options)
        api_key (str): The API key for the LLM service
        base_url (str): Optional base URL for custom endpoints

    Returns:
        The configured LLM instance
    """
    if not api_key:
        raise ValueError("API key is required")

    if model_choice not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model choice: {model_choice}. "
            f"Supported models: {list(SUPPORTED_MODELS.keys())}"
        )

    model_info = SUPPORTED_MODELS[model_choice]
    provider = model_info["provider"]
    model_name = model_info["model_name"]
    temperature = model_info.get("temperature", 0)

    if provider == "openai":
        kwargs = dict(model=model_name, api_key=api_key)
        if temperature is not None:
            kwargs["temperature"] = temperature
        if base_url and base_url.strip():
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)
    elif provider == "anthropic":
        kwargs = dict(model=model_name, api_key=api_key)
        if temperature is not None:
            kwargs["temperature"] = temperature
        if base_url and base_url.strip():
            kwargs["base_url"] = base_url
        return ChatAnthropic(**kwargs)
