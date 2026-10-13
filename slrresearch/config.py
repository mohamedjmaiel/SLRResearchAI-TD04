import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Configuration choisie dans .env."""

    provider: str = os.getenv("LLM_PROVIDER", "ollama").lower()
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    ollama_base_url: str = os.getenv(
        "OLLAMA_BASE_URL", "http://localhost:11434"
    )
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


@dataclass(frozen=True)
class DecodingParameters:
    mode: str
    temperature: float
    top_p: float
    top_k: int | None
    max_output_tokens: int


def resolve_decoding(
    mode: str,
    sampling_temperature: float = 0.2,
    top_p: float = 0.9,
    max_output_tokens: int = 500,
) -> DecodingParameters:
    """Traduit le mode pédagogique en paramètres fournisseur."""
    clean_mode = mode.strip().lower()
    if clean_mode == "greedy":
        return DecodingParameters("greedy", 0.0, 1.0, 1, max_output_tokens)
    if clean_mode == "sampling":
        return DecodingParameters(
            "sampling", sampling_temperature, top_p, 40, max_output_tokens
        )
    raise ValueError("Le mode doit être 'greedy' ou 'sampling'")


def build_model(
    settings: Settings,
    decoding: DecodingParameters | None = None,
):
    """Construit le ChatModel LangChain sélectionné dans .env."""
    decoding = decoding or resolve_decoding("greedy")

    if settings.provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=decoding.temperature,
            top_p=decoding.top_p,
            top_k=decoding.top_k,
            num_predict=decoding.max_output_tokens,
        )

    if settings.provider == "groq":
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY manquante")
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=decoding.temperature,
            max_tokens=decoding.max_output_tokens,
            model_kwargs={"top_p": decoding.top_p},
        )

    raise ValueError("LLM_PROVIDER doit être 'ollama' ou 'groq'")
