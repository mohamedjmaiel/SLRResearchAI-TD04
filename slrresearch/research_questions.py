from dataclasses import dataclass
from time import perf_counter

from langchain_core.messages import HumanMessage, SystemMessage


@dataclass
class FreeTextQuestions:
    text: str
    provider: str
    latency_seconds: float
    system_prompt: str
    user_prompt: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None


def build_system_prompt() -> str:
    """Acquis du TD01 : règles stables envoyées au modèle."""
    return (
        "Tu es un assistant méthodologique spécialisé dans les revues "
        "systématiques de la littérature (SLR). "
        "Reste dans le périmètre du sujet fourni. "
        "N'invente ni référence ni résultat scientifique. "
        "Réponds en français."
    )


def build_user_prompt(topic: str) -> str:
    """Acquis du TD01 : prompt zero-shot structuré en cinq rubriques."""
    clean = topic.strip()
    if not clean:
        raise ValueError("Sujet vide")
    return f"""[OBJECTIF]
Générer exactement trois questions de recherche pour une SLR.

[CONTEXTE]
Les questions serviront ensuite à construire des requêtes scientifiques.

[ENTRÉE]
Sujet de la revue : {clean}

[CONTRAINTES]
Questions ouvertes, distinctes et centrées sur le sujet.
Ne produire ni référence inventée, ni réponse aux questions, ni JSON.

[SORTIE ATTENDUE]
Exactement trois lignes numérotées RQ1., RQ2. et RQ3."""


def build_zero_shot_prompt(topic: str) -> str:
    """Compatibilité pédagogique : retourne le User Prompt du TD01."""
    return build_user_prompt(topic)


def build_zero_shot_messages(topic: str):
    return [
        SystemMessage(content=build_system_prompt()),
        HumanMessage(content=build_user_prompt(topic)),
    ]


def _observable_metadata(response):
    usage = getattr(response, "usage_metadata", None) or {}
    metadata = getattr(response, "response_metadata", None) or {}
    return (
        usage.get("input_tokens"),
        usage.get("output_tokens"),
        metadata.get("finish_reason") or metadata.get("done_reason"),
    )


def generate_questions(
    model, topic: str, provider: str = "demo"
) -> FreeTextQuestions:
    """Acquis du TD01 : un seul appel LangChain, puis observation."""
    messages = build_zero_shot_messages(topic)
    started = perf_counter()
    response = model.invoke(messages)
    latency = perf_counter() - started
    text = str(response.content).strip()
    input_tokens, output_tokens, finish_reason = _observable_metadata(response)
    return FreeTextQuestions(
        text=text,
        provider=provider,
        latency_seconds=latency,
        system_prompt=str(messages[0].content),
        user_prompt=str(messages[1].content),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        finish_reason=finish_reason,
    )
