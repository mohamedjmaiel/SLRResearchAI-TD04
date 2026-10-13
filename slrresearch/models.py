"""Contrats de données du TD03 (code fourni, à lire).

PaperRecord : un article normalisé, quelle que soit l'API d'origine.
TraceEvent  : la trace d'une tentative d'action. Au niveau 3, on l'appelle
              aussi « observation » : c'est ce que l'agent apprend après
              chaque action.
ActionError : une action refusée ou invalide, AVANT tout appel externe.
"""
from typing import Literal

from pydantic import BaseModel, Field

CALL_STATUSES = ("success", "rate_limited", "http_error", "network_error")

Status = Literal[
    "success",        # l'outil a répondu (éventuellement 0 article)
    "rate_limited",   # HTTP 429 : quota atteint
    "http_error",     # autre code HTTP
    "network_error",  # connexion impossible
    "refused",        # outil absent du registre ou action interdite
    "invalid",        # action mal formée (contrat AgentAction violé)
]


class PaperRecord(BaseModel):
    """Article normalisé ; `sources` garde la provenance (PRISMA)."""

    id: str
    title: str
    doi: str | None = None
    url: str | None = None
    year: int | None = None
    open_access: bool | None = None
    sources: list[str] = Field(default_factory=list)


class TraceEvent(BaseModel):
    """Une ligne de trace = une tentative d'action, réussie ou non."""

    step: int = 0
    tool: str
    query: str = ""
    status: Status
    found: int = 0
    error: str = ""


# Au niveau 3, une observation est exactement un TraceEvent.
Observation = TraceEvent


class ActionError(ValueError):
    """Action refusée (`refused`) ou invalide (`invalid`) : aucun appel externe."""

    def __init__(self, status: Literal["refused", "invalid"], message: str):
        super().__init__(message)
        self.status = status


def short_error(text: str, limit: int = 140) -> str:
    """Résumé lisible d'un message d'erreur (Pydantic produit plusieurs lignes)."""
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    lines = [line for line in lines if not line.startswith("For further information")]
    summary = " · ".join(lines)
    return summary if len(summary) <= limit else summary[: limit - 1] + "…"
