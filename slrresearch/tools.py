"""Outils de recherche du TD03 (code fourni, à lire, à ne pas modifier).

Un outil = un nom unique + une fonction search(query, limit) -> list[dict].
La sortie est BRUTE : chaque API nomme ses champs à sa manière.
Les échecs sont signalés par des exceptions typées (ToolError).

Deux familles d'outils portent les mêmes noms :
- outils simulés : rejouent data/catalog_td03.json (reproductible, sans réseau) ;
- outils réels   : appellent OpenAlex, Crossref et Semantic Scholar.
"""
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx

CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog_td03.json"
TOOL_NAMES = ("openalex", "crossref", "semantic_scholar", "openalex_broad")


# ---------------------------------------------------------------------------
# Erreurs typées : elles font partie du contrat d'un outil
# ---------------------------------------------------------------------------

class ToolError(Exception):
    """Échec d'un outil ; `status` est la valeur à écrire dans la trace."""

    status = "http_error"


class RateLimitError(ToolError):
    status = "rate_limited"


class HttpError(ToolError):
    status = "http_error"


class NetworkError(ToolError):
    status = "network_error"


@dataclass
class SearchTool:
    name: str
    search: Callable[[str, int], list[dict]]
    description: str = ""


# ---------------------------------------------------------------------------
# Normalisation des champs propres à chaque API
# ---------------------------------------------------------------------------

def extract_fields(source: str, raw: dict) -> dict:
    """Ramène une réponse brute à des champs communs (DOI encore brut)."""
    if source in ("openalex", "openalex_broad"):
        return {
            "id": raw.get("id") or "",
            "title": raw.get("display_name") or raw.get("title") or "",
            "doi": raw.get("doi"),
            "url": raw.get("doi") or raw.get("id"),
            "year": raw.get("publication_year"),
            "open_access": (raw.get("open_access") or {}).get("is_oa"),
        }
    if source == "crossref":
        titles = raw.get("title") or [""]
        parts = (raw.get("issued") or {}).get("date-parts") or [[None]]
        return {
            "id": "crossref:" + (raw.get("DOI") or raw.get("URL") or ""),
            "title": titles[0] if titles else "",
            "doi": raw.get("DOI"),
            "url": raw.get("URL"),
            "year": parts[0][0] if parts and parts[0] else None,
            "open_access": None,
        }
    if source == "semantic_scholar":
        return {
            "id": raw.get("paperId") or "",
            "title": raw.get("title") or "",
            "doi": (raw.get("externalIds") or {}).get("DOI"),
            "url": raw.get("url"),
            "year": raw.get("year"),
            "open_access": raw.get("isOpenAccess"),
        }
    raise ValueError(f"Source inconnue : {source}")


# ---------------------------------------------------------------------------
# Adaptation des requêtes TD02 au langage de chaque API
# (l'agent transmet la requête reçue ; l'outil la traduit pour son API)
# ---------------------------------------------------------------------------

FIELD_PREFIX = re.compile(r'"[A-Za-z][A-Za-z ]*"\s*:|\b[A-Z][A-Za-z]*:(?=\s*\()')
TOKEN = re.compile(r'"[^"]*"|\(|\)|[^\s()"]+')
OPERATORS = {"AND", "OR", "NOT"}


def strip_field_prefixes(query: str) -> str:
    """Retire "All Metadata": (IEEE) ou Abstract: (ACM)."""
    return re.sub(r"\s+", " ", FIELD_PREFIX.sub("", query)).strip()


def scholar_to_boolean(query: str) -> str:
    """Scholar `a OR b c OR d` (AND implicite) -> `(a OR b) AND (c OR d)`."""
    clean = strip_field_prefixes(query)
    if " AND " in clean or "(" in clean:
        return clean
    groups: list[list[str]] = []
    previous_or = False
    for token in TOKEN.findall(clean):
        if token == "OR":
            previous_or = True
            continue
        if previous_or and groups:
            groups[-1].append(token)
        else:
            groups.append([token])
        previous_or = False
    return " AND ".join(
        group[0] if len(group) == 1 else "(" + " OR ".join(group) + ")"
        for group in groups
    )


def to_keywords(query: str, max_words: int = 20) -> str:
    """Requête booléenne -> mots-clés (APIs sans syntaxe booléenne)."""
    words: list[str] = []
    for token in TOKEN.findall(strip_field_prefixes(query)):
        if token in ("(", ")") or token.upper() in OPERATORS:
            continue
        for word in token.strip('"').split():
            if word.casefold() not in (w.casefold() for w in words):
                words.append(word)
    return " ".join(words[:max_words])


# ---------------------------------------------------------------------------
# Outils réels (API ouvertes)
# ---------------------------------------------------------------------------

_CLIENT: httpx.Client | None = None


def get_client() -> httpx.Client:
    global _CLIENT
    if _CLIENT is None:
        mailto = os.getenv("OPENALEX_MAILTO", "")
        agent = "SLRResearchAI-TD03" + (f" (mailto:{mailto})" if mailto else "")
        _CLIENT = httpx.Client(timeout=15.0, headers={"User-Agent": agent})
    return _CLIENT


def get_json(url: str, params: dict) -> dict:
    """GET JSON ; transforme chaque échec en erreur typée."""
    try:
        response = get_client().get(url, params=params)
    except httpx.RequestError as error:
        raise NetworkError(f"Connexion impossible : {error.__class__.__name__}") from error
    if response.status_code == 429:
        raise RateLimitError("HTTP 429 : quota atteint")
    if response.status_code >= 400:
        raise HttpError(f"HTTP {response.status_code}")
    return response.json()


def openalex_search(query: str, limit: int) -> list[dict]:
    data = get_json("https://api.openalex.org/works",
                    {"search": scholar_to_boolean(query), "per-page": limit})
    return data.get("results", [])


def openalex_broad_search(query: str, limit: int) -> list[dict]:
    data = get_json("https://api.openalex.org/works",
                    {"search": to_keywords(query), "per-page": limit})
    return data.get("results", [])


def crossref_search(query: str, limit: int) -> list[dict]:
    data = get_json("https://api.crossref.org/works",
                    {"query.bibliographic": to_keywords(query), "rows": limit})
    return (data.get("message") or {}).get("items", [])


def semantic_scholar_search(query: str, limit: int) -> list[dict]:
    data = get_json(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        {"query": to_keywords(query), "limit": limit,
         "fields": "title,externalIds,url,year,isOpenAccess"},
    )
    return data.get("data", []) or []


DESCRIPTIONS = {
    "openalex": "OpenAlex, requête booléenne (base logique scholar)",
    "crossref": "Crossref, mots-clés (base logique ieee)",
    "semantic_scholar": "Semantic Scholar, mots-clés (base logique acm)",
    "openalex_broad": "OpenAlex élargi, mots-clés (source de secours)",
}

REAL_FUNCTIONS = {
    "openalex": openalex_search,
    "crossref": crossref_search,
    "semantic_scholar": semantic_scholar_search,
    "openalex_broad": openalex_broad_search,
}


# ---------------------------------------------------------------------------
# Outils simulés : catalogue contrôlé
# ---------------------------------------------------------------------------

def load_catalog(path: Path = CATALOG_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def scenario_names() -> list[str]:
    return list(load_catalog()["scenarios"])


class FakeSearch:
    """Rejoue les réponses prévues pour un outil, appel après appel."""

    def __init__(self, responses: list):
        self.responses = list(responses) or [[]]
        self.calls: list[str] = []

    def __call__(self, query: str, limit: int) -> list[dict]:
        self.calls.append(query)
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        response = self.responses[index]
        if isinstance(response, dict) and "error" in response:
            error = response["error"]
            if error == 429:
                raise RateLimitError("HTTP 429 : quota atteint (simulé)")
            if error == "network":
                raise NetworkError("Connexion impossible (simulé)")
            raise HttpError(f"HTTP {error} (simulé)")
        return list(response)[:limit]


def build_registry(
    sources: str = "simule",
    scenario: str = "stable",
    allowed: tuple[str, ...] | list[str] | None = None,
) -> dict[str, SearchTool]:
    """Registre des outils autorisés, indexé par nom.

    sources  : "simule" (catalogue) ou "reel" (API ouvertes)
    allowed  : restreint les outils enregistrés (moindre privilège)
    """
    names = [name for name in TOOL_NAMES if allowed is None or name in allowed]
    if sources == "simule":
        plan = load_catalog()["scenarios"][scenario]
        functions = {name: FakeSearch(plan.get(name, [[]])) for name in names}
    elif sources == "reel":
        functions = {name: REAL_FUNCTIONS[name] for name in names}
    else:
        raise ValueError("sources doit valoir 'simule' ou 'reel'")
    return {
        name: SearchTool(name, functions[name], DESCRIPTIONS[name]) for name in names
    }
