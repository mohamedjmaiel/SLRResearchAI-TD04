"""Résumés et jeu étiqueté du TD04 (code fourni, à lire, à ne pas modifier).

Le screening titre/résumé a besoin du RÉSUMÉ de chaque article. Le TD03 ne
l'a pas conservé : cet outil (lecture seule) le retrouve.
- simulé : data/resumes_td04.json (articles du catalogue du TD03) ;
- réel   : OpenAlex (abstract_inverted_index), sinon résumé indisponible.

Le jeu étiqueté (data/jeu_etiquete_td04.json) sert à MESURER les erreurs d'une
politique de screening avant de l'appliquer à de vrais articles.
"""
import json
import os
from pathlib import Path

import httpx
from pydantic import BaseModel

from slrresearch.models import PaperRecord

DATA = Path(__file__).resolve().parent.parent / "data"
ABSTRACTS_PATH = DATA / "resumes_td04.json"
GOLD_PATH = DATA / "jeu_etiquete_td04.json"


class ScreeningPaper(BaseModel):
    """Ce que le screening titre/résumé examine."""

    id: str
    title: str
    abstract: str | None = None     # None : résumé indisponible
    doi: str | None = None


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def paper_key(paper: PaperRecord) -> str:
    return paper.doi or paper.id


# ---------------------------------------------------------------------------
# Résumés simulés
# ---------------------------------------------------------------------------

def simulated_abstract(paper: PaperRecord) -> str | None:
    entry = load_json(ABSTRACTS_PATH)["abstracts"].get(paper_key(paper))
    return entry["abstract"] if entry else None


# ---------------------------------------------------------------------------
# Résumés réels (OpenAlex)
# ---------------------------------------------------------------------------

def rebuild_abstract(inverted: dict[str, list[int]] | None) -> str | None:
    """OpenAlex stocke le résumé comme {mot: [positions]} : on le reconstruit."""
    if not inverted:
        return None
    positions = {pos: word for word, places in inverted.items() for pos in places}
    return " ".join(positions[i] for i in sorted(positions))


def openalex_abstract(paper: PaperRecord, client: httpx.Client | None = None) -> str | None:
    if paper.doi:
        url = f"https://api.openalex.org/works/doi:{paper.doi}"
    elif paper.id.startswith("https://openalex.org/W"):
        url = "https://api.openalex.org/works/" + paper.id.rsplit("/", 1)[-1]
    else:
        return None
    mailto = os.getenv("OPENALEX_MAILTO", "")
    own = client is None
    client = client or httpx.Client(timeout=15.0, headers={
        "User-Agent": "SLRResearchAI-TD04" + (f" (mailto:{mailto})" if mailto else "")})
    try:
        response = client.get(url)
        if response.status_code != 200:
            return None
        return rebuild_abstract(response.json().get("abstract_inverted_index"))
    except httpx.HTTPError:
        return None                      # réseau indisponible : résumé inconnu
    finally:
        if own:
            client.close()


def fetch_abstracts(papers: list[PaperRecord], sources: str = "simule") -> list[ScreeningPaper]:
    """PaperRecord du TD03 -> ScreeningPaper avec résumé (ou None)."""
    finder = simulated_abstract if sources == "simule" else openalex_abstract
    return [ScreeningPaper(id=p.id, title=p.title, doi=p.doi, abstract=finder(p)) for p in papers]


# ---------------------------------------------------------------------------
# Jeu étiqueté
# ---------------------------------------------------------------------------

def load_gold() -> tuple[str, list[ScreeningPaper], dict[str, str]]:
    """(question, articles, étiquettes humaines {id: include|exclude})."""
    data = load_json(GOLD_PATH)
    papers = [ScreeningPaper(id=p["id"], title=p["title"], abstract=p["abstract"]) for p in data["papers"]]
    return data["question"], papers, {p["id"]: p["gold"] for p in data["papers"]}


def gold_concepts() -> list[dict]:
    """Concepts validés (TD02) de la RQ du jeu étiqueté : la base de son protocole."""
    return load_json(GOLD_PATH)["concepts"]


def gold_cases() -> dict[str, str]:
    """Ce que chaque article du jeu étiqueté rend observable."""
    return {p["id"]: p["cas"] for p in load_json(GOLD_PATH)["papers"]}
