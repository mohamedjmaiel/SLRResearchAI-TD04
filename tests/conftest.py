"""Fixtures partagées par les tests du TD04 (ne pas modifier)."""
from types import SimpleNamespace

import pytest

from slrresearch.abstracts import ScreeningPaper, load_gold
from slrresearch.demo_llm import DemoChatModel
from slrresearch.screening_agent import DEFAULT_EXCLUSION, Criterion, Protocol

QUESTION = "Comment les LLM améliorent-ils le rappel du screening des SLR ?"

# Protocole de référence : les concepts validés au TD02 pour la RQ1.
INCLUSION = [
    Criterion(id="C1", label="large language models", terms=["large language models", "LLM", "generative AI"]),
    Criterion(id="C2", label="screening", terms=["screening", "study selection", "title and abstract screening"]),
    Criterion(id="C3", label="recall", terms=["recall", "sensitivity"]),
]


@pytest.fixture
def protocol():
    return Protocol(question=QUESTION, inclusion=[c.model_copy() for c in INCLUSION],
                    exclusion=[c.model_copy() for c in DEFAULT_EXCLUSION])


@pytest.fixture
def gold():
    """(articles du jeu étiqueté indexés par id, étiquettes humaines)."""
    _, papers, labels = load_gold()
    return {p.id: p for p in papers}, labels


@pytest.fixture
def demo_model():
    return DemoChatModel()


def make_paper(paper_id="P1", title="Titre", abstract="Résumé."):
    return ScreeningPaper(id=paper_id, title=title, abstract=abstract)


class FakeModel:
    """ChatModel de test : rejoue des sorties texte."""

    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.outputs.pop(0))


@pytest.fixture
def fake_model():
    return FakeModel
