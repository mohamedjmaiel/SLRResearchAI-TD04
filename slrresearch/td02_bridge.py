"""Pont TD02 -> TD03 (code fourni).

sujet -> RQ validées (TD02) -> concepts proposés par le LLM -> revue humaine
      -> requêtes déterministes {"RQ1": {"scholar": ..., "ieee": ..., "acm": ...}}

Le TD03 consomme le dictionnaire d'UNE RQ : {"scholar": ..., "ieee": ..., "acm": ...}.
"""
from typing import Callable

from slrresearch.search_plan import (
    ConceptProposal,
    SearchConcepts,
    WorkflowOutcome,
    build_few_shot_prompt,
    build_queries,
    call_llm,
    propose_concepts,
    run_controlled_workflow,
)

REFERENCE_TOPIC = (
    "Utilisation des LLM pour assister le screening des SLR en génie logiciel"
)
# Extraction JSON des concepts : budget plus large que pour les RQ (cf. TD02).
CONCEPT_MAX_OUTPUT_TOKENS = 1500


def generate_questions_for_topic(model, topic: str) -> WorkflowOutcome:
    """TD02 : prompt few-shot, contrôle, repair borné, escalade éventuelle."""
    clean = topic.strip()
    if not clean:
        raise ValueError("Sujet vide")
    text = call_llm(model, build_few_shot_prompt(clean))
    return run_controlled_workflow(model, text)


def propose_concepts_for_topic(
    model, outcome: WorkflowOutcome, topic: str
) -> list[ConceptProposal]:
    """TD02 : une proposition de concepts par RQ validée."""
    if outcome.status == "human_required":
        raise ValueError(
            "Les RQ n'ont pas pu être validées : décision humaine requise (TD02)"
        )
    return propose_concepts(model, outcome.questions, topic)


def prepare_queries(
    model,
    topic: str,
    reviewer: Callable[[ConceptProposal], SearchConcepts],
    concept_model=None,
) -> tuple[WorkflowOutcome, list[SearchConcepts], dict[str, dict[str, str]]]:
    """Chaîne complète du TD02, revue humaine comprise."""
    outcome = generate_questions_for_topic(model, topic)
    proposals = propose_concepts_for_topic(concept_model or model, outcome, topic)
    validated = [reviewer(proposal) for proposal in proposals]
    return outcome, validated, build_queries(validated)
