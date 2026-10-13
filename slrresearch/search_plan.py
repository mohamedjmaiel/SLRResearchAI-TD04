import json
import re
from typing import Callable, Literal

from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field, field_validator, model_validator


class ResearchQuestion(BaseModel):
    """Contrat des données utilisables après le contrôle textuel."""

    identifier: str = Field(pattern=r"^RQ[0-9]+$")
    question: str = Field(min_length=15)


class WorkflowOutcome(BaseModel):
    """État observable du workflow après validation et repair éventuel."""

    status: Literal["accepted", "repaired", "human_required"]
    text: str
    questions: list[ResearchQuestion] = Field(default_factory=list)
    error: str | None = None
    repair_attempts: int = 0


RQ_PATTERN = re.compile(r"^(RQ[0-9]+)\.\s*(.{15,})$")

# ---------------------------------------------------------------------------
# Requêtes bibliographiques dynamiques : approche hybride
#
#   RQ validées
#       ↓  LLM      : extraction des concepts et proposition de synonymes
#       ↓  Pydantic : contrôle de la structure            (ConceptExtraction)
#       ↓  humain   : validation, suppression, ajout       (SearchConcepts)
#       ↓  Python   : construction booléenne déterministe  (build_boolean_query)
#       ↓  adaptateurs : Scholar, IEEE Xplore, ACM DL      (build_*_query)
#
# Le LLM interprète, Pydantic impose le contrat, l'humain décide de la
# stratégie scientifique, Python produit une syntaxe reproductible.
# ---------------------------------------------------------------------------

SEARCH_ENGINES = ("scholar", "ieee", "acm")

MIN_CONCEPTS = 1            # 1 seul concept : accepté mais signalé (requête large)
MAX_CONCEPTS = 4            # au-delà, trop de AND : silence documentaire
MAX_TERMS_PER_CONCEPT = 6   # au-delà, bruit ; 4 x 6 = 24 termes <= limite IEEE
MIN_TERM_LENGTH = 2
MAX_TERM_LENGTH = 80
TERM_LANGUAGE = "anglais"   # langue dominante des bases indexées

SCHOLAR_MAX_LENGTH = 256    # Google Scholar ignore la suite d'une requête trop longue
IEEE_MAX_TERMS = 25         # IEEE Xplore Command Search : 25 termes au plus
IEEE_DEFAULT_FIELD = "All Metadata"
ACM_DEFAULT_FIELD = "Abstract"

BOOLEAN_OPERATORS = frozenset({"AND", "OR", "NOT", "NEAR", "ONEAR"})
# Caractères qui casseraient la syntaxe booléenne ou les séparateurs de saisie :
# guillemets, parenthèses, crochets, accolades, jokers, pipe, deux-points
# (préfixe de champ IEEE/ACM) et point-virgule (séparateur de la revue humaine).
FORBIDDEN_TERM_CHARS = re.compile(r"[\"“”«»()\[\]{}*?|:;]")
WHITESPACE = re.compile(r"\s+")
JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def normalize_term(term: str) -> str:
    """Nettoie un terme : caractères interdits, espaces multiples, ponctuation."""
    cleaned = FORBIDDEN_TERM_CHARS.sub(" ", str(term))
    return WHITESPACE.sub(" ", cleaned).strip(" .,-'")


class ConceptGroup(BaseModel):
    """Un concept de la question et ses variantes terminologiques (OR).

    `terms[0]` est le terme principal ; les suivants sont ses synonymes,
    acronymes ou variantes orthographiques.
    """

    label: str
    terms: list[str]

    @field_validator("label")
    @classmethod
    def clean_label(cls, value: str) -> str:
        cleaned = normalize_term(value)
        if len(cleaned) < MIN_TERM_LENGTH:
            raise ValueError("Libellé de concept vide")
        return cleaned

    @field_validator("terms")
    @classmethod
    def clean_terms(cls, values: list[str]) -> list[str]:
        """Normalise, écarte les opérateurs, dédoublonne sans tenir compte de la casse."""
        seen: set[str] = set()
        cleaned_terms: list[str] = []
        for raw in values:
            term = normalize_term(raw)
            if len(term) < MIN_TERM_LENGTH or term.upper() in BOOLEAN_OPERATORS:
                continue
            if len(term) > MAX_TERM_LENGTH:
                raise ValueError(
                    f"Terme trop long (> {MAX_TERM_LENGTH} caractères) : {term[:40]}..."
                )
            if term.casefold() in seen:
                continue
            seen.add(term.casefold())
            cleaned_terms.append(term)
        if not cleaned_terms:
            raise ValueError("Aucun terme exploitable dans ce concept")
        if len(cleaned_terms) > MAX_TERMS_PER_CONCEPT:
            raise ValueError(
                f"Au plus {MAX_TERMS_PER_CONCEPT} termes par concept "
                f"({len(cleaned_terms)} reçus)"
            )
        return cleaned_terms


class ConceptExtraction(BaseModel):
    """Contrat de la sortie du LLM : concepts distincts d'une question."""

    concepts: list[ConceptGroup] = Field(
        min_length=MIN_CONCEPTS,
        max_length=MAX_CONCEPTS,
    )

    @model_validator(mode="after")
    def check_distinct_concepts(self):
        """Libellés distincts ; un même terme n'appartient qu'à un seul concept."""
        labels = [group.label.casefold() for group in self.concepts]
        if len(set(labels)) != len(labels):
            raise ValueError("Deux concepts portent le même libellé")
        owner: dict[str, str] = {}
        for group in self.concepts:
            for term in group.terms:
                key = term.casefold()
                if key in owner and owner[key] != group.label:
                    raise ValueError(
                        f"Le terme '{term}' apparaît dans les concepts "
                        f"'{owner[key]}' et '{group.label}'"
                    )
                owner[key] = group.label
        return self


class ConceptProposal(BaseModel):
    """Proposition automatique pour une RQ, avant toute décision humaine."""

    identifier: str = Field(pattern=r"^RQ[0-9]+$")
    question: str
    status: Literal["accepted", "repaired", "human_required"]
    extraction: ConceptExtraction | None = None
    raw_output: str = ""
    error: str | None = None
    repair_attempts: int = 0

    @model_validator(mode="after")
    def check_consistency(self):
        if self.status == "human_required" and not self.error:
            raise ValueError("Une escalade humaine doit porter une erreur")
        if self.status != "human_required" and self.extraction is None:
            raise ValueError("Une proposition acceptée doit porter des concepts")
        return self


class ReviewLog(BaseModel):
    """Trace des décisions humaines (reproductibilité du protocole SLR)."""

    added_concepts: list[str] = Field(default_factory=list)
    removed_concepts: list[str] = Field(default_factory=list)
    added_terms: dict[str, list[str]] = Field(default_factory=dict)
    removed_terms: dict[str, list[str]] = Field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not (
            self.added_concepts or self.removed_concepts
            or self.added_terms or self.removed_terms
        )


class SearchConcepts(ConceptExtraction):
    """Concepts validés par l'humain : seule entrée admise par les adaptateurs."""

    identifier: str = Field(pattern=r"^RQ[0-9]+$")
    question: str = Field(min_length=15)
    origin: Literal["llm_accepted", "human_edited", "human_authored"]
    review: ReviewLog = Field(default_factory=ReviewLog)


def extract_non_empty_lines(text: str) -> list[str]:
    """Nettoie le texte et retourne les lignes non vides."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        raise ValueError("Texte vide")
    return lines


def parse_rq_line(line: str) -> dict:
    """Contrôle toute la ligne et extrait son identifiant et sa question."""
    match = RQ_PATTERN.fullmatch(line)
    if not match:
        raise ValueError(f"Ligne non conforme : {line}")
    return {
        "identifier": match.group(1),
        "question": match.group(2).strip(),
    }

def validate_identifiers(records: list[dict]) -> None:
    """Exige exactement RQ1, RQ2 et RQ3 dans cet ordre."""
    identifiers = [record["identifier"] for record in records]
    if identifiers != ["RQ1", "RQ2", "RQ3"]:
        raise ValueError("Identifiants attendus : RQ1, RQ2, RQ3")


def validate_no_duplicates(records: list[dict]) -> None:
    """Refuse deux formulations identiques."""
    questions = [record["question"] for record in records]
    if len(set(questions)) != len(questions):
        raise ValueError("Les trois questions doivent être différentes")


def validate_free_text(text: str) -> list[dict]:
    """Compose les contrôles textuels déterministes."""
    lines = extract_non_empty_lines(text)
    records = [parse_rq_line(line) for line in lines]
    validate_identifiers(records)
    validate_no_duplicates(records)
    return records


def build_few_shot_prompt(topic: str) -> str:
    """Construit un prompt avec deux exemples de sorties conformes."""
    clean = topic.strip()
    if not clean:
        raise ValueError("Sujet vide")
    return f"""Propose exactement trois questions de recherche.
N'ajoute ni introduction, ni conclusion, ni JSON.

Exemple 1 - Sujet : IA et éducation
RQ1. Comment l'IA influence-t-elle la personnalisation pédagogique ?
RQ2. Quels risques l'IA introduit-elle dans l'évaluation ?
RQ3. Quel contrôle humain doit-il être conservé ?

Exemple 2 - Sujet : DevOps et qualité
RQ1. Comment DevOps influence-t-il la fréquence des déploiements ?
RQ2. Quels facteurs réduisent-ils les défauts en production ?
RQ3. Quelles métriques évaluent-elles cette amélioration ?

Sujet courant : {clean}
Retourne exactement trois lignes RQ1., RQ2. et RQ3."""


def build_repair_prompt(text: str, error: str) -> str:
    """Demande au LLM une correction de forme ciblée."""
    return f"""Tu corriges la forme d'une sortie destinée à un programme.

[TÂCHE]
Corrige uniquement la forme du texte.
Préserve le sens des trois questions.

[ERREUR DÉTECTÉE]
{error}

[CONTRAT ATTENDU]
- Exactement trois lignes.
- Chaque ligne commence respectivement par RQ1., RQ2. et RQ3.
- N'ajoute ni introduction, ni explication, ni Markdown, ni JSON.

[EXEMPLE CONFORME]
RQ1. Comment l'IA influence-t-elle la personnalisation pédagogique ?
RQ2. Quels risques l'IA introduit-elle dans l'évaluation ?
RQ3. Quel contrôle humain doit-il être conservé ?

[TEXTE À CORRIGER]
{text}

[SORTIE]
"""


def repair_with_llm(model, text: str, error: str) -> str:
    """Code fourni : un seul appel conditionnel de réparation."""
    response = model.invoke(build_repair_prompt(text, error))
    return str(response.content).strip()


#def validate_questions(records: list[dict]) -> list[ResearchQuestion]:
#    """Applique le contrat Pydantic à chaque dictionnaire."""
    #return [ResearchQuestion.model_validate(record) for record in records]

def validate_questions(records: list[dict]) -> list[ResearchQuestion]:
    """TODO 4 - Appliquer le contrat Pydantic à chaque dictionnaire."""
    questions = []
    for record in records:
        # TODO 4 - Remplacer None par l'appel Pydantic indiqué dans l'énoncé.
        question = ResearchQuestion.model_validate(record)
        questions.append(question)
    return questions


# ---------------------------------------------------------------------
# Étapes 1 et 2 : le LLM propose, Pydantic contrôle (repair borné à 1)
# ---------------------------------------------------------------------

def build_concept_prompt(question: str, topic: str = "") -> str:
    """Demande au LLM les concepts de la question et leurs variantes (JSON)."""
    clean = question.strip()
    if not clean:
        raise ValueError("Question vide")
    context = (
        f"\n[CONTEXTE]\nSujet de la revue : {topic.strip()}\n"
        "Utilise ce sujet pour expliciter les références implicites de la "
        "question (ex. « cette amélioration », « ce domaine »).\n"
        if topic and topic.strip()
        else ""
    )
    return f"""Tu prépares une recherche bibliographique dans des bases scientifiques.

[TÂCHE]
Identifie les concepts principaux de la question de recherche.
Pour chaque concept, donne les termes équivalents employés dans la
littérature : terme principal d'abord, puis synonymes, acronymes,
variantes orthographiques ou formulations voisines.
{context}
[RÈGLES]
- Entre {MIN_CONCEPTS} et {MAX_CONCEPTS} concepts réellement distincts (idéalement 2 ou 3).
- Entre 1 et {MAX_TERMS_PER_CONCEPT} termes par concept, en {TERM_LANGUAGE}.
- Aucun opérateur (AND, OR, NOT), guillemet, parenthèse ni joker.
- Un même terme n'appartient qu'à un seul concept.
- Ignore les mots interrogatifs ou génériques (comment, quels, impact,
  facteurs, étude) sauf s'ils sont l'objet même de la question.
- Retourne uniquement un objet JSON, sans texte ni Markdown.

[FORMAT]
{{"concepts": [{{"label": "...", "terms": ["...", "..."]}}]}}

[EXEMPLE 1]
Question : Comment le télétravail influence-t-il la productivité des développeurs ?
{{"concepts": [
  {{"label": "remote work", "terms": ["remote work", "telework", "telecommuting", "work from home"]}},
  {{"label": "productivity", "terms": ["productivity", "performance", "efficiency"]}},
  {{"label": "software developers", "terms": ["software developers", "software engineers", "programmers"]}}
]}}

[EXEMPLE 2]
Question : Quels facteurs favorisent l'adoption de l'irrigation goutte-à-goutte par les petits exploitants ?
{{"concepts": [
  {{"label": "drip irrigation", "terms": ["drip irrigation", "trickle irrigation", "micro-irrigation"]}},
  {{"label": "adoption", "terms": ["adoption", "uptake", "diffusion"]}},
  {{"label": "smallholder farmers", "terms": ["smallholder farmers", "small-scale farmers", "family farms"]}}
]}}

[QUESTION]
{clean}

[SORTIE]
"""


def build_concept_repair_prompt(question: str, text: str, error: str) -> str:
    """Demande une correction ciblée d'une extraction de concepts invalide."""
    return f"""Tu corriges une sortie JSON destinée à un programme.

[TÂCHE]
Corrige uniquement la forme de la sortie. Préserve les concepts de la question.

[ERREUR DÉTECTÉE]
{error}

[CONTRAT ATTENDU]
- Un seul objet JSON : {{"concepts": [{{"label": "...", "terms": ["..."]}}]}}
- Entre {MIN_CONCEPTS} et {MAX_CONCEPTS} concepts distincts.
- Entre 1 et {MAX_TERMS_PER_CONCEPT} termes par concept, en {TERM_LANGUAGE}.
- Un terme n'appartient qu'à un seul concept.
- Ni opérateur booléen, ni guillemet interne, ni parenthèse, ni Markdown.

[QUESTION]
{question}

[SORTIE À CORRIGER]
{text}

[SORTIE]
"""


def call_llm(model, prompt: str) -> str:
    """Un appel LLM, retourné sous forme de texte nettoyé."""
    response = model.invoke(prompt)
    return str(response.content).strip()


def extract_json_object(text: str) -> str:
    """Isole l'objet JSON d'une sortie LLM (tolère ```json``` et du texte autour)."""
    clean = (text or "").strip()
    if not clean:
        raise ValueError("Sortie LLM vide")
    fenced = JSON_FENCE.search(clean)
    if fenced:
        clean = fenced.group(1).strip()
    start, end = clean.find("{"), clean.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("Aucun objet JSON dans la sortie LLM")
    return clean[start:end + 1]


def parse_concept_output(text: str) -> ConceptExtraction:
    """Sortie LLM -> ConceptExtraction, ou ValueError si vide/invalide/incomplète.

    json.JSONDecodeError et pydantic.ValidationError héritent de ValueError :
    le workflow n'a qu'un seul type d'échec à gérer.
    """
    data = json.loads(extract_json_object(text))
    return ConceptExtraction.model_validate(data)


def propose_concepts_for_question(
    model,
    question: ResearchQuestion,
    topic: str = "",
) -> ConceptProposal:
    """Extrait, valide, répare au plus une fois, puis propose ou escalade.

    Ne construit AUCUNE requête : la proposition attend une décision humaine.
    """
    raw_text = call_llm(model, build_concept_prompt(question.question, topic))
    try:
        extraction = parse_concept_output(raw_text)
        return ConceptProposal(
            identifier=question.identifier,
            question=question.question,
            status="accepted",
            extraction=extraction,
            raw_output=raw_text,
        )
    except ValueError as first_error:
        repaired_text = call_llm(
            model,
            build_concept_repair_prompt(question.question, raw_text, str(first_error)),
        )
        try:
            extraction = parse_concept_output(repaired_text)
            return ConceptProposal(
                identifier=question.identifier,
                question=question.question,
                status="repaired",
                extraction=extraction,
                raw_output=repaired_text,
                repair_attempts=1,
            )
        except ValueError as second_error:
            return ConceptProposal(
                identifier=question.identifier,
                question=question.question,
                status="human_required",
                raw_output=repaired_text,
                error=str(second_error),
                repair_attempts=1,
            )


def propose_concepts(
    model,
    questions: list[ResearchQuestion],
    topic: str = "",
) -> list[ConceptProposal]:
    """Une proposition de concepts par RQ validée, dans l'ordre des RQ."""
    if not questions:
        raise ValueError("Au moins une RQ validée est nécessaire")
    return [propose_concepts_for_question(model, q, topic) for q in questions]


# ---------------------------------------------------------------------------
# Étape 3 : l'humain valide, supprime ou ajoute des termes
# ---------------------------------------------------------------------------

def format_concepts_text(groups: list[ConceptGroup]) -> str:
    """Représentation éditable : une ligne « concept: terme1; terme2 » par concept."""
    return "\n".join(f"{group.label}: {'; '.join(group.terms)}" for group in groups)


def parse_concepts_text(text: str) -> list[dict]:
    """Inverse de format_concepts_text, pour les saisies humaines (CLI, Streamlit)."""
    groups = []
    for number, line in enumerate((x.strip() for x in text.splitlines()), start=1):
        if not line:
            continue
        label, separator, raw_terms = line.partition(":")
        if not separator or not label.strip():
            raise ValueError(
                f"Ligne {number} : format attendu « concept: terme1; terme2 »"
            )
        delimiter = ";" if ";" in raw_terms else ","
        terms = [term.strip() for term in raw_terms.split(delimiter) if term.strip()]
        groups.append({"label": label.strip(), "terms": terms or [label.strip()]})
    if not groups:
        raise ValueError("Aucun concept saisi")
    return groups


def compute_review_log(
    proposed: list[ConceptGroup],
    validated: list[ConceptGroup],
) -> ReviewLog:
    """Compare la proposition du LLM et la décision humaine (par libellé)."""
    before = {group.label.casefold(): group for group in proposed}
    after = {group.label.casefold(): group for group in validated}
    log = ReviewLog(
        added_concepts=[after[k].label for k in after if k not in before],
        removed_concepts=[before[k].label for k in before if k not in after],
    )
    for key in after:
        if key not in before:
            continue
        old = {term.casefold(): term for term in before[key].terms}
        new = {term.casefold(): term for term in after[key].terms}
        added = [new[t] for t in new if t not in old]
        removed = [old[t] for t in old if t not in new]
        if added:
            log.added_terms[after[key].label] = added
        if removed:
            log.removed_terms[after[key].label] = removed
    return log


def apply_human_review(
    proposal: ConceptProposal,
    edited_concepts: list[dict] | list[ConceptGroup],
) -> SearchConcepts:
    """Transforme la décision humaine en SearchConcepts validé par Pydantic.

    `edited_concepts` est la liste finale décidée par l'utilisateur : les
    concepts supprimés en sont absents, les termes ajoutés y figurent. Le
    contrat imposé au LLM s'applique aussi à l'humain (ValidationError).
    """
    groups = [
        item if isinstance(item, ConceptGroup) else ConceptGroup.model_validate(item)
        for item in edited_concepts
    ]
    proposed = proposal.extraction.concepts if proposal.extraction else []
    review = compute_review_log(proposed, groups)
    if not proposed:
        origin = "human_authored"
    elif review.is_empty:
        origin = "llm_accepted"
    else:
        origin = "human_edited"
    return SearchConcepts(
        identifier=proposal.identifier,
        question=proposal.question,
        concepts=groups,
        origin=origin,
        review=review,
    )


def accept_proposal(proposal: ConceptProposal) -> SearchConcepts:
    """Décision humaine explicite : accepter la proposition telle quelle."""
    if proposal.extraction is None:
        raise ValueError(
            f"{proposal.identifier} : aucune proposition exploitable "
            f"({proposal.error}) ; les concepts doivent être saisis par un humain"
        )
    return apply_human_review(proposal, proposal.extraction.concepts)


# ---------------------------------------------------------------------------
# Étape 4 : construction déterministe (OR intra-concept, AND inter-concepts)
# ---------------------------------------------------------------------------

def format_term(term: str) -> str:
    """Met entre guillemets les expressions composées (recherche d'expression)."""
    if re.search(r"[\s\-/]", term):
        return f'"{term}"'
    return term


def ordered_terms(group: ConceptGroup) -> list[str]:
    """Terme principal en tête, variantes triées : sortie indépendante de l'ordre LLM."""
    main, *variants = group.terms
    return [main, *sorted(variants, key=lambda term: (term.casefold(), term))]


def build_or_group(group: ConceptGroup) -> str:
    """Regroupe les termes équivalents d'un concept avec OR."""
    formatted = [format_term(term) for term in ordered_terms(group)]
    if len(formatted) == 1:
        return formatted[0]
    return "(" + " OR ".join(formatted) + ")"


def build_boolean_query(concepts: ConceptExtraction) -> str:
    """Requête générique indépendante du moteur : (a OR b) AND (c OR d)."""
    return " AND ".join(build_or_group(group) for group in concepts.concepts)


# ---------------------------------------------------------------------------
# Étape 5 : adaptateurs par moteur (seule partie qui connaît leurs syntaxes)
# ---------------------------------------------------------------------------

def _render_scholar(groups: list[list[str]]) -> str:
    # Scholar : l'espace vaut AND, OR est prioritaire, parenthèses non documentées.
    return " ".join(" OR ".join(format_term(t) for t in terms) for terms in groups)


def fit_scholar_terms(
    concepts: ConceptExtraction,
    max_length: int = SCHOLAR_MAX_LENGTH,
) -> tuple[list[list[str]], list[str]]:
    """Retire de façon déterministe des variantes jusqu'à respecter la limite.

    On retire la dernière variante du concept qui en a le plus ; le terme
    principal de chaque concept n'est jamais retiré.
    """
    groups = [ordered_terms(group) for group in concepts.concepts]
    dropped: list[str] = []
    while len(_render_scholar(groups)) > max_length:
        candidates = [i for i, terms in enumerate(groups) if len(terms) > 1]
        if not candidates:
            break
        largest = max(candidates, key=lambda i: (len(groups[i]), -i))
        dropped.append(groups[largest].pop())
    return groups, dropped


def build_scholar_query(
    concepts: SearchConcepts,
    max_length: int = SCHOLAR_MAX_LENGTH,
) -> str:
    """Google Scholar : a OR "b c" d OR e (AND implicite, 256 caractères)."""
    groups, _ = fit_scholar_terms(concepts, max_length)
    return _render_scholar(groups)


def build_ieee_query(
    concepts: SearchConcepts,
    field: str = IEEE_DEFAULT_FIELD,
) -> str:
    """IEEE Xplore (Command Search) : chaque terme porte son champ.

    Forme : ("All Metadata":a OR "All Metadata":"b c") AND ("All Metadata":d)
    """
    clauses = []
    for group in concepts.concepts:
        qualified = [f'"{field}":{format_term(t)}' for t in ordered_terms(group)]
        clauses.append("(" + " OR ".join(qualified) + ")")
    return " AND ".join(clauses)


def build_acm_query(
    concepts: SearchConcepts,
    field: str = ACM_DEFAULT_FIELD,
) -> str:
    """ACM Digital Library : un groupe OR par concept, préfixé du champ.

    Forme : Abstract:(a OR "b c") AND Abstract:(d OR e)
    """
    clauses = [
        f"{field}:(" + " OR ".join(format_term(t) for t in ordered_terms(group)) + ")"
        for group in concepts.concepts
    ]
    return " AND ".join(clauses)


QUERY_ADAPTERS = {
    "scholar": build_scholar_query,
    "ieee": build_ieee_query,
    "acm": build_acm_query,
}


def query_warnings(concepts: SearchConcepts) -> list[str]:
    """Signale à l'humain les limites atteintes (sans modifier la stratégie)."""
    warnings = []
    if len(concepts.concepts) == 1:
        warnings.append("Un seul concept : requête probablement trop large")
    _, dropped = fit_scholar_terms(concepts)
    if dropped:
        warnings.append(
            f"Scholar ({SCHOLAR_MAX_LENGTH} caractères max) : variantes retirées : "
            + ", ".join(dropped)
        )
    if len(build_scholar_query(concepts)) > SCHOLAR_MAX_LENGTH:
        warnings.append("Scholar : requête trop longue même avec les termes principaux")
    term_count = sum(len(group.terms) for group in concepts.concepts)
    if term_count > IEEE_MAX_TERMS:
        warnings.append(f"IEEE Xplore : {term_count} termes (> {IEEE_MAX_TERMS})")
    return warnings


def validate_search_concepts(items) -> list[SearchConcepts]:
    """Contrat d'entrée de la partie déterministe : concepts validés, RQ uniques."""
    concepts = [
        item if isinstance(item, SearchConcepts) else SearchConcepts.model_validate(item)
        for item in items
    ]
    if not concepts:
        raise ValueError("Au moins une RQ aux concepts validés est nécessaire")
    identifiers = [item.identifier for item in concepts]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Une même RQ apparaît plusieurs fois")
    return concepts


def build_queries(concepts: list[SearchConcepts]) -> dict[str, dict[str, str]]:
    """Produit, pour chaque RQ validée, une requête adaptée à chaque moteur.

    Aucun appel LLM : à concepts identiques, requêtes identiques.
    Sortie : {"RQ1": {"scholar": ..., "ieee": ..., "acm": ...}, ...}
    """
    validated = validate_search_concepts(concepts)
    return {
        item.identifier: {
            engine: QUERY_ADAPTERS[engine](item) for engine in SEARCH_ENGINES
        }
        for item in validated
    }


# ---------------------------------------------------------------------------
# Chaînes LCEL : la revue humaine est une frontière explicite
# ---------------------------------------------------------------------------

def build_concept_pipeline(model, topic: str = ""):
    """Partie automatisée : regex | Pydantic | LLM + contrat de concepts.

    Entrée : texte des RQ. Sortie : list[ConceptProposal].
    """
    text_control = RunnableLambda(validate_free_text)
    data_contract = RunnableLambda(validate_questions)
    concept_proposer = RunnableLambda(
        lambda questions: propose_concepts(model, questions, topic)
    )
    return text_control | data_contract | concept_proposer


def build_query_pipeline():
    """Partie déterministe : list[SearchConcepts] -> requêtes par moteur."""
    return RunnableLambda(validate_search_concepts) | RunnableLambda(build_queries)


def build_question_pipeline(
    model,
    reviewer: Callable[[ConceptProposal], SearchConcepts],
    topic: str = "",
):
    """Chaîne complète : automatisation | revue humaine | construction déterministe.

    `reviewer` matérialise l'humain : il reçoit une proposition et retourne
    les concepts qu'il valide (ex. revue interactive dans main.py).
    """
    human_review = RunnableLambda(
        lambda proposals: [reviewer(proposal) for proposal in proposals]
    )
    return build_concept_pipeline(model, topic) | human_review | build_query_pipeline()


def run_controlled_workflow(model, text: str) -> WorkflowOutcome:
    """Valide, répare une fois, revalide ou escalade vers un humain."""
    try:
        # Code fourni - chemin nominal : aucune réparation si le texte est valide.
        records = validate_free_text(text)
        questions = validate_questions(records)
        return WorkflowOutcome(
            status="accepted",
            text=text,
            questions=questions,
            repair_attempts=0,
            error=None,
        )

    except ValueError as first_error:
        # Code fourni - une seule tentative de réparation après l'échec.
        repaired_text = repair_with_llm(
            model,
            text,
            str(first_error),
        )
        try:
            # Correction 7.1 : le texte réparé repasse par exactement les
            # mêmes contrôles déterministes et Pydantic que le texte initial.
            repaired_records = validate_free_text(repaired_text)
            repaired_questions = validate_questions(repaired_records)

            return WorkflowOutcome(
                status="repaired",
                text=repaired_text,
                questions=repaired_questions,
                repair_attempts=1,
                error=None,
            )

        except ValueError as second_error:
            # Correction 7.2 : après l'unique repair autorisé, le système
            # s'abstient et transmet la décision à un humain.
            return WorkflowOutcome(
                status="human_required",
                text=repaired_text,
                questions=[],
                repair_attempts=1,
                error=str(second_error),
            )
