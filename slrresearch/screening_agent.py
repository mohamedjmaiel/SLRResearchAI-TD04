"""ScreeningAgent du TD04 : décider, s'abstenir, escalader.

Pour chaque article (titre + résumé), deux évaluateurs indépendants jugent :
    les RÈGLES (protocole exécutable)   et   le LLM (jugement structuré, contrôlé)
puis une politique de décision choisit l'une de trois issues :
    agir (include / exclude automatique) · s'abstenir · escalader vers un humain.

Partie A  du protocole aux preuves        TODO 1 à 3
Partie B  le LLM, second évaluateur       TODO 4 et 5
Partie C  décider, s'abstenir, escalader  TODO 6 à 8
Partie D  boucler avec l'humain           TODO 9 et 10

Ce fichier est le seul à modifier : TODO 1 à TODO 10.
"""
import re
from typing import Literal

from pydantic import BaseModel, Field

from slrresearch.abstracts import ScreeningPaper
from slrresearch.search_plan import ConceptGroup, call_llm, extract_json_object

DEFAULT_THRESHOLD = 0.75   # confiance minimale pour une décision automatique
EXCLUDE_MARGIN = 0.10      # exclure à tort coûte plus cher : on exige davantage
THRESHOLDS = (0.5, 0.6, 0.75, 0.9)

Decision = Literal["include", "exclude", "uncertain"]
Decider = Literal["regles", "llm", "accord", "humain"]


# ===========================================================================
# Contrats (code fourni, à lire)
# ===========================================================================

class Criterion(BaseModel):
    """Un critère d'éligibilité : satisfait si l'un de ses termes apparaît."""

    id: str                                   # C1, C2... (inclusion) ou E1, E2... (exclusion)
    label: str
    terms: list[str] = Field(min_length=1)


class Protocol(BaseModel):
    """Protocole de screening d'une RQ : tous les C requis, aucun E toléré."""

    question: str
    inclusion: list[Criterion] = Field(min_length=1)
    exclusion: list[Criterion] = Field(default_factory=list)


class Evidence(BaseModel):
    """Une preuve = un critère + un extrait du titre ou du résumé."""

    criterion: str
    quote: str = Field(min_length=3)


class ScreeningDecision(BaseModel):
    """Une décision explicable : quoi, avec quelle confiance, pourquoi, preuves à l'appui."""

    paper_id: str
    decision: Decision
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    decider: Decider


class LLMJudgment(BaseModel):
    """Ce que le LLM doit renvoyer (contrat de sortie, comme au TD02)."""

    decision: Decision
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(min_length=1)
    evidence: list[Evidence] = Field(default_factory=list)


class ScreeningRecord(BaseModel):
    """Le dossier d'un article : deux avis, une issue, une décision finale éventuelle."""

    paper_id: str
    title: str
    rules: ScreeningDecision
    llm: ScreeningDecision
    route: Literal["auto", "humain"]
    escalation: str = ""                      # pourquoi l'humain est sollicité
    final: ScreeningDecision | None = None    # None : en attente d'un humain


class ScreeningState(BaseModel):
    protocol: Protocol
    threshold: float
    records: list[ScreeningRecord] = Field(default_factory=list)

    def human_queue(self) -> list[ScreeningRecord]:
        return [r for r in self.records if r.final is None]

    def counts(self) -> dict[str, int]:
        finals = [r.final.decision for r in self.records if r.final]
        return {
            "examinés": len(self.records),
            "inclus": finals.count("include"),
            "exclus": finals.count("exclude"),
            "en_attente_humain": len(self.human_queue()),
            "décidés_par_un_humain": sum(1 for r in self.records if r.final and r.final.decider == "humain"),
        }


def paper_text(paper: ScreeningPaper) -> str:
    """Code fourni : le texte examiné (titre + résumé)."""
    return paper.title + ". " + (paper.abstract or "")


def sentences(text: str) -> list[str]:
    """Code fourni : découpe en phrases (une preuve = une phrase)."""
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+", text) if s.strip()]


def normalize(text: str) -> str:
    """Code fourni : minuscules et espaces compactés (comparaison robuste)."""
    return " ".join(text.lower().split())


DEFAULT_EXCLUSION = [
    Criterion(id="E1", label="domaine clinique", terms=["clinical trial", "patients", "medical"]),
    Criterion(id="E2", label="publication non éligible", terms=["editorial", "erratum", "keynote"]),
]


# ===========================================================================
# Partie A - Du protocole aux preuves
# ===========================================================================

def protocol_from_concepts(
    question: str,
    concepts: list[ConceptGroup],
    exclusion: list[Criterion] | None = None,
) -> Protocol:
    """TODO 1 - Pont TD02 : chaque concept validé devient un critère d'inclusion."""
    # TODO 1 - Pour chaque concept (numéroté à partir de 1), créer
    #          Criterion(id=f"C{numéro}", label=group.label, terms=list(group.terms)).
    #          Indice : enumerate(concepts, start=1).
    raise NotImplementedError("TODO 1 - protocol_from_concepts")
    return Protocol(
        question=question,
        inclusion=inclusion,
        exclusion=list(DEFAULT_EXCLUSION if exclusion is None else exclusion),
    )


def find_evidence(text: str, criterion: Criterion) -> Evidence | None:
    """TODO 2 - Chercher une preuve d'un critère : la première phrase qui contient un de ses termes.

    Un terme est un MOT ENTIER, sans tenir compte de la casse, pluriel en « s »
    accepté : « LLM » trouve « LLMs », mais pas « LLMOps ».
    """
    for sentence in sentences(text):
        for term in criterion.terms:
            pattern = r"\b" + re.escape(term) + r"s?\b"
            # TODO 2 - Si re.search(pattern, sentence, flags=re.IGNORECASE) trouve le terme :
            #          retourner Evidence(criterion=criterion.id, quote=sentence).
            raise NotImplementedError("TODO 2 - find_evidence")
    return None


def rule_screen(paper: ScreeningPaper, protocol: Protocol) -> ScreeningDecision:
    """TODO 3 - Premier évaluateur : les règles du protocole, preuves à l'appui."""
    if not paper.abstract:
        return ScreeningDecision(paper_id=paper.id, decision="uncertain", confidence=0.0,
                                 reasons=["résumé absent : information insuffisante"], decider="regles")
    text = paper_text(paper)
    excluded = [e for c in protocol.exclusion if (e := find_evidence(text, c))]
    found = {c.id: find_evidence(text, c) for c in protocol.inclusion}
    matched = [e for e in found.values() if e]
    missing = [cid for cid, e in found.items() if e is None]
    # TODO 3 - Quatre cas, dans cet ordre (decider="regles") :
    #  1. excluded non vide -> "exclude", confiance 0.9, evidence=excluded,
    #     reasons=[f"critère d'exclusion {e.criterion} présent" for e in excluded]
    #  2. missing vide      -> "include", confiance 0.9, evidence=matched,
    #     reasons=["tous les critères d'inclusion sont présents"]
    #  3. matched vide      -> "exclude", confiance 0.85,
    #     reasons=["aucun critère d'inclusion présent"]
    #  4. sinon             -> "uncertain", confiance 0.5, evidence=matched,
    #     reasons=[f"critère absent : {cid}" for cid in missing]
    raise NotImplementedError("TODO 3 - rule_screen")


# ===========================================================================
# Partie B - Le LLM, second évaluateur contrôlé
# ===========================================================================

def build_screening_prompt(paper: ScreeningPaper, protocol: Protocol) -> str:
    """Code fourni : prompt du LLM évaluateur."""
    def describe(criteria):
        return "\n".join(f"- {c.id} · {c.label} : {', '.join(c.terms)}" for c in criteria) or "- (aucun)"
    return f"""Tu es relecteur pour une revue systématique de la littérature (screening titre/résumé).

[QUESTION DE RECHERCHE]
{protocol.question}

[CRITÈRES D'INCLUSION] (tous requis)
{describe(protocol.inclusion)}

[CRITÈRES D'EXCLUSION] (un seul suffit)
{describe(protocol.exclusion)}

[ARTICLE]
id : {paper.doi or paper.id}
titre : {paper.title}
résumé : {paper.abstract or "(indisponible)"}

[RÈGLES]
- Réponds uniquement par un objet JSON :
  {{"decision": "include|exclude|uncertain", "confidence": 0.0-1.0,
    "reasons": ["..."], "evidence": [{{"criterion": "C1", "quote": "extrait EXACT du titre ou du résumé"}}]}}
- Chaque citation doit être recopiée mot pour mot depuis le titre ou le résumé.
- Si l'information est insuffisante, réponds "uncertain" : ne devine pas.
- N'appelle aucune fonction : écris l'objet JSON comme simple texte.

[SCREENING]
"""


def parse_judgment(text: str, paper_id: str) -> ScreeningDecision:
    """TODO 4 - Sortie du LLM -> ScreeningDecision. Hors contrat : abstention, pas plantage."""
    # TODO 4.1 - Dans un try : judgment = LLMJudgment.model_validate_json(extract_json_object(text))
    # TODO 4.2 - except ValueError as error : retourner une décision "uncertain",
    #            confiance 0.0, reasons=[f"sortie LLM hors contrat : {...}"], decider="llm".
    # TODO 4.3 - Sinon : ScreeningDecision(paper_id=paper_id, decider="llm", **judgment.model_dump())
    raise NotImplementedError("TODO 4 - parse_judgment")


def verify_evidence(decision: ScreeningDecision, paper: ScreeningPaper) -> ScreeningDecision:
    """TODO 5 - Ne croire que les preuves vérifiables dans le titre ou le résumé.

    1. Une citation introuvable dans le texte est retirée.
    2. Une seule citation inventée suffit : la décision devient "uncertain".
    3. Une inclusion sans aucune preuve devient "uncertain".
    """
    text = normalize(paper_text(paper))
    checked = decision.model_copy(deep=True)
    # TODO 5.1 - kept = preuves dont normalize(e.quote) est dans text ;
    #            invented = citations (e.quote) qui n'y sont pas. checked.evidence = kept.
    # TODO 5.2 - Si invented n'est pas vide : decision "uncertain",
    #            confidence = min(confidence, 0.5),
    #            ajouter la raison f"citation introuvable : « {invented[0]} »".
    # TODO 5.3 - Sinon, si c'est une inclusion sans preuve gardée : même rétrogradation,
    #            raison "inclusion sans preuve".
    # Retourner checked (une copie : la décision reçue n'est pas modifiée).
    raise NotImplementedError("TODO 5 - verify_evidence")


def llm_screen(model, paper: ScreeningPaper, protocol: Protocol) -> ScreeningDecision:
    """Code fourni : le LLM propose un jugement ; le logiciel le contrôle."""
    try:
        text = call_llm(model, build_screening_prompt(paper, protocol))
    except Exception as error:                     # réseau, clé, quota : le LLM se tait
        return ScreeningDecision(paper_id=paper.id, decision="uncertain", confidence=0.0,
                                 reasons=[f"LLM indisponible : {error.__class__.__name__}"], decider="llm")
    return verify_evidence(parse_judgment(text, paper.id), paper)


# ===========================================================================
# Partie C - Décider, s'abstenir, escalader
# ===========================================================================

def route(decision: ScreeningDecision, threshold: float = DEFAULT_THRESHOLD) -> Literal["auto", "humain"]:
    """TODO 6 - Agir ou s'abstenir : la décision est-elle assez sûre pour être automatique ?"""
    # TODO 6 - Règle 1 : "uncertain" -> "humain" (abstention).
    # TODO 6 - Règle 2 : "include" -> "auto" si confidence >= threshold, sinon "humain".
    # TODO 6 - Règle 3 : "exclude" -> "auto" si confidence >= min(1.0, threshold + EXCLUDE_MARGIN),
    #          sinon "humain" (exclure à tort coûte plus cher qu'inclure à tort).
    raise NotImplementedError("TODO 6 - route")


def escalation_reason(rules: ScreeningDecision, llm: ScreeningDecision, threshold: float) -> str:
    """Code fourni : pourquoi l'humain est sollicité."""
    if rules.decision != llm.decision:
        return f"désaccord : {rules.decision} (règles) / {llm.decision} (LLM)"
    if rules.decision == "uncertain":
        return "abstention des deux évaluateurs"
    return f"confiance insuffisante (seuil {threshold})"


def combine(paper: ScreeningPaper, rules: ScreeningDecision, llm: ScreeningDecision,
            threshold: float = DEFAULT_THRESHOLD) -> ScreeningRecord:
    """TODO 7 - Double évaluation : agir seulement si les deux avis concordent et sont sûrs."""
    # TODO 7.1 - Si rules.decision == llm.decision ET route(rules, threshold) == "auto"
    #            ET route(llm, threshold) == "auto" : décision finale automatique
    #            ScreeningDecision(paper_id=paper.id, decision=rules.decision,
    #              confidence=min(rules.confidence, llm.confidence),
    #              reasons=["accord des règles et du LLM"],
    #              evidence=rules.evidence + llm.evidence, decider="accord"),
    #            puis retourner ScreeningRecord(..., route="auto", final=final).
    # TODO 7.2 - Sinon : ScreeningRecord(..., route="humain",
    #            escalation=escalation_reason(rules, llm, threshold)), sans final.
    raise NotImplementedError("TODO 7 - combine")


def screen_papers(papers: list[ScreeningPaper], protocol: Protocol, model,
                  threshold: float = DEFAULT_THRESHOLD) -> ScreeningState:
    """Code fourni : deux avis par article, puis la politique de décision."""
    state = ScreeningState(protocol=protocol, threshold=threshold)
    for paper in papers:
        state.records.append(combine(paper, rule_screen(paper, protocol),
                                     llm_screen(model, paper, protocol), threshold))
    return state


def evaluate(state: ScreeningState, gold: dict[str, str]) -> dict:
    """TODO 8 - Mesurer les erreurs des décisions AUTOMATIQUES par rapport aux étiquettes humaines."""
    auto = [r for r in state.records if r.route == "auto" and r.final is not None]
    # TODO 8 - false_exclusions : décisions auto "exclude" alors que gold dit "include"
    #          (une étude pertinente PERDUE) ;
    #          false_inclusions : décisions auto "include" alors que gold dit "exclude".
    raise NotImplementedError("TODO 8 - evaluate")
    total = len(state.records)
    human = total - len(auto)
    return {
        "articles": total,
        "auto": len(auto),
        "humain": human,
        "charge_humaine": round(human / total, 2) if total else 0.0,
        "faux_exclus": false_exclusions,
        "faux_inclus": false_inclusions,
        "erreurs_auto": false_exclusions + false_inclusions,
    }


# ===========================================================================
# Partie D - Boucler avec l'humain
# ===========================================================================

def apply_human_review(state: ScreeningState, reviews: dict[str, str]) -> int:
    """TODO 9 - Intégrer les décisions humaines de la file d'attente.

    reviews : {paper_id: "include" | "exclude"}. Retourne le nombre de décisions appliquées.
    Une revue ne remplace jamais une décision déjà prise ; "uncertain" n'est pas une décision.
    """
    queue = {r.paper_id: r for r in state.human_queue()}
    # TODO 9.1 - Vérifier TOUTES les revues avant d'en appliquer une seule :
    #   paper_id absent de queue -> ValueError ; choix hors {"include", "exclude"} -> ValueError.
    # TODO 9.2 - Puis, pour chaque revue : queue[paper_id].final = ScreeningDecision(
    #   paper_id=..., decision=choice, confidence=1.0,
    #   reasons=[f"décision humaine ({queue[paper_id].escalation})"], decider="humain").
    # TODO 9.3 - Retourner le nombre de décisions appliquées.
    return 0


def rescreen(state: ScreeningState, threshold: float) -> ScreeningState:
    """Code fourni : rejouer la politique avec un autre seuil, sans rappeler le LLM."""
    papers = {r.paper_id: ScreeningPaper(id=r.paper_id, title=r.title) for r in state.records}
    new = ScreeningState(protocol=state.protocol, threshold=threshold)
    for r in state.records:
        new.records.append(combine(papers[r.paper_id], r.rules, r.llm, threshold))
    return new


def sweep_thresholds(state: ScreeningState, gold: dict[str, str],
                     thresholds: tuple[float, ...] = THRESHOLDS) -> list[dict]:
    """Code fourni : une ligne d'évaluation par seuil."""
    return [{"seuil": t, **evaluate(rescreen(state, t), gold)} for t in thresholds]


def choose_threshold(rows: list[dict]) -> float | None:
    """TODO 10 - Le seuil qui minimise la charge humaine SANS aucune fausse exclusion.

    À charge égale, garder le seuil le plus élevé (le plus prudent).
    Aucun seuil sûr : None (tout reste à l'humain).
    """
    # TODO 10 - Garder les lignes sans fausse exclusion ; aucune -> None.
    #           Parmi elles, choisir la plus petite charge "humain" ;
    #           à égalité, le "seuil" le plus élevé. Retourner ce seuil.
    return None


def automation_only(state: ScreeningState, evaluator: Literal["regles", "llm"]) -> ScreeningState:
    """Code fourni : référence « sans humain » - un seul évaluateur décide de tout.

    Sans humain, un article incertain n'est pas retenu : il devient une exclusion.
    """
    new = ScreeningState(protocol=state.protocol, threshold=0.0)
    for r in state.records:
        opinion = r.rules if evaluator == "regles" else r.llm
        decision = "exclude" if opinion.decision == "uncertain" else opinion.decision
        final = opinion.model_copy(update={"decision": decision})
        new.records.append(ScreeningRecord(paper_id=r.paper_id, title=r.title, rules=r.rules,
                                           llm=r.llm, route="auto", final=final))
    return new


def compare_policies(state: ScreeningState, gold: dict[str, str]) -> list[dict]:
    """Code fourni : automatisation complète vs double évaluation avec humain."""
    return [
        {"politique": "règles seules, sans humain", **evaluate(automation_only(state, "regles"), gold)},
        {"politique": "LLM seul, sans humain", **evaluate(automation_only(state, "llm"), gold)},
        {"politique": f"double évaluation + humain (seuil {state.threshold})", **evaluate(state, gold)},
    ]
