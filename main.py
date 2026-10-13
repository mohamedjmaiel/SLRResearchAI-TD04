"""TD04 - ScreeningAgent : décider, s'abstenir, escalader (ligne de commande).

    python main.py                                       # mode guidé (questions)
    python main.py --llm demo --etape jeu                # mesurer la politique sur le jeu étiqueté
    python main.py --llm demo --accepter --rq RQ1        # sujet -> TD02 -> TD03 -> screening

Chaîne exécutée :
    étape 1 (jeu)   : jeu étiqueté -> deux évaluateurs -> politique -> erreurs mesurées
    étape 2 (sujet) : sujet -> TD02 (RQ, concepts validés = critères) -> TD03 (articles)
                      -> résumés -> screening -> file humaine -> revue humaine
"""
import argparse

from slrresearch.abstracts import fetch_abstracts, gold_cases, gold_concepts, load_gold
from slrresearch.config import Settings, build_model, resolve_decoding
from slrresearch.demo_llm import DemoChatModel
from slrresearch.models import short_error
from slrresearch.screening_agent import (
    DEFAULT_EXCLUSION,
    DEFAULT_THRESHOLD,
    Criterion,
    apply_human_review,
    choose_threshold,
    compare_policies,
    protocol_from_concepts,
    screen_papers,
    sweep_thresholds,
)
from slrresearch.search_agent import run_level, state_summary
from slrresearch.search_plan import (
    ConceptGroup,
    ConceptProposal,
    SearchConcepts,
    accept_proposal,
    apply_human_review as review_concepts,
    build_queries,
    format_concepts_text,
    parse_concepts_text,
)
from slrresearch.td02_bridge import (
    CONCEPT_MAX_OUTPUT_TOKENS,
    REFERENCE_TOPIC,
    generate_questions_for_topic,
    propose_concepts_for_topic,
)
from slrresearch.tools import build_registry, scenario_names

SCREENING_MAX_OUTPUT_TOKENS = 600
SHORT = {"include": "INCLURE", "exclude": "EXCLURE", "uncertain": "INCERTAIN"}


def parse_args():
    parser = argparse.ArgumentParser(description="TD04 - ScreeningAgent")
    parser.add_argument("--llm", choices=["demo", "reel"], help="LLM de démonstration ou .env")
    parser.add_argument("--sources", choices=["simule", "reel"], help="catalogues simulés ou API réelles")
    parser.add_argument("--scenario", choices=scenario_names(), help="scénario de recherche (TD03)")
    parser.add_argument("--niveau", type=int, choices=[1, 2, 3], default=2, help="niveau du SearchAgent (TD03)")
    parser.add_argument("--etape", choices=["jeu", "sujet", "tout"], help="jeu étiqueté, sujet, ou les deux")
    parser.add_argument("--seuil", type=float, help=f"seuil de confiance (défaut {DEFAULT_THRESHOLD})")
    parser.add_argument("--sujet", help="sujet de la revue")
    parser.add_argument("--rq", help="RQ à examiner (ex. RQ1)")
    parser.add_argument("--accepter", action="store_true", help="accepter les concepts proposés (TD02)")
    parser.add_argument("--sans-revue", action="store_true", help="ne pas trancher la file humaine maintenant")
    return parser.parse_args()


def read(prompt: str, default: str = "") -> str:
    """input() avec valeur par défaut (Entrée, ou entrée standard fermée)."""
    try:
        return input(prompt).strip() or default
    except EOFError:
        print()
        return default


def ask(prompt: str, choices: list[str], default: str) -> str:
    while True:
        answer = read(f"{prompt} {choices} [{default}] : ", default)
        if answer in choices:
            return answer
        print("  Choix invalide.")


def choose(guided: bool, given, prompt: str, choices: list[str], default: str) -> str:
    if given:
        return given
    return ask(prompt, choices, default) if guided else default


def yes(prompt: str, default: str = "o") -> bool:
    return read(f"{prompt} [{'O/n' if default == 'o' else 'o/N'}] : ", default).lower() in ("o", "oui", "y")


def short_id(paper_id: str) -> str:
    last = paper_id.rstrip("/").rsplit("/", 1)[-1]
    if paper_id.startswith("http"):
        return last
    if "/" in paper_id:
        return paper_id.split(":", 1)[0] + ":" + last
    return paper_id[:24]


def build_models(llm_mode: str):
    """(modèle RQ, modèle concepts, modèle évaluateur)."""
    if llm_mode == "demo":
        model = DemoChatModel()
        return model, model, model
    settings = Settings()
    name = settings.ollama_model if settings.provider == "ollama" else settings.groq_model
    print(f"LLM réel : {settings.provider} | {name}")
    return (
        build_model(settings, resolve_decoding("greedy")),
        build_model(settings, resolve_decoding("greedy", max_output_tokens=CONCEPT_MAX_OUTPUT_TOKENS)),
        build_model(settings, resolve_decoding("greedy", max_output_tokens=SCREENING_MAX_OUTPUT_TOKENS)),
    )


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

def opinion(decision) -> str:
    return f"{SHORT[decision.decision]:9} {decision.confidence:.2f}"


def print_protocol(protocol) -> None:
    print(f"\n=== Protocole : {protocol.question} ===")
    for c in protocol.inclusion:
        print(f"  inclusion {c.id} · {c.label:24}: {', '.join(c.terms)}")
    for c in protocol.exclusion:
        print(f"  exclusion {c.id} · {c.label:24}: {', '.join(c.terms)}")


def print_records(state, gold=None, cases=None) -> None:
    head = f"  {'article':24} | {'règles':14} | {'LLM':14} | issue"
    print(head + (" | étiquette humaine" if gold else ""))
    for r in state.records:
        issue = (f"AUTO → {SHORT[r.final.decision]}" if r.route == "auto"
                 else f"HUMAIN ({r.escalation})")
        line = f"  {short_id(r.paper_id):24} | {opinion(r.rules):14} | {opinion(r.llm):14} | {issue}"
        if gold:
            line += f" | {SHORT[gold[r.paper_id]]}"
        print(line)
        if cases:
            print(f"  {'':24}   ↳ {cases[r.paper_id]}")


def print_table(rows: list[dict], first: str) -> None:
    keys = [first, "auto", "humain", "charge_humaine", "faux_exclus", "faux_inclus"]
    width = max(len(first), *(len(str(row[first])) for row in rows))
    print("  " + " | ".join(f"{k:>14}" if k != first else f"{k:{width}}" for k in keys))
    for row in rows:
        print("  " + " | ".join(f"{str(row[k]):>14}" if k != first else f"{str(row[k]):{width}}" for k in keys))


# ---------------------------------------------------------------------------
# Étape 1 : mesurer la politique sur le jeu étiqueté
# ---------------------------------------------------------------------------

def run_gold(model, threshold: float) -> None:
    question, papers, labels = load_gold()
    protocol = protocol_from_concepts(question, [ConceptGroup(**c) for c in gold_concepts()])
    print_protocol(protocol)
    state = screen_papers(papers, protocol, model, threshold)
    print(f"\n=== Étape 1 · jeu étiqueté ({len(papers)} articles, seuil {threshold}) ===")
    print_records(state, labels, gold_cases())
    print("\n=== Automatisation complète ou humain dans la boucle ? ===")
    print_table(compare_policies(state, labels), "politique")
    print("\n=== Effet du seuil de confiance ===")
    print_table(sweep_thresholds(state, labels), "seuil")
    best = choose_threshold(sweep_thresholds(state, labels))
    print(f"  Seuil recommandé (aucune fausse exclusion, charge minimale) : {best if best is not None else 'aucun'}")


# ---------------------------------------------------------------------------
# Étape 2 : du sujet aux décisions
# ---------------------------------------------------------------------------

def review_proposal(proposal: ConceptProposal) -> SearchConcepts:
    """Revue humaine du TD02 : les concepts validés deviendront les critères d'inclusion."""
    print(f"\n--- {proposal.identifier} · {proposal.question}")
    if proposal.extraction:
        print(format_concepts_text(proposal.extraction.concepts))
        if yes("Accepter ces concepts ?"):
            return accept_proposal(proposal)
    while True:
        print("Saisissez un concept par ligne « concept: terme1; terme2 » (ligne vide = fin).")
        lines = []
        while line := read("  > "):
            lines.append(line)
        if not lines and proposal.extraction:
            print("Aucune saisie : les concepts proposés sont conservés.")
            return accept_proposal(proposal)
        try:
            return review_concepts(proposal, parse_concepts_text("\n".join(lines)))
        except ValueError as error:
            print(f"Saisie refusée par le contrat Pydantic : {error}")


def ask_exclusions(guided: bool) -> list[Criterion]:
    print("\nCritères d'exclusion par défaut :")
    for c in DEFAULT_EXCLUSION:
        print(f"  {c.id} · {c.label}: {'; '.join(c.terms)}")
    if not guided or yes("Les conserver ?"):
        return list(DEFAULT_EXCLUSION)
    print("Un critère par ligne « libellé: terme1; terme2 » (ligne vide = aucun).")
    lines = []
    while line := read("  > "):
        lines.append(line)
    try:
        groups = parse_concepts_text("\n".join(lines)) if lines else []
    except ValueError as error:
        print(f"Saisie refusée ({error}) : critères par défaut conservés.")
        return list(DEFAULT_EXCLUSION)
    return [Criterion(id=f"E{i}", label=g["label"], terms=g["terms"]) for i, g in enumerate(groups, start=1)]


def human_review(state, interactive: bool) -> dict[str, str]:
    """Point d'arbitrage humain : trancher (ou reporter) chaque article de la file."""
    queue = state.human_queue()
    if not queue or not interactive:
        return {}
    print(f"\n=== Revue humaine : {len(queue)} article(s) en attente ===")
    reviews = {}
    for r in queue:
        print(f"\n  {short_id(r.paper_id)} · {r.title}")
        print(f"    motif : {r.escalation}")
        print(f"    règles : {opinion(r.rules)} · {'; '.join(r.rules.reasons)}")
        print(f"    LLM    : {opinion(r.llm)} · {'; '.join(r.llm.reasons)}")
        answer = read("    [i]nclure, [e]xclure, Entrée = plus tard : ", "").lower()
        if answer in ("i", "include", "inclure"):
            reviews[r.paper_id] = "include"
        elif answer in ("e", "exclude", "exclure"):
            reviews[r.paper_id] = "exclude"
    return reviews


def run_topic(args, guided, llm_mode, sources, scenario, threshold) -> None:
    model, concept_model, judge = build_models(llm_mode)
    topic = args.sujet or (read(f"Sujet de la revue [{REFERENCE_TOPIC}] : ", REFERENCE_TOPIC)
                           if guided else REFERENCE_TOPIC)
    print(f"Sujet : {topic}")
    if llm_mode == "demo" and topic != REFERENCE_TOPIC:
        print("Attention : le LLM de démonstration simule les réponses du sujet de référence.")
    try:
        outcome = generate_questions_for_topic(model, topic)
        print(f"\n=== TD02 · RQ ({outcome.status}) ===")
        if outcome.status == "human_required":
            print(f"Arrêt : validation humaine requise ({outcome.error}).")
            return
        for question in outcome.questions:
            print(f"  {question.identifier}. {question.question}")
        proposals = propose_concepts_for_topic(concept_model, outcome, topic)
    except Exception as error:  # clé absente, réseau, quota...
        print(f"Appel LLM impossible : {short_error(error)}")
        return
    validated = [accept_proposal(p) if args.accepter and p.extraction else review_proposal(p)
                 for p in proposals]
    plans = build_queries(validated)
    rq = choose(guided, args.rq, "RQ à examiner", list(plans), "RQ1")
    concepts = next(item for item in validated if item.identifier == rq)

    try:
        protocol = protocol_from_concepts(concepts.question, concepts.concepts, ask_exclusions(guided))
    except NotImplementedError as error:
        print(f"\nNon disponible : {error}")
        return
    print_protocol(protocol)

    try:
        search = run_level(args.niveau, plans[rq], build_registry(sources, scenario), model=judge,
                           objective=f"{rq} · {concepts.question}")
    except Exception as error:
        print(f"Recherche impossible : {short_error(error)}")
        return
    summary = state_summary(search)
    print(f"\n=== TD03 · {summary['articles_uniques']} article(s) unique(s) ({summary['statut']}) ===")
    papers = fetch_abstracts(search.papers, sources)
    for paper in papers:
        print(f"  - {short_id(paper.id):24} | {paper.title[:60]:60} | "
              f"{'résumé disponible' if paper.abstract else 'résumé absent'}")
    if not papers:
        print("Aucun article à examiner.")
        return

    try:
        state = screen_papers(papers, protocol, judge, threshold)
    except NotImplementedError as error:
        print(f"\nNon disponible : {error}")
        return
    print(f"\n=== TD04 · screening titre/résumé (seuil {threshold}) ===")
    print_records(state)

    reviews = human_review(state, interactive=not args.sans_revue)
    applied = apply_human_review(state, reviews) if reviews else 0
    print(f"\n=== Bilan : {applied} décision(s) humaine(s) appliquée(s) ===")
    for key, value in state.counts().items():
        print(f"  {key:22}: {value}")
    for r in state.records:
        if r.final:
            print(f"  {SHORT[r.final.decision]:9} · {r.final.decider:7} · {short_id(r.paper_id)}")
        else:
            print(f"  {'ATTENTE':9} · {'-':7} · {short_id(r.paper_id)} ({r.escalation})")


def main():
    args = parse_args()
    # Mode guidé : aucune option de choix fournie. Sinon, les choix non précisés
    # prennent leur valeur par défaut ; les décisions humaines restent interactives.
    guided = not any([args.llm, args.sources, args.scenario, args.etape, args.sujet, args.rq, args.seuil])
    llm_mode = choose(guided, args.llm, "LLM", ["demo", "reel"], "demo")
    step = choose(guided, args.etape, "Étape", ["jeu", "sujet", "tout"], "tout")
    threshold = args.seuil if args.seuil is not None else DEFAULT_THRESHOLD
    if step in ("jeu", "tout"):
        try:
            run_gold(build_models(llm_mode)[2], threshold)
        except NotImplementedError as error:
            print(f"\nNon disponible : {error}")
    if step in ("sujet", "tout"):
        sources = choose(guided, args.sources, "Sources (recherche et résumés)", ["simule", "reel"], "simule")
        scenario = "perturbe"
        if sources == "simule":
            scenario = choose(guided, args.scenario, "Scénario de recherche (TD03)", scenario_names(), "perturbe")
        run_topic(args, guided, llm_mode, sources, scenario, threshold)


if __name__ == "__main__":
    main()
