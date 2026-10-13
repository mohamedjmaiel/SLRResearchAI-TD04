"""TD04 - Interface pédagogique : décider, s'abstenir, escalader.

    python -m streamlit run streamlit_app.py
"""
import streamlit as st

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

LABEL = {"include": "inclure", "exclude": "exclure", "uncertain": "incertain"}
ICON = {"include": "✅", "exclude": "⛔", "uncertain": "❔"}


def short_id(paper_id: str) -> str:
    last = paper_id.rstrip("/").rsplit("/", 1)[-1]
    if paper_id.startswith("http"):
        return last
    if "/" in paper_id:
        return paper_id.split(":", 1)[0] + ":" + last
    return paper_id[:24]


def opinion(decision) -> str:
    return f"{ICON[decision.decision]} {LABEL[decision.decision]} · {decision.confidence:.2f}"


def issue(record) -> str:
    if record.final is not None:
        who = "humain" if record.final.decider == "humain" else "auto"
        return f"{ICON[record.final.decision]} {LABEL[record.final.decision]} ({who})"
    return f"🙋 humain : {record.escalation}"


st.set_page_config(page_title="SLRResearchAI · TD04", page_icon=":material/rule:", layout="wide")
st.title("TD04 · ScreeningAgent : décider, s'abstenir, escalader")
st.code(
    "article (titre + résumé) → deux évaluateurs : règles du protocole · LLM contrôlé (preuves vérifiées)\n"
    "        → politique : agir (inclure / exclure)  ·  s'abstenir  ·  escalader vers un humain\n"
    "        → erreurs mesurées sur un jeu étiqueté, puis application aux articles de votre sujet",
    language="text",
)

# ---------------------------------------------------------------- réglages
with st.sidebar:
    st.header("Réglages")
    llm_mode = st.radio("LLM", ["Démonstration (sans clé)", "Réel (.env)"], key="llm_mode")
    threshold = st.slider("Seuil de confiance", 0.5, 0.95, DEFAULT_THRESHOLD, 0.05, key="threshold",
                          help="Une exclusion automatique exige seuil + 0.10.")
    sources = st.radio("Sources (recherche et résumés)", ["Catalogues simulés", "API réelles"], key="sources")
    simulated = sources == "Catalogues simulés"
    scenario = st.selectbox("Scénario de recherche (TD03)", scenario_names(),
                            index=scenario_names().index("perturbe"), disabled=not simulated, key="scenario")
    level = st.radio("SearchAgent du TD03", [1, 2, 3], index=1, horizontal=True, key="level",
                     format_func=lambda n: f"niveau {n}")
SOURCES = "simule" if simulated else "reel"


def get_models():
    """(modèle RQ, modèle concepts, modèle évaluateur)."""
    if llm_mode.startswith("Démonstration"):
        model = DemoChatModel()
        return model, model, model
    settings = Settings()
    return (
        build_model(settings, resolve_decoding("greedy")),
        build_model(settings, resolve_decoding("greedy", max_output_tokens=CONCEPT_MAX_OUTPUT_TOKENS)),
        build_model(settings, resolve_decoding("greedy", max_output_tokens=600)),
    )


def records_table(state, labels=None, cases=None):
    rows = []
    for r in state.records:
        row = {"article": short_id(r.paper_id), "titre": r.title, "règles": opinion(r.rules),
               "LLM": opinion(r.llm), "issue": issue(r)}
        if labels:
            row["étiquette humaine"] = LABEL[labels[r.paper_id]]
        if cases:
            row["ce que l'article montre"] = cases[r.paper_id]
        rows.append(row)
    st.dataframe(rows, hide_index=True, width="stretch")


def metrics_table(rows, first):
    st.dataframe([{first: row[first], "auto": row["auto"], "humain": row["humain"],
                   "charge humaine": row["charge_humaine"], "faux exclus": row["faux_exclus"],
                   "faux inclus": row["faux_inclus"]} for row in rows], hide_index=True, width="stretch")


for key in ("gold", "outcome", "proposals", "plans", "protocol", "papers", "state", "message"):
    st.session_state.setdefault(f"td04_{key}", None)
st.session_state.setdefault("td04_run", 0)

tab_gold, tab_topic = st.tabs(["1 · Mesurer la politique (jeu étiqueté)", "2 · Du sujet aux décisions"])

# ---------------------------------------------------------------- étape 1 : jeu étiqueté
with tab_gold:
    st.caption("15 articles fictifs, étiquetés par deux relecteurs humains. On mesure les erreurs des "
               "décisions automatiques AVANT d'appliquer la politique à de vrais articles.")
    if st.button("Évaluer la politique sur le jeu étiqueté", type="primary", icon=":material/fact_check:"):
        try:
            question, papers, labels = load_gold()
            protocol = protocol_from_concepts(question, [ConceptGroup(**c) for c in gold_concepts()])
            state = screen_papers(papers, protocol, get_models()[2], threshold)
            st.session_state.td04_gold = {"state": state, "labels": labels,
                                          "policies": compare_policies(state, labels),
                                          "sweep": sweep_thresholds(state, labels)}
        except NotImplementedError as error:
            st.session_state.td04_gold = None
            st.warning(f"Non disponible : {error}")
        except Exception as error:
            st.session_state.td04_gold = None
            st.error(f"Évaluation impossible : {short_error(error)}")
    gold = st.session_state.td04_gold
    if gold:
        state = gold["state"]
        own = next(row for row in gold["policies"] if row["politique"].startswith("double"))
        cols = st.columns(4)
        cols[0].metric("Décisions automatiques", own["auto"])
        cols[1].metric("File humaine", own["humain"], help="abstentions et escalades")
        cols[2].metric("Fausses exclusions", own["faux_exclus"], help="études pertinentes perdues")
        cols[3].metric("Fausses inclusions", own["faux_inclus"])
        records_table(state, gold["labels"], gold_cases())
        st.subheader("Automatisation complète ou humain dans la boucle ?")
        metrics_table(gold["policies"], "politique")
        st.subheader(f"Effet du seuil (seuil courant : {state.threshold})")
        metrics_table(gold["sweep"], "seuil")
        best = choose_threshold(gold["sweep"])
        st.info(f"Seuil recommandé (aucune fausse exclusion, charge humaine minimale) : "
                f"**{best if best is not None else 'aucun'}**")

# ---------------------------------------------------------------- étape 2 : sujet
with tab_topic:
    topic = st.text_input("Sujet de la revue", REFERENCE_TOPIC, key="topic")
    if llm_mode.startswith("Démonstration") and topic.strip() != REFERENCE_TOPIC:
        st.warning("Le LLM de démonstration simule les réponses du sujet de référence. "
                   "Choisissez « Réel (.env) » pour traiter votre propre sujet.")
    if st.button("1 · Générer les RQ et proposer les concepts (TD02)", type="primary",
                 icon=":material/psychology:"):
        try:
            model, concept_model, _ = get_models()
            outcome = generate_questions_for_topic(model, topic)
            st.session_state.td04_outcome = outcome
            st.session_state.td04_proposals = (None if outcome.status == "human_required"
                                               else propose_concepts_for_topic(concept_model, outcome, topic))
            for key in ("plans", "protocol", "papers", "state", "message"):
                st.session_state[f"td04_{key}"] = None
            st.session_state.td04_run += 1
        except Exception as error:
            st.error(f"Génération impossible : {short_error(error)}")

    outcome = st.session_state.td04_outcome
    if outcome is not None and outcome.status == "human_required":
        st.error(f"Validation humaine requise : {outcome.error}")
    proposals = st.session_state.td04_proposals
    if proposals:
        st.caption("Revue humaine des concepts : chaque concept validé deviendra un critère d'inclusion.")
        columns = st.columns(len(proposals))
        for column, proposal in zip(columns, proposals):
            with column:
                default = format_concepts_text(proposal.extraction.concepts) if proposal.extraction else ""
                st.text_area(f"{proposal.identifier} · {proposal.question}", default, height=130,
                             key=f"concepts_{st.session_state.td04_run}_{proposal.identifier}")
        if st.button("2 · Valider les concepts", icon=":material/how_to_reg:"):
            validated, errors = [], []
            for proposal in proposals:
                text = st.session_state[f"concepts_{st.session_state.td04_run}_{proposal.identifier}"]
                try:
                    validated.append(review_concepts(proposal, parse_concepts_text(text)))
                except ValueError as error:
                    errors.append(f"{proposal.identifier} : {short_error(error)}")
            for error in errors:
                st.error("Contrat Pydantic : " + error)
            if not errors:
                st.session_state.td04_plans = {"queries": build_queries(validated),
                                               "concepts": {v.identifier: v for v in validated}}
                for key in ("protocol", "papers", "state", "message"):
                    st.session_state[f"td04_{key}"] = None

    plans = st.session_state.td04_plans
    if plans:
        st.subheader("Protocole de screening")
        rq = st.selectbox("RQ à examiner", list(plans["queries"]), key="rq")
        concepts = plans["concepts"][rq]
        st.write("Critères d'inclusion (tous requis) : " + " · ".join(
            f"**C{i}** {g.label}" for i, g in enumerate(concepts.concepts, start=1)))
        exclusions = st.text_area(
            "Critères d'exclusion (un par ligne, « libellé: terme1; terme2 »)",
            "\n".join(f"{c.label}: {'; '.join(c.terms)}" for c in DEFAULT_EXCLUSION), height=90,
            key=f"exclusions_{st.session_state.td04_run}")
        if st.button("3 · Rechercher (TD03) et trier les articles", type="primary", icon=":material/rule:"):
            try:
                groups = parse_concepts_text(exclusions) if exclusions.strip() else []
                exclusion = [Criterion(id=f"E{i}", label=g["label"], terms=g["terms"])
                             for i, g in enumerate(groups, start=1)]
                protocol = protocol_from_concepts(concepts.question, concepts.concepts, exclusion)
                _, _, judge = get_models()
                search = run_level(level, plans["queries"][rq], build_registry(SOURCES, scenario),
                                   model=judge, objective=f"{rq} · {concepts.question}")
                papers = fetch_abstracts(search.papers, SOURCES)
                st.session_state.td04_protocol = protocol
                st.session_state.td04_papers = {"papers": papers, "summary": state_summary(search)}
                st.session_state.td04_state = screen_papers(papers, protocol, judge, threshold)
                st.session_state.td04_message = None
            except NotImplementedError as error:
                st.warning(f"Non disponible : {error}")
            except ValueError as error:
                st.error(f"Critères d'exclusion refusés : {short_error(error)}")
            except Exception as error:
                st.error(f"Screening impossible : {short_error(error)}")

    state = st.session_state.td04_state
    if plans and state is not None:
        found = st.session_state.td04_papers
        st.caption(f"TD03 : {found['summary']['articles_uniques']} article(s) unique(s) · "
                   f"{sum(1 for p in found['papers'] if p.abstract)} résumé(s) disponible(s)")
        counts = state.counts()
        cols = st.columns(4)
        cols[0].metric("Inclus", counts["inclus"])
        cols[1].metric("Exclus", counts["exclus"])
        cols[2].metric("En attente d'un humain", counts["en_attente_humain"])
        cols[3].metric("Décidés par un humain", counts["décidés_par_un_humain"])
        records_table(state)

        queue = state.human_queue()
        abstracts = {p.id: p.abstract for p in found["papers"]}
        if queue:
            st.subheader("Revue humaine : trancher les cas escaladés")
            for r in queue:
                with st.expander(f"{short_id(r.paper_id)} · {r.title} — {r.escalation}"):
                    st.write(abstracts.get(r.paper_id) or "_Résumé indisponible._")
                    st.write(f"Règles : {opinion(r.rules)} — {'; '.join(r.rules.reasons)}")
                    st.write(f"LLM : {opinion(r.llm)} — {'; '.join(r.llm.reasons)}")
                    st.radio("Décision humaine", ["plus tard", "inclure", "exclure"], horizontal=True,
                             key=f"review_{st.session_state.td04_run}_{r.paper_id}")
            if st.button("4 · Appliquer les décisions humaines", icon=":material/gavel:"):
                reviews = {}
                for r in queue:
                    choice = st.session_state[f"review_{st.session_state.td04_run}_{r.paper_id}"]
                    if choice != "plus tard":
                        reviews[r.paper_id] = "include" if choice == "inclure" else "exclude"
                try:
                    applied = apply_human_review(state, reviews)
                    st.session_state.td04_message = f"{applied} décision(s) humaine(s) appliquée(s)."
                except ValueError as error:
                    st.session_state.td04_message = f"Revue refusée : {error}"
                st.rerun()
        if st.session_state.td04_message:
            st.info(st.session_state.td04_message)
