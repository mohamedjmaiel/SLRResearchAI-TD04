"""Tests du TD04 : le contrat exécutable de chaque TODO (ne pas modifier).

    python -m pytest -v -k "fourni"      code fourni (vert dès le départ)
    python -m pytest -v -k "todo_3_"     un TODO
"""
import httpx
import pytest

from slrresearch.abstracts import (
    ScreeningPaper,
    fetch_abstracts,
    gold_cases,
    load_gold,
    openalex_abstract,
    rebuild_abstract,
)
from slrresearch.demo_llm import DemoChatModel
from slrresearch.models import PaperRecord
from slrresearch.screening_agent import (
    EXCLUDE_MARGIN,
    Criterion,
    Evidence,
    ScreeningDecision,
    ScreeningRecord,
    ScreeningState,
    apply_human_review,
    automation_only,
    build_screening_prompt,
    choose_threshold,
    combine,
    compare_policies,
    escalation_reason,
    evaluate,
    find_evidence,
    llm_screen,
    parse_judgment,
    protocol_from_concepts,
    route,
    rule_screen,
    screen_papers,
    sweep_thresholds,
    verify_evidence,
)
from slrresearch.search_agent import run_level
from slrresearch.search_plan import ConceptGroup, accept_proposal
from slrresearch.td02_bridge import REFERENCE_TOPIC, prepare_queries
from slrresearch.tools import build_registry

from conftest import QUESTION, make_paper

QUERIES = {
    "scholar": '"large language models" OR LLM screening OR "study selection"',
    "ieee": '("All Metadata":"large language models" OR "All Metadata":LLM) AND ("All Metadata":screening)',
    "acm": 'Abstract:("large language models" OR LLM) AND Abstract:(screening)',
}


def decision(paper_id="P1", value="include", confidence=0.9, decider="regles", evidence=(), reasons=("r",)):
    return ScreeningDecision(paper_id=paper_id, decision=value, confidence=confidence, decider=decider,
                             reasons=list(reasons), evidence=list(evidence))


def gold_state(gold, protocol, model, threshold=0.75):
    papers, labels = gold
    return screen_papers(list(papers.values()), protocol, model, threshold), labels


# ===========================================================================
# Code fourni (vert dès le départ)
# ===========================================================================

def test_fourni_pont_td02_concepts_de_la_rq():
    _, validated, plans = prepare_queries(DemoChatModel(), REFERENCE_TOPIC, accept_proposal)
    assert [g.label for g in validated[0].concepts] == ["large language models", "screening", "recall"]
    assert set(plans["RQ1"]) == {"scholar", "ieee", "acm"}


def test_fourni_td03_puis_resumes_simules():
    search = run_level(2, QUERIES, build_registry("simule", "perturbe"))
    papers = fetch_abstracts(search.papers, "simule")
    by_id = {p.id: p for p in papers}
    assert len(papers) == 4
    assert by_id["https://openalex.org/W1001"].abstract.startswith("We evaluate large language models")
    assert by_id["s2-7f3a"].abstract is None


def test_fourni_resume_openalex_reconstruit():
    assert rebuild_abstract({"LLMs": [1], "Screening": [0], "help": [2]}) == "Screening LLMs help"
    assert rebuild_abstract(None) is None


def test_fourni_resume_openalex_transport_simule():
    def handler(request):
        if "10.1/ok" in str(request.url):
            return httpx.Response(200, json={"abstract_inverted_index": {"We": [0], "screen": [1]}})
        if "10.1/lent" in str(request.url):
            raise httpx.ConnectError("hors ligne", request=request)
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert openalex_abstract(PaperRecord(id="x", title="t", doi="10.1/ok"), client) == "We screen"
    assert openalex_abstract(PaperRecord(id="x", title="t", doi="10.1/absent"), client) is None
    assert openalex_abstract(PaperRecord(id="x", title="t", doi="10.1/lent"), client) is None
    assert openalex_abstract(PaperRecord(id="s2-1", title="t"), client) is None


def test_fourni_jeu_etiquete():
    question, papers, labels = load_gold()
    assert question == QUESTION and len(papers) == 15
    assert sorted(labels.values()).count("include") == 9
    assert set(gold_cases()) == set(labels)


def test_fourni_prompt_de_screening(protocol):
    prompt = build_screening_prompt(make_paper("G01", "LLM screening", "We screen."), protocol)
    for expected in ("id : G01", "C1 · large language models", "E2 · publication non éligible",
                     "mot pour mot", "[SCREENING]"):
        assert expected in prompt


def test_fourni_llm_de_demonstration_rejoue_les_jugements(protocol, demo_model, gold):
    papers, _ = gold
    assert '"decision": "include"' in demo_model.invoke(build_screening_prompt(papers["G01"], protocol)).content
    assert demo_model.invoke(build_screening_prompt(papers["G09"], protocol)).content.startswith("This paper")


def test_fourni_llm_indisponible_vaut_abstention(protocol):
    class Down:
        def invoke(self, prompt):
            raise ConnectionError("pas de réseau")

    result = llm_screen(Down(), make_paper(), protocol)
    assert (result.decision, result.confidence, result.decider) == ("uncertain", 0.0, "llm")


def test_fourni_resume_absent_abstention(protocol):
    result = rule_screen(make_paper(abstract=None), protocol)
    assert (result.decision, result.confidence) == ("uncertain", 0.0)


def test_fourni_raison_d_escalade():
    assert escalation_reason(decision(value="include"), decision(value="exclude"), 0.75).startswith("désaccord")
    assert escalation_reason(decision(value="uncertain"), decision(value="uncertain"), 0.75).startswith("abstention")
    assert "seuil 0.75" in escalation_reason(decision(), decision(), 0.75)


def test_fourni_etat_et_file_humaine(protocol):
    auto = ScreeningRecord(paper_id="A", title="a", rules=decision("A"), llm=decision("A"), route="auto",
                           final=decision("A", decider="accord"))
    waiting = ScreeningRecord(paper_id="B", title="b", rules=decision("B"), llm=decision("B", "exclude"),
                              route="humain", escalation="désaccord")
    state = ScreeningState(protocol=protocol, threshold=0.75, records=[auto, waiting])
    assert [r.paper_id for r in state.human_queue()] == ["B"]
    assert state.counts()["inclus"] == 1 and state.counts()["en_attente_humain"] == 1
    alone = automation_only(state, "llm")
    assert [r.final.decision for r in alone.records] == ["include", "exclude"]


# ===========================================================================
# Partie A - Du protocole aux preuves
# ===========================================================================

def test_todo_1_un_critere_par_concept():
    concepts = [ConceptGroup(label="large language models", terms=["large language models", "LLM"]),
                ConceptGroup(label="screening", terms=["screening"])]
    protocol = protocol_from_concepts(QUESTION, concepts)
    assert [(c.id, c.label) for c in protocol.inclusion] == [("C1", "large language models"), ("C2", "screening")]
    assert protocol.inclusion[0].terms == ["large language models", "LLM"]
    assert [c.id for c in protocol.exclusion] == ["E1", "E2"]


def test_todo_1_exclusions_choisies_par_l_humain():
    concepts = [ConceptGroup(label="screening", terms=["screening"])]
    custom = [Criterion(id="E1", label="langue", terms=["chinese"])]
    assert protocol_from_concepts(QUESTION, concepts, custom).exclusion == custom
    assert protocol_from_concepts(QUESTION, concepts, []).exclusion == []


def test_todo_1_depuis_la_revue_du_td02():
    _, validated, _ = prepare_queries(DemoChatModel(), REFERENCE_TOPIC, accept_proposal)
    protocol = protocol_from_concepts(validated[0].question, validated[0].concepts)
    assert [c.id for c in protocol.inclusion] == ["C1", "C2", "C3"]
    assert "sensitivity" in protocol.inclusion[2].terms


def test_todo_2_mot_entier_et_pluriel():
    llm = Criterion(id="C1", label="llm", terms=["LLM"])
    assert find_evidence("We compare LLMs on screening.", llm).quote == "We compare LLMs on screening."
    assert find_evidence("We deploy LLMOps pipelines.", llm) is None


def test_todo_2_sans_tenir_compte_de_la_casse():
    c = Criterion(id="C2", label="screening", terms=["study selection"])
    assert find_evidence("Automating Study Selection.", c).criterion == "C2"


def test_todo_2_la_preuve_est_la_phrase():
    c = Criterion(id="C3", label="recall", terms=["recall", "sensitivity"])
    text = "Title here. We report a sensitivity of 0.9. Recall is high."
    assert find_evidence(text, c).quote == "We report a sensitivity of 0.9."


def test_todo_2_aucune_preuve():
    c = Criterion(id="C3", label="recall", terms=["recall"])
    assert find_evidence("Nothing relevant here.", c) is None


def test_todo_3_exclusion_prioritaire(protocol, gold):
    result = rule_screen(gold[0]["G03"], protocol)
    assert (result.decision, result.confidence, result.decider) == ("exclude", 0.9, "regles")
    assert {e.criterion for e in result.evidence} == {"E1"}


def test_todo_3_inclusion_quand_tout_est_present(protocol, gold):
    result = rule_screen(gold[0]["G01"], protocol)
    assert (result.decision, result.confidence) == ("include", 0.9)
    assert [e.criterion for e in result.evidence] == ["C1", "C2", "C3"]


def test_todo_3_aucun_critere(protocol, gold):
    result = rule_screen(gold[0]["G04"], protocol)
    assert (result.decision, result.confidence, result.evidence) == ("exclude", 0.85, [])


def test_todo_3_cas_partiel_abstention(protocol, gold):
    result = rule_screen(gold[0]["G05"], protocol)
    assert (result.decision, result.confidence) == ("uncertain", 0.5)
    assert result.reasons == ["critère absent : C3"]


def test_todo_3_le_piege_lexical(protocol, gold):
    assert rule_screen(gold[0]["G06"], protocol).decision == "include"   # tous les mots, mauvais sens


# ===========================================================================
# Partie B - Le LLM, second évaluateur contrôlé
# ===========================================================================

def test_todo_4_json_conforme():
    text = 'Voici :\n```json\n{"decision": "include", "confidence": 0.8, "reasons": ["ok"], ' \
           '"evidence": [{"criterion": "C1", "quote": "We use LLMs"}]}\n```'
    result = parse_judgment(text, "P1")
    assert (result.decision, result.confidence, result.decider) == ("include", 0.8, "llm")
    assert result.evidence[0].quote == "We use LLMs"


def test_todo_4_texte_libre_vaut_abstention():
    result = parse_judgment("I would probably include it.", "P1")
    assert (result.decision, result.confidence, result.decider) == ("uncertain", 0.0, "llm")
    assert result.reasons[0].startswith("sortie LLM hors contrat")


def test_todo_4_confiance_hors_bornes():
    assert parse_judgment('{"decision": "include", "confidence": 1.5, "reasons": ["x"]}', "P1").decision == "uncertain"


def test_todo_4_decision_inconnue_ou_sans_raison():
    assert parse_judgment('{"decision": "maybe", "confidence": 0.5, "reasons": ["x"]}', "P1").decision == "uncertain"
    assert parse_judgment('{"decision": "include", "confidence": 0.9, "reasons": []}', "P1").decision == "uncertain"


def test_todo_5_preuves_verifiees_conservees():
    paper = make_paper(abstract="We evaluate LLMs for screening. Recall is 0.9.")
    original = decision(value="include", decider="llm",
                        evidence=[Evidence(criterion="C1", quote="we  evaluate LLMs for screening")])
    checked = verify_evidence(original, paper)
    assert (checked.decision, checked.confidence, len(checked.evidence)) == ("include", 0.9, 1)


def test_todo_5_citation_inventee(gold):
    papers, _ = gold
    original = decision("G07", "include", 0.85, "llm",
                        evidence=[Evidence(criterion="C1", quote="we use GPT-4 to rank the records"),
                                  Evidence(criterion="C2", quote="during citation screening")])
    checked = verify_evidence(original, papers["G07"])
    assert (checked.decision, checked.confidence) == ("uncertain", 0.5)
    assert [e.criterion for e in checked.evidence] == ["C2"]
    assert checked.reasons[-1].startswith("citation introuvable")
    assert original.decision == "include" and len(original.evidence) == 2      # copie, pas modification


def test_todo_5_inclusion_sans_preuve():
    checked = verify_evidence(decision(value="include", decider="llm"), make_paper())
    assert checked.decision == "uncertain" and checked.reasons[-1] == "inclusion sans preuve"


def test_todo_5_exclusion_sans_citation_acceptee():
    checked = verify_evidence(decision(value="exclude", confidence=0.95, decider="llm"), make_paper())
    assert (checked.decision, checked.confidence) == ("exclude", 0.95)


# ===========================================================================
# Partie C - Décider, s'abstenir, escalader
# ===========================================================================

def test_todo_6_abstention():
    assert route(decision(value="uncertain", confidence=0.99)) == "humain"


def test_todo_6_inclusion_selon_le_seuil():
    assert route(decision(value="include", confidence=0.75), 0.75) == "auto"
    assert route(decision(value="include", confidence=0.74), 0.75) == "humain"


def test_todo_6_exclusion_exige_une_marge():
    assert EXCLUDE_MARGIN == 0.10
    assert route(decision(value="exclude", confidence=0.80), 0.75) == "humain"
    assert route(decision(value="exclude", confidence=0.85), 0.75) == "auto"


def test_todo_6_seuil_plafonne_a_un():
    assert route(decision(value="exclude", confidence=1.0), 0.95) == "auto"
    assert route(decision(value="exclude", confidence=0.99), 0.95) == "humain"


def test_todo_7_accord_sur_et_confiant(protocol, demo_model, gold):
    paper = gold[0]["G01"]
    record = combine(paper, rule_screen(paper, protocol), llm_screen(demo_model, paper, protocol))
    assert (record.route, record.final.decision, record.final.decider) == ("auto", "include", "accord")
    assert record.final.confidence == 0.9 and record.escalation == ""


def test_todo_7_desaccord_escalade(protocol, demo_model, gold):
    paper = gold[0]["G06"]
    record = combine(paper, rule_screen(paper, protocol), llm_screen(demo_model, paper, protocol))
    assert (record.route, record.final) == ("humain", None)
    assert record.escalation == "désaccord : include (règles) / exclude (LLM)"


def test_todo_7_accord_mais_pas_assez_sur():
    record = combine(make_paper(), decision(value="exclude", confidence=0.8),
                     decision(value="exclude", confidence=0.95, decider="llm"), 0.75)
    assert record.route == "humain" and record.escalation.startswith("confiance insuffisante")


def test_todo_7_double_abstention(protocol, demo_model, gold):
    paper = gold[0]["G08"]
    record = combine(paper, rule_screen(paper, protocol), llm_screen(demo_model, paper, protocol))
    assert record.escalation == "abstention des deux évaluateurs"


def test_todo_8_double_evaluation_sans_erreur_automatique(protocol, demo_model, gold):
    state, labels = gold_state(gold, protocol, demo_model)
    metrics = evaluate(state, labels)
    assert (metrics["auto"], metrics["humain"], metrics["erreurs_auto"]) == (6, 9, 0)
    assert metrics["charge_humaine"] == 0.6


def test_todo_8_automatisation_complete_perd_des_etudes(protocol, demo_model, gold):
    state, labels = gold_state(gold, protocol, demo_model)
    rows = {row["politique"].split(",")[0]: row for row in compare_policies(state, labels)}
    assert (rows["règles seules"]["faux_exclus"], rows["règles seules"]["faux_inclus"]) == (4, 1)
    assert (rows["LLM seul"]["faux_exclus"], rows["LLM seul"]["faux_inclus"]) == (4, 0)


def test_todo_8_un_seuil_trop_bas_coute_une_etude(protocol, demo_model, gold):
    state, labels = gold_state(gold, protocol, demo_model)
    rows = {row["seuil"]: row for row in sweep_thresholds(state, labels)}
    assert rows[0.5]["faux_exclus"] == 1 and rows[0.75]["faux_exclus"] == 0
    assert rows[0.9]["humain"] > rows[0.75]["humain"]


def test_todo_8_rien_d_automatique(protocol):
    state = ScreeningState(protocol=protocol, threshold=0.75)
    assert evaluate(state, {}) == {"articles": 0, "auto": 0, "humain": 0, "charge_humaine": 0.0,
                                   "faux_exclus": 0, "faux_inclus": 0, "erreurs_auto": 0}


# ===========================================================================
# Partie D - Boucler avec l'humain
# ===========================================================================

def test_todo_9_decisions_humaines(protocol, demo_model, gold):
    state, _ = gold_state(gold, protocol, demo_model)
    assert apply_human_review(state, {"G06": "exclude", "G10": "include"}) == 2
    final = {r.paper_id: r.final for r in state.records}
    assert (final["G06"].decision, final["G06"].decider) == ("exclude", "humain")
    assert "désaccord" in final["G10"].reasons[0]
    assert state.counts()["décidés_par_un_humain"] == 2 and state.counts()["en_attente_humain"] == 7


def test_todo_9_pas_de_reecriture_d_une_decision_automatique(protocol, demo_model, gold):
    state, _ = gold_state(gold, protocol, demo_model)
    with pytest.raises(ValueError):
        apply_human_review(state, {"G06": "exclude", "G01": "exclude"})
    assert state.counts()["décidés_par_un_humain"] == 0          # rien n'a été appliqué


def test_todo_9_l_humain_doit_trancher(protocol, demo_model, gold):
    state, _ = gold_state(gold, protocol, demo_model)
    with pytest.raises(ValueError):
        apply_human_review(state, {"G08": "uncertain"})


def test_todo_10_seuil_sur_le_jeu_etiquete(protocol, demo_model, gold):
    state, labels = gold_state(gold, protocol, demo_model)
    assert choose_threshold(sweep_thresholds(state, labels)) == 0.75


def test_todo_10_aucun_seuil_sur():
    assert choose_threshold([{"seuil": 0.9, "humain": 4, "faux_exclus": 0}]) == 0.9
    assert choose_threshold([{"seuil": 0.5, "humain": 1, "faux_exclus": 2}]) is None


def test_todo_10_egalite_le_plus_prudent():
    rows = [{"seuil": 0.6, "humain": 3, "faux_exclus": 0}, {"seuil": 0.75, "humain": 3, "faux_exclus": 0},
            {"seuil": 0.9, "humain": 5, "faux_exclus": 0}, {"seuil": 0.5, "humain": 2, "faux_exclus": 1}]
    assert choose_threshold(rows) == 0.75
