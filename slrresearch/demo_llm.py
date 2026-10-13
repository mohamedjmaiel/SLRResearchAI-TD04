"""LLM de démonstration (code fourni) : reproductible, sans clé ni réseau.

Il imite un ChatModel LangChain (méthode invoke -> objet avec .content) et
répond selon le type de prompt reçu :
- prompt few-shot du TD02          -> trois RQ conformes ;
- prompt de concepts du TD02       -> un JSON de concepts ;
- prompt de décision du niveau 3   -> une action, selon un scénario ReAct ;
- prompt de screening du TD04      -> le jugement prévu pour l'article
  (data/jeu_etiquete_td04.json et data/resumes_td04.json).

Scénarios ReAct :
- "discipline"   : propose des actions valides et termine proprement ;
- "indiscipline" : se trompe d'outil, répond hors contrat, oublie un champ,
                   répète une action... puis finit par chercher et terminer.
"""
import json
import re
from pathlib import Path
from types import SimpleNamespace

DATA = Path(__file__).resolve().parent.parent / "data"

DEMO_QUESTIONS = (
    "RQ1. Comment les LLM améliorent-ils le rappel du screening des SLR ?\n"
    "RQ2. Quel contrôle humain doit-il être conservé lors du screening assisté ?\n"
    "RQ3. Quelles métriques évaluent-elles la qualité du screening automatisé ?"
)

LLM_TERMS = {"label": "large language models",
             "terms": ["large language models", "LLM", "generative AI"]}
SCREENING_TERMS = {"label": "screening",
                   "terms": ["screening", "study selection", "title and abstract screening"]}
DEMO_CONCEPTS = {
    "rappel": [LLM_TERMS, SCREENING_TERMS, {"label": "recall", "terms": ["recall", "sensitivity"]}],
    "humain": [{"label": "human oversight", "terms": ["human oversight", "human-in-the-loop"]},
               SCREENING_TERMS, LLM_TERMS],
    "métriques": [{"label": "evaluation metrics",
                   "terms": ["evaluation metrics", "precision", "work saved over sampling"]},
                  SCREENING_TERMS, {"label": "automation", "terms": ["automation", "automated"]}],
}

REACT_SCENARIOS = ("discipline", "indiscipline")


def _text_of(prompt) -> str:
    if isinstance(prompt, str):
        return prompt
    return "\n".join(str(getattr(message, "content", message)) for message in prompt)


def _state_of(text: str) -> dict:
    match = re.search(r"\[ÉTAT\]\n(\{.*?\})\n\[", text, re.DOTALL)
    return json.loads(match.group(1)) if match else {}


class DemoChatModel:
    """Faux ChatModel déterministe pour la démonstration et les tests."""

    def __init__(self, react_scenario: str = "discipline"):
        if react_scenario not in REACT_SCENARIOS:
            raise ValueError(f"Scénario ReAct inconnu : {react_scenario}")
        self.react_scenario = react_scenario
        self.prompts: list[str] = []

    def invoke(self, prompt):
        text = _text_of(prompt)
        self.prompts.append(text)
        return SimpleNamespace(content=self._answer(text))

    # -- réponses ----------------------------------------------------------
    def _answer(self, text: str) -> str:
        if "[SCREENING]" in text:
            return self._screen(text)
        if "[DÉCISION]" in text:
            return self._decide(_state_of(text))
        if "Tu prépares une recherche bibliographique" in text:
            question = text.split("[QUESTION]\n", 1)[-1].split("\n", 1)[0]
            for key, concepts in DEMO_CONCEPTS.items():
                if key in question:
                    return json.dumps({"concepts": concepts}, ensure_ascii=False)
            return json.dumps({"concepts": [LLM_TERMS, SCREENING_TERMS]}, ensure_ascii=False)
        return DEMO_QUESTIONS

    def _screen(self, text: str) -> str:
        match = re.search(r"^id : (.+)$", text, re.MULTILINE)
        key = match.group(1).strip() if match else ""
        scripted = {}
        gold = DATA / "jeu_etiquete_td04.json"
        if gold.exists():
            scripted.update({p["id"]: p["llm_demo"] for p in json.loads(gold.read_text(encoding="utf-8"))["papers"]})
        abstracts = DATA / "resumes_td04.json"
        if abstracts.exists():
            scripted.update({k: v["llm_demo"] for k, v in
                             json.loads(abstracts.read_text(encoding="utf-8"))["abstracts"].items()})
        answer = scripted.get(key)
        if answer is None:
            answer = {"decision": "uncertain", "confidence": 0.3,
                      "reasons": ["article inconnu du LLM de démonstration"], "evidence": []}
        if isinstance(answer, str):
            return answer
        return "```json\n" + json.dumps(answer, ensure_ascii=False) + "\n```"

    def _decide(self, state: dict) -> str:
        queries = state.get("requetes_td02", {})
        step = state.get("pas_utilises", 0)
        if self.react_scenario == "indiscipline":
            script = [
                {"action": "search", "tool": "scholar", "query": queries.get("scholar", "")},
                "Je pense qu'il faudrait d'abord chercher sur OpenAlex.",
                {"action": "finish", "tool": "openalex"},
                {"action": "search", "tool": "openalex", "query": queries.get("scholar", ""), "limit": 10},
                {"action": "search", "tool": "openalex", "query": queries.get("scholar", ""), "limit": 10},
                {"action": "search", "tool": "crossref", "query": queries.get("ieee", ""), "limit": 10},
                {"action": "finish"},
            ]
            choice = script[min(step, len(script) - 1)]
            return choice if isinstance(choice, str) else json.dumps(choice, ensure_ascii=False)
        # Scénario discipliné : suit le plan, relance une fois, puis termine.
        history = state.get("historique", [])
        called = [item["outil"] for item in history]
        last = history[-1] if history else None
        if last and last["statut"] == "rate_limited" and called.count(last["outil"]) == 1:
            base = {"openalex": "scholar", "crossref": "ieee",
                    "semantic_scholar": "acm"}.get(last["outil"], "scholar")
            action = {"action": "search", "tool": last["outil"], "query": queries.get(base, "")}
        else:
            for tool, base in (("openalex", "scholar"), ("crossref", "ieee"),
                               ("semantic_scholar", "acm")):
                if tool not in called and base in queries:
                    action = {"action": "search", "tool": tool, "query": queries[base], "limit": 10}
                    break
            else:
                if state.get("articles_uniques", 0) < 3 and "openalex_broad" not in called:
                    action = {"action": "search", "tool": "openalex_broad",
                              "query": queries.get("scholar", "")}
                else:
                    action = {"action": "finish"}
        return "```json\n" + json.dumps(action, ensure_ascii=False) + "\n```"
