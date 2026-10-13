"""SearchAgent du TD03 : des outils à l'agent, en trois niveaux.

Niveau 1 - workflow outillé        run_workflow()          le plan fixe (ROUTES) décide
Niveau 2 - agent à politique codée run_agent(decide_next)  des règles Python décident
Niveau 3 - agent ReAct contrôlé    run_agent(llm_decide)   le LLM propose, le logiciel dispose

Ce fichier est le seul à modifier : TODO 1 à TODO 10.
"""
import json
from typing import Callable, Literal

from pydantic import BaseModel, Field, model_validator

from slrresearch.models import CALL_STATUSES, ActionError, PaperRecord, TraceEvent
from slrresearch.search_plan import call_llm, extract_json_object
from slrresearch.tools import SearchTool, ToolError, extract_fields

# Base logique du TD02 -> outil du registre (routage explicite et versionné).
ROUTES = {"scholar": "openalex", "ieee": "crossref", "acm": "semantic_scholar"}
FALLBACK_TOOL = "openalex_broad"   # source de secours : OpenAlex en mots-clés
DOI_PREFIXES = ("https://doi.org/", "http://doi.org/", "doi.org/", "doi:")
DEFAULT_LIMIT = 10
MAX_CALLS = 5      # niveau 2 : budget d'appels de la politique codée
MAX_STEPS = 8      # niveaux 2 et 3 : borne de la boucle
MIN_UNIQUE = 3     # niveau 2 : en dessous, on interroge la source de secours


# ===========================================================================
# Contrats de l'agent
# ===========================================================================

class AgentAction(BaseModel):
    """Ce qu'un décideur (plan, règles ou LLM) peut proposer : rien d'autre."""

    action: Literal["search", "finish"]
    tool: str | None = None      # nom d'un outil du registre, jamais une base
    query: str | None = None
    limit: int | None = Field(default=None, ge=1, le=50)

    @model_validator(mode="after")
    def check_rules(self):
        """TODO 8 - Règles conditionnelles du contrat d'action."""
        if self.action == "search" and (not self.tool or not self.query):
            raise ValueError("search exige un tool et une query")
        if self.action == "finish" and self.tool is not None:
            raise ValueError("finish n'utilise aucun outil")
        return self


class SearchState(BaseModel):
    """État de l'agent : ce que le système sait avant la prochaine décision."""

    objective: str = ""
    queries: dict[str, str]                       # requêtes du TD02, par base
    raw_records: list[PaperRecord] = Field(default_factory=list)
    papers: list[PaperRecord] = Field(default_factory=list)
    actions_executed: list[AgentAction | None] = Field(default_factory=list)
    observations: list[TraceEvent] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    step: int = 0
    finished: bool = False
    raw_count: int = 0
    duplicates_removed: int = 0
    status: Literal["pending", "complete", "partial", "failed"] = "pending"

    def add(self, action: AgentAction | None, records: list[PaperRecord],
            event: TraceEvent) -> None:
        """Met à jour l'état après une action : action, observation, résultats."""
        event.step = self.step + 1
        self.actions_executed.append(action)
        self.observations.append(event)
        self.raw_records.extend(records)
        if event.status != "success":
            self.errors.append(f"{event.tool} : {event.error or event.status}")
        self.step += 1

    def tools_called(self) -> list[str]:
        """Outils réellement appelés (succès ou échec d'appel), dans l'ordre."""
        return [e.tool for e in self.observations if e.status in CALL_STATUSES]

    def unique_count(self) -> int:
        return len(deduplicate(self.raw_records)[0])


# ===========================================================================
# Niveau 1 - Workflow outillé
# ===========================================================================

def normalize_doi(doi: str | None) -> str | None:
    """TODO 1 - DOI comparable : minuscules, sans préfixe, sans espaces."""
    if not doi:
        return None
    clean = doi.strip().lower()
    for prefix in DOI_PREFIXES:
        if clean.startswith(prefix):
            clean = clean[len(prefix):]
    clean = clean.replace(" ", "")
    return clean or None


def to_paper_record(source: str, raw: dict) -> PaperRecord:
    """Code fourni : réponse brute d'une API -> PaperRecord, avec provenance."""
    fields = extract_fields(source, raw)
    return PaperRecord(
        id=fields["id"] or f"{source}:{fields['title']}",
        title=fields["title"] or "(sans titre)",
        doi=normalize_doi(fields["doi"]),
        url=fields["url"],
        year=fields["year"],
        open_access=fields["open_access"],
        sources=[source],
    )


def call_tool(
    tool: SearchTool, query: str, limit: int = DEFAULT_LIMIT
) -> tuple[list[PaperRecord], TraceEvent]:
    """TODO 2 - Appeler un outil, normaliser sa sortie, tracer l'appel.

    Un échec (ToolError) n'est jamais propagé : il devient une observation.
    """
    try:
        raw_items = tool.search(query, limit)
    except ToolError as error:
        return [], TraceEvent(
            tool=tool.name, query=query, status=error.status, error=str(error)
        )
    records = [to_paper_record(tool.name, raw) for raw in raw_items]
    return records, TraceEvent(
        tool=tool.name, query=query, status="success", found=len(records)
    )


def dedup_key(record: PaperRecord) -> str:
    """TODO 3.1 - Clé de déduplication : DOI normalisé, sinon titre normalisé."""
    if record.doi:
        return "doi:" + record.doi
    return "title:" + " ".join(record.title.lower().split())


def deduplicate(records: list[PaperRecord]) -> tuple[list[PaperRecord], int]:
    """TODO 3.2 - Un article, une entrée, toutes ses sources.

    Retourne (articles uniques, nombre de doublons retirés).
    """
    unique: dict[str, PaperRecord] = {}
    for record in records:
        key = dedup_key(record)
        if key not in unique:
            unique[key] = record.model_copy(deep=True)
            continue
        kept = unique[key]
        for source in record.sources:
            if source not in kept.sources:
                kept.sources.append(source)
    return list(unique.values()), len(records) - len(unique)


def overall_status(observations: list[TraceEvent]) -> str:
    """Code fourni : complete, partial ou failed, outil par outil.

    Un outil a réussi s'il a répondu au moins une fois. Un outil refusé compte
    comme une source manquante. Une action invalide n'est pas un appel.
    """
    outcome: dict[str, bool] = {}
    for event in observations:
        if event.status in CALL_STATUSES or event.status == "refused":
            outcome[event.tool] = outcome.get(event.tool, False) or event.status == "success"
    if outcome and all(outcome.values()):
        return "complete"
    if any(outcome.values()):
        return "partial"
    return "failed"


def finalize(state: SearchState) -> SearchState:
    """Code fourni : consolider les résultats et calculer le statut global."""
    state.papers, state.duplicates_removed = deduplicate(state.raw_records)
    state.raw_count = len(state.raw_records)
    state.status = overall_status(state.observations)
    return state


def authorize(tool_name: str | None, registry: dict[str, SearchTool]) -> SearchTool:
    """Code fourni : moindre privilège, refus AVANT tout appel externe."""
    if not tool_name or tool_name not in registry:
        raise ActionError("refused", f"Outil non autorisé : {tool_name}")
    return registry[tool_name]


def refused_event(action: AgentAction | None, error: ActionError) -> TraceEvent:
    """Code fourni : une action refusée ou invalide devient une observation."""
    return TraceEvent(
        tool=(action.tool if action and action.tool else "(aucun)"),
        query=(action.query or "") if action else "",
        status=error.status,
        error=str(error),
    )


def run_workflow(
    queries: dict[str, str],
    registry: dict[str, SearchTool],
    limit: int = DEFAULT_LIMIT,
    objective: str = "",
) -> SearchState:
    """TODO 4 - Niveau 1 : exécuter le plan fixe ROUTES, sans décision."""
    state = SearchState(objective=objective, queries=queries)
    for base, tool_name in ROUTES.items():
        if base not in queries:
            continue
        action = AgentAction(action="search", tool=tool_name,
                             query=queries[base], limit=limit)
        try:
            tool = authorize(tool_name, registry)
        except ActionError as error:
            state.add(action, [], refused_event(action, error))
            continue
        records, event = call_tool(tool, action.query, limit)
        state.add(action, records, event)
    state.finished = True
    return finalize(state)


# ===========================================================================
# Niveau 2 - Agent à politique codée
# ===========================================================================

def retry_after_rate_limit(state: SearchState) -> AgentAction | None:
    """TODO 9 - Relancer UNE fois la dernière action si elle a reçu un 429."""
    if not state.observations or state.actions_executed[-1] is None:
        return None
    last_event = state.observations[-1]
    last_action = state.actions_executed[-1]
    if last_event.status != "rate_limited":
        return None
    attempts = [
        action for action in state.actions_executed
        if action is not None
        and action.tool == last_action.tool and action.query == last_action.query
    ]
    if len(attempts) > 1:
        return None
    return last_action.model_copy()


def decide_next(
    state: SearchState,
    max_calls: int = MAX_CALLS,
    min_unique: int = MIN_UNIQUE,
) -> AgentAction:
    """TODO 5 - Politique codée : règles évaluées dans l'ordre."""
    called = state.tools_called()                      # appels réels (budget)
    attempted = {event.tool for event in state.observations}  # appels + refus
    # Règle 1 - budget d'appels épuisé -> finish (garantit la terminaison).
    if len(called) >= max_calls:
        return AgentAction(action="finish")
    # Règle 2 - relance unique après un 429 (voir TODO 9).
    retry = retry_after_rate_limit(state)
    if retry is not None:
        return retry
    # Règle 3 - plan incomplet -> source suivante de ROUTES.
    for base, tool_name in ROUTES.items():
        if base in state.queries and tool_name not in attempted:
            return AgentAction(action="search", tool=tool_name,
                               query=state.queries[base], limit=DEFAULT_LIMIT)
    # Règle 4 - plan complet mais trop peu d'articles -> secours, une seule fois.
    if (state.unique_count() < min_unique and FALLBACK_TOOL not in attempted
            and "scholar" in state.queries):
        return AgentAction(action="search", tool=FALLBACK_TOOL,
                           query=state.queries["scholar"], limit=DEFAULT_LIMIT)
    # Règle 5 - sinon -> finish.
    return AgentAction(action="finish")


def validate_action(proposal) -> AgentAction:
    """Code fourni : appliquer le contrat AgentAction à une proposition."""
    data = proposal.model_dump() if isinstance(proposal, AgentAction) else proposal
    try:
        return AgentAction.model_validate(data)
    except ValueError as error:
        raise ActionError("invalid", f"Action invalide : {error}") from error


def check_not_repeated(action: AgentAction, state: SearchState) -> None:
    """TODO 10 - Refuser une action déjà exécutée (sauf relance après un 429)."""
    previous = [
        index for index, done in enumerate(state.actions_executed)
        if done is not None and done.action == "search"
        and done.tool == action.tool and done.query == action.query
    ]
    if not previous:
        return
    if len(previous) == 1 and state.observations[previous[0]].status == "rate_limited":
        return
    raise ActionError("refused", f"Action déjà exécutée : {action.tool}")


def run_agent(
    state: SearchState,
    decide: Callable[[SearchState], AgentAction],
    registry: dict[str, SearchTool],
    max_steps: int = MAX_STEPS,
) -> SearchState:
    """TODO 6 - Boucle bornée : décider -> contrôler -> agir -> observer.

    La même boucle sert aux niveaux 2 et 3 : seul `decide` change.
    """
    for _ in range(max_steps):
        action = None
        try:
            action = validate_action(decide(state))
            if action.action == "finish":
                state.finished = True
                break
            check_not_repeated(action, state)
            tool = authorize(action.tool, registry)
        except ActionError as error:
            state.add(action, [], refused_event(action, error))
            continue
        records, event = call_tool(tool, action.query, action.limit or DEFAULT_LIMIT)
        state.add(action, records, event)
    return finalize(state)


# ===========================================================================
# Niveau 3 - Agent ReAct contrôlé
# ===========================================================================

def describe_state(state: SearchState, registry: dict[str, SearchTool]) -> str:
    """Code fourni : la partie de l'état utile à la décision (≠ état complet)."""
    view = {
        "objectif": state.objective,
        "requetes_td02": state.queries,
        "outils_autorises": sorted(registry),
        "pas_utilises": state.step,
        "pas_max": MAX_STEPS,
        "articles_uniques": state.unique_count(),
        "historique": [
            {"pas": e.step, "outil": e.tool, "statut": e.status, "trouves": e.found}
            for e in state.observations[-6:]
        ],
    }
    return json.dumps(view, ensure_ascii=False, indent=1)


def build_decision_prompt(state: SearchState, registry: dict[str, SearchTool]) -> str:
    """Code fourni : prompt de décision du niveau 3."""
    tools = "\n".join(
        f"- {name} : {tool.description}" for name, tool in sorted(registry.items())
    )
    return f"""Tu pilotes un agent de recherche bibliographique pour une revue systématique.

[OBJECTIF]
Retrouver des articles réels pour la question de recherche, en interrogeant
les outils autorisés avec les requêtes du TD02.

[OUTILS AUTORISÉS]
{tools}

[ÉTAT]
{describe_state(state, registry)}
[RÈGLES]
- Réponds uniquement par un objet JSON conforme au contrat AgentAction.
- Rechercher : {{"action": "search", "tool": "<outil autorisé>", "query": "<requête>", "limit": 10}}
- Terminer   : {{"action": "finish"}}
- tool désigne un outil du registre (ex. openalex), jamais une base (scholar, ieee, acm).
- Utilise les requêtes du TD02 sans les recopier à la main ; ne répète pas une action réussie.
- Termine lorsque chaque source utile a été interrogée.
- N'appelle aucune fonction : écris l'objet JSON comme simple texte de réponse.

[DÉCISION]
"""


def parse_agent_action(text: str) -> AgentAction:
    """TODO 7 - Sortie texte du LLM -> AgentAction, ou ActionError("invalid")."""
    try:
        return AgentAction.model_validate_json(extract_json_object(text))
    except ValueError as error:
        raise ActionError("invalid", f"Sortie LLM hors contrat : {error}") from error


def recover_rejected_generation(error: Exception) -> str | None:
    """Code fourni : récupérer le texte d'une réponse rejetée par le fournisseur.

    Certains modèles (ex. gpt-oss sur Groq) tentent un « appel de fonction »
    natif alors qu'aucune fonction n'est déclarée. Le fournisseur rejette la
    réponse (HTTP 400, code tool_use_failed) mais renvoie la génération dans
    `failed_generation`. On en extrait les arguments : ils passeront par le
    même contrat AgentAction que toute autre proposition du LLM.
    """
    body = getattr(error, "body", None)
    details = body.get("error", body) if isinstance(body, dict) else {}
    failed = details.get("failed_generation") if isinstance(details, dict) else None
    if not failed:
        return None
    try:
        data = json.loads(failed)
    except ValueError:
        return failed                       # texte brut : parse_agent_action jugera
    if isinstance(data, dict) and "arguments" in data:
        arguments = data["arguments"]
        return arguments if isinstance(arguments, str) else json.dumps(arguments)
    return failed


def llm_decide(model, state: SearchState, registry: dict[str, SearchTool]) -> AgentAction:
    """Code fourni : le LLM PROPOSE une action ; il n'exécute rien."""
    prompt = build_decision_prompt(state, registry)
    try:
        text = call_llm(model, prompt)
    except Exception as error:              # erreur du fournisseur LLM
        text = recover_rejected_generation(error)
        if text is None:
            raise                           # réseau, clé, quota : vraie panne
    return parse_agent_action(text)


def make_llm_decider(model, registry: dict[str, SearchTool]):
    """Code fourni : adapte llm_decide à l'interface decide(state)."""
    return lambda state: llm_decide(model, state, registry)


# ===========================================================================
# Code fourni : lancer un niveau avec la même entrée et la même sortie
# ===========================================================================

def run_level(
    level: int,
    queries: dict[str, str],
    registry: dict[str, SearchTool],
    model=None,
    objective: str = "",
) -> SearchState:
    """Niveau 1, 2 ou 3 : mêmes requêtes en entrée, un SearchState en sortie."""
    if level == 1:
        return run_workflow(queries, registry, objective=objective)
    state = SearchState(objective=objective, queries=queries)
    if level == 2:
        return run_agent(state, decide_next, registry)
    if level == 3:
        if model is None:
            raise ValueError("Le niveau 3 exige un modèle de chat")
        return run_agent(state, make_llm_decider(model, registry), registry)
    raise ValueError("Niveau attendu : 1, 2 ou 3")


def state_summary(state: SearchState) -> dict:
    """Indicateurs PRISMA et de contrôle d'un SearchState."""
    calls = [e for e in state.observations if e.status in CALL_STATUSES]
    return {
        "statut": state.status,
        "pas": state.step,
        "appels_outils": len(calls),
        "refus_ou_invalides": state.step - len(calls),
        "articles_bruts": state.raw_count,
        "articles_uniques": len(state.papers),
        "doublons_retires": state.duplicates_removed,
        "termine": state.finished,
    }
