# TD04 - ScreeningAgent : décider, s'abstenir, escalader

## 1. Mission

Au TD03, le `SearchAgent` a trouvé des articles. Une revue systématique ne les
garde pas tous : chaque article passe un **screening titre/résumé**. Il est
inclus s'il répond aux critères du protocole, exclu sinon.

Le screening est une **décision sous incertitude**. Un résumé peut être absent,
ambigu ou trompeur. Un évaluateur, humain ou automatique, peut se tromper. Et
les deux erreurs n'ont pas le même coût :

| Erreur | Conséquence |
|---|---|
| inclure à tort | un article de plus à lire plus tard : coût de temps |
| **exclure à tort** | une étude pertinente **perdue** : la revue est biaisée |

Un agent de screening ne doit donc pas seulement décider. Il doit savoir
**quand ne pas décider**. Dans ce TD, chaque article reçoit une issue parmi
trois :

```text
agir         la décision est sûre : inclusion ou exclusion automatique
s'abstenir   l'information est insuffisante : pas de décision automatique
escalader    les avis divergent ou la confiance est trop faible : un humain tranche
```

Idée directrice :

> **Une décision automatique doit être justifiée, vérifiable et assez sûre.
> Sinon, l'agent s'abstient et passe la main.**

## 2. Objectifs pédagogiques

À la fin du TD, vous devez savoir :

1. traduire une question de recherche en critères d'éligibilité exécutables ;
2. produire une décision explicable : décision, confiance, raisons, preuves ;
3. distinguer un critère lexical (règle) d'un jugement de LLM, et connaître leurs limites ;
4. contrôler un jugement de LLM : contrat de sortie et citations vérifiées ;
5. formaliser une politique de décision à trois issues : agir, s'abstenir, escalader ;
6. tenir compte de l'asymétrie des erreurs (exclure à tort coûte plus cher) ;
7. combiner deux évaluateurs indépendants et escalader leurs désaccords ;
8. mesurer les erreurs d'une politique sur un jeu étiqueté, et comparer
   automatisation complète et humain dans la boucle ;
9. intégrer les décisions humaines et choisir un seuil à partir des données.

## 3. Prérequis

- TD02 : contrats Pydantic, `extract_json_object()`, revue humaine des concepts.
- TD03 : `PaperRecord`, trace, décider → contrôler → agir, abstention plutôt
  que plantage (sortie hors contrat → observation `invalid`).
- Python : `re.search`, compréhensions de listes, `BaseModel.model_copy()`.

Le TD02 et le TD03 complets sont fournis. Vous ne les modifiez pas : vous les
utilisez.

## 4. Architecture

```text
ÉTAPE 1 · mesurer           jeu étiqueté (15 articles, décisions humaines connues)
ÉTAPE 2 · appliquer         sujet → TD02 (RQ, concepts validés) → TD03 (articles) → résumés

                  pour chaque article (titre + résumé)
                     │
       ┌─────────────┴──────────────┐
  RÈGLES (Partie A)            LLM (Partie B)
  protocole → preuves          jugement JSON → contrat → citations vérifiées
       └─────────────┬──────────────┘
                     ▼
        POLITIQUE (Partie C) : route() · combine()
        accord ET confiance suffisante → décision AUTOMATIQUE
        sinon                          → FILE HUMAINE (motif d'escalade)
                     ▼
        évaluation (erreurs, charge humaine) · revue humaine (Partie D)
```

### Fichiers

Fichier à compléter :

```text
slrresearch/screening_agent.py      TODO 1 à TODO 10
```

Fichiers fournis, à lire mais à ne pas modifier :

| Fichier | Rôle |
|---|---|
| `slrresearch/abstracts.py` | résumés (simulés ou OpenAlex), jeu étiqueté |
| `slrresearch/search_agent.py` | acquis complet du TD03 |
| `slrresearch/td02_bridge.py` · `search_plan.py` | acquis du TD02 : du sujet aux concepts validés |
| `slrresearch/demo_llm.py` | LLM de démonstration, sans clé ni réseau |
| `data/jeu_etiquete_td04.json` | 15 articles fictifs étiquetés par deux relecteurs humains |
| `data/resumes_td04.json` | résumés simulés des articles du TD03 |
| `main.py` · `streamlit_app.py` | exécution en ligne de commande et interface |
| `tests/test_td04.py` | contrat exécutable de chaque TODO |

### Contrats (fournis, en haut de `screening_agent.py`)

```python
class Criterion(BaseModel):        # un critère : satisfait si l'un de ses termes apparaît
    id: str                        # C1, C2... (inclusion) · E1, E2... (exclusion)
    label: str
    terms: list[str]

class Protocol(BaseModel):         # tous les C requis, aucun E toléré
    question: str
    inclusion: list[Criterion]
    exclusion: list[Criterion]

class Evidence(BaseModel):         # une preuve = un critère + un extrait du texte
    criterion: str
    quote: str

class ScreeningDecision(BaseModel):
    paper_id: str
    decision: Literal["include", "exclude", "uncertain"]
    confidence: float              # entre 0 et 1
    reasons: list[str]
    evidence: list[Evidence]
    decider: Literal["regles", "llm", "accord", "humain"]

class ScreeningRecord(BaseModel):  # le dossier d'un article
    paper_id, title
    rules: ScreeningDecision       # avis des règles
    llm: ScreeningDecision         # avis du LLM (contrôlé)
    route: Literal["auto", "humain"]
    escalation: str                # pourquoi l'humain est sollicité
    final: ScreeningDecision | None   # None : en attente d'un humain
```

`ScreeningState` regroupe le protocole, le seuil et les dossiers ;
`state.human_queue()` renvoie les dossiers en attente.

### Constantes

```python
DEFAULT_THRESHOLD = 0.75   # confiance minimale pour une décision automatique
EXCLUDE_MARGIN = 0.10      # une exclusion automatique exige seuil + 0.10
```

### Le jeu étiqueté

Avant de confier des décisions à un agent, on **mesure** ses erreurs sur des
articles dont la bonne décision est connue. Les 15 articles de
`data/jeu_etiquete_td04.json` portent la RQ1 du sujet de référence :
« Comment les LLM améliorent-ils le rappel du screening des SLR ? ». Chacun
rend un cas observable (champ `cas`) :

- cas nets ;
- synonymes ;
- piège lexical ;
- critère implicite ;
- résumé absent ;
- citation inventée par le LLM ;
- sortie LLM hors contrat ;
- désaccord ;
- erreur commune aux deux évaluateurs.

## 5. Installation et lancement

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env          # utile seulement pour le mode « LLM réel »
```

| Commande | Effet |
|---|---|
| `python -m pytest -v -k "fourni"` | tests du code fourni (verts dès le départ) |
| `python -m pytest -v -k "todo_4_"` | tests d'un TODO (le `_` final évite que `todo_1` sélectionne aussi `todo_10`) |
| `python -m pytest -v` | validation complète |
| `python main.py` | exécution guidée |
| `python main.py --llm demo --etape jeu` | étape 1 seule : mesurer la politique |
| `python main.py --llm demo --etape sujet --accepter --rq RQ1` | étape 2 seule : sujet → décisions |
| `python -m streamlit run streamlit_app.py` | interface pédagogique (onglets 1 et 2) |

Sans option, `main.py` est guidé :

- il demande le mode du LLM, l'étape, puis **le sujet de la revue** ;
- le TD02 génère les RQ et les concepts, que vous validez ;
- ces concepts deviennent les critères d'inclusion ;
- le TD03 trouve les articles ;
- le screening les trie ;
- vous tranchez la file humaine.

Dès qu'une option de choix est fournie, les choix non précisés prennent leur
valeur par défaut. La revue humaine reste interactive, sauf avec `--sans-revue`.
`--seuil 0.6` change le seuil de confiance.

Le mode `demo` n'exige ni clé ni réseau ; son LLM rejoue des jugements prévus
pour le sujet de référence. Le mode `reel` utilise le fournisseur de `.env`
(saisissez alors votre propre sujet) ; avec `--sources reel`, les résumés
viennent d'OpenAlex.

---

## Étape 0 - Observer avant de coder

1. Lancez `python -m pytest -v -k "fourni"` : tous ces tests passent.
2. Lancez `python -m pytest -q` : les tests `todo_*` échouent. C'est votre
   contrat de travail.
3. Ouvrez `data/jeu_etiquete_td04.json`. Lisez les articles `G01`, `G06`,
   `G07` et `G15` : titre, résumé, étiquette `gold`, réponse prévue du LLM
   (`llm_demo`).
4. Lancez `python main.py --llm demo --etape jeu` : l'exécution s'arrête sur
   « Non disponible : TODO 1 ».

Répondez par écrit :

- Pour `G06`, un relecteur humain exclut l'article. Pourquoi une recherche de
  mots-clés l'inclurait-elle ?
- Pour `G07`, la citation proposée par le LLM figure-t-elle dans le résumé ?
- Pour `G15`, que disent les deux évaluateurs, et que dit l'humain ?
- Quelle erreur vous semble la plus grave pour une revue systématique : inclure
  `G06` à tort ou exclure `G15` à tort ?

---

# Partie A - Du protocole aux preuves

Le premier évaluateur applique le protocole à la lettre. Ses décisions sont
déterministes, explicables et testables, mais il ne comprend pas le sens.

## TODO 1 - Des concepts validés aux critères

### Objectif

Au TD02, vous avez validé les concepts d'une RQ (par exemple « large language
models », « screening », « recall »). Ils ont servi à construire les requêtes.
Ils servent maintenant de **critères d'inclusion** : un article pertinent doit
traiter de chacun d'eux.

### Fonction à compléter

```python
def protocol_from_concepts(question, concepts, exclusion=None) -> Protocol:
    # TODO 1 - Pour chaque concept (numéroté à partir de 1), créer
    #          Criterion(id=f"C{numéro}", label=group.label, terms=list(group.terms)).
    #          Indice : enumerate(concepts, start=1).
```

La suite (fournie) ajoute les critères d'exclusion : ceux que l'humain a
choisis, sinon `DEFAULT_EXCLUSION` (E1 domaine clinique, E2 publication non
éligible).

### Vérification

```bash
python -m pytest -v -k "todo_1_"
```

## TODO 2 - Trouver une preuve

### Objectif

Une décision explicable cite ses **preuves** : la phrase du titre ou du résumé
qui satisfait un critère. Pour éviter les faux positifs d'une recherche par
sous-chaîne, un terme doit apparaître comme **mot entier**. La casse est
ignorée et le pluriel en « s » est accepté.

```text
« LLM » trouve  « We compare LLMs. »
« LLM » ignore  « We deploy LLMOps pipelines. »
```

### Fonction à compléter

```python
def find_evidence(text, criterion) -> Evidence | None:
    for sentence in sentences(text):
        for term in criterion.terms:
            pattern = r"\b" + re.escape(term) + r"s?\b"        # fourni
            # TODO 2 - Si re.search(pattern, sentence, flags=re.IGNORECASE) trouve le terme :
            #          retourner Evidence(criterion=criterion.id, quote=sentence).
    return None
```

`\b` désigne une frontière de mot ; `re.escape` protège les caractères
spéciaux d'un terme (`GPT-4`, `C++`).

### Vérification

```bash
python -m pytest -v -k "todo_2_"
```

## TODO 3 - Le premier évaluateur : les règles

### Objectif

Le code fourni calcule les preuves d'exclusion (`excluded`), les preuves
d'inclusion trouvées (`matched`) et les critères manquants (`missing`). Un
résumé absent donne déjà une abstention (fourni). Vous écrivez la décision :

| # | Situation | Décision | Confiance |
|---|---|---|---|
| 1 | un critère d'exclusion est présent | `exclude` | 0.9 |
| 2 | tous les critères d'inclusion sont présents | `include` | 0.9 |
| 3 | aucun critère d'inclusion | `exclude` | 0.85 |
| 4 | certains critères seulement | `uncertain` | 0.5 |

Les raisons exactes à écrire sont données dans le squelette. Ici, la confiance
est **fixée par la règle** : ce n'est pas une probabilité calculée.

### Vérification

```bash
python -m pytest -v -k "todo_3_"
```

Le dernier test vérifie une limite volontaire. Les règles **incluent** `G06`
(« passenger screening » à l'aéroport) : tous les mots sont présents, mais pas
dans le bon sens.

### Erreurs fréquentes

- Tester l'inclusion avant l'exclusion : `G03` (essais cliniques) serait inclus.
- Confondre `missing` vide (tout est présent) et `matched` vide (rien n'est
  présent).

### Point de contrôle A

Les règles sont rapides, gratuites et explicables, mais elles ne comprennent
pas le sens. Elles incluent `G06` à tort. Elles s'abstiennent sur `G12`, qui
parle de « GPT-4 » et de « false negatives » sans écrire « LLM » ni
« recall ». Elles excluent `G15`, qui n'emploie aucun terme du protocole. Il
faut un second avis.

---

# Partie B - Le LLM, second évaluateur contrôlé

Le LLM comprend le sens et les synonymes. Mais il peut répondre hors format,
se tromper avec assurance ou **inventer une citation**. Comme au TD02, on ne lui
fait pas confiance : on contrôle sa sortie. Le prompt est fourni
(`build_screening_prompt`). `llm_screen` (fourni) appelle le LLM, puis vos
TODO 4 et 5.

## TODO 4 - Le contrat de sortie

### Objectif

Le LLM doit renvoyer un objet JSON conforme à `LLMJudgment` : décision,
confiance entre 0 et 1, au moins une raison, preuves. Une sortie hors contrat
n'est pas une erreur fatale : c'est une **abstention**.

### Fonction à compléter

```python
def parse_judgment(text: str, paper_id: str) -> ScreeningDecision:
    # TODO 4.1 - Dans un try : judgment = LLMJudgment.model_validate_json(extract_json_object(text))
    # TODO 4.2 - except ValueError as error : retourner une décision "uncertain",
    #            confiance 0.0, reasons=[f"sortie LLM hors contrat : {...}"], decider="llm".
    # TODO 4.3 - Sinon : ScreeningDecision(paper_id=paper_id, decider="llm", **judgment.model_dump())
```

### Vérification

```bash
python -m pytest -v -k "todo_4_"
```

## TODO 5 - Vérifier les preuves

### Objectif

Un LLM peut citer une phrase qui n'existe pas (`G07` : « we use GPT-4 to rank
the records »). Une décision fondée sur une preuve inventée ne vaut rien. Trois
règles s'appliquent :

1. une citation introuvable dans le titre ou le résumé est retirée ;
2. une seule citation inventée suffit : la décision devient `uncertain`
   (confiance plafonnée à 0.5) ;
3. une inclusion sans aucune preuve devient aussi `uncertain`.

Une exclusion sans citation reste possible : l'absence d'un critère ne se cite
pas.

### Fonction à compléter

```python
def verify_evidence(decision, paper) -> ScreeningDecision:
    text = normalize(paper_text(paper))              # fourni
    checked = decision.model_copy(deep=True)         # fourni : on travaille sur une copie
    # TODO 5.1 - kept / invented, puis checked.evidence = kept
    # TODO 5.2 - citation inventée -> "uncertain", confiance min(…, 0.5), raison
    # TODO 5.3 - inclusion sans preuve -> même rétrogradation, raison "inclusion sans preuve"
    # Retourner checked
```

`normalize()` (fourni) met en minuscules et compacte les espaces des deux
côtés de la comparaison.

### Vérification

```bash
python -m pytest -v -k "todo_5_"
```

### Erreurs fréquentes

- Modifier `decision` au lieu de `checked` : le test vérifie que la décision
  reçue reste intacte.
- Comparer sans `normalize` : une différence de casse ou d'espace rejette une
  citation exacte.

---

# Partie C - Décider, s'abstenir, escalader

## TODO 6 - Agir ou s'abstenir : la politique de seuil

### Objectif

Un avis devient une décision automatique seulement s'il est assez sûr. Comme
exclure à tort coûte plus cher, une exclusion exige une confiance plus élevée.

| # | Avis | Issue |
|---|---|---|
| 1 | `uncertain` | `humain` (abstention) |
| 2 | `include` | `auto` si confiance ≥ seuil, sinon `humain` |
| 3 | `exclude` | `auto` si confiance ≥ min(1.0, seuil + `EXCLUDE_MARGIN`), sinon `humain` |

### Fonction à compléter

```python
def route(decision, threshold=DEFAULT_THRESHOLD) -> Literal["auto", "humain"]:
    # TODO 6 - Règle 1, Règle 2, Règle 3
```

### Vérification

```bash
python -m pytest -v -k "todo_6_"
```

## TODO 7 - Deux évaluateurs : accord ou escalade

### Objectif

En revue systématique, deux relecteurs examinent chaque article ; un désaccord
est arbitré. On applique la même règle aux règles et au LLM : on **agit**
seulement si les deux avis sont **identiques** et **sûrs**. Sinon, on escalade
avec un motif (`escalation_reason`, fourni) :

- désaccord ;
- abstention des deux évaluateurs ;
- confiance insuffisante.

### Fonction à compléter

```python
def combine(paper, rules, llm, threshold=DEFAULT_THRESHOLD) -> ScreeningRecord:
    # TODO 7.1 - accord ET route(rules) == route(llm) == "auto" -> décision finale "accord",
    #            confiance = min des deux, preuves réunies, route="auto"
    # TODO 7.2 - sinon -> route="humain", escalation=escalation_reason(rules, llm, threshold), sans final
```

`screen_papers` (fourni) applique `rule_screen`, `llm_screen` et `combine` à
chaque article.

### Vérification

```bash
python -m pytest -v -k "todo_7_"
python main.py --llm demo --etape jeu
```

Attendu : le tableau du jeu étiqueté s'affiche. Six articles sont décidés
automatiquement (`G01`, `G02`, `G03`, `G04`, `G11`, `G13`) ; les neuf autres
vont à l'humain, chacun avec son motif. L'exécution s'arrête ensuite sur
« Non disponible : TODO 8 ».

## TODO 8 - Mesurer les erreurs

### Objectif

On compare les décisions **automatiques** aux étiquettes humaines. Une fausse
exclusion est une étude pertinente perdue ; une fausse inclusion est un article
à lire en trop.

### Fonction à compléter

```python
def evaluate(state, gold) -> dict:
    auto = [...]                                     # fourni : décisions automatiques
    # TODO 8 - false_exclusions : auto "exclude" alors que gold dit "include" ;
    #          false_inclusions : auto "include" alors que gold dit "exclude".
    # (le dictionnaire de résultats est fourni)
```

`compare_policies` (fourni) compare trois politiques :

- les règles seules, sans humain ;
- le LLM seul, sans humain ;
- la double évaluation avec humain.

Sans humain, un article incertain n'est pas retenu : il devient une exclusion.
`sweep_thresholds` (fourni) rejoue la politique pour plusieurs seuils, sans
rappeler le LLM.

### Vérification

```bash
python -m pytest -v -k "todo_8_"
python main.py --llm demo --etape jeu
```

### Point de contrôle C

Lisez les deux tableaux affichés et répondez :

- Combien d'études pertinentes chaque politique sans humain perd-elle ?
- Quel est le prix de la double évaluation (charge humaine) ?
- Pourquoi le seuil 0.5 fait-il perdre `G15` alors que le seuil 0.75 l'escalade ?

---

# Partie D - Boucler avec l'humain

## TODO 9 - Intégrer les décisions humaines

### Objectif

La file humaine n'est utile que si ses décisions reviennent dans l'état.

Deux garde-fous :

- une revue ne remplace jamais une décision automatique déjà prise ;
- « incertain » n'est pas une décision humaine.

Toutes les revues sont vérifiées **avant** d'en appliquer une seule.

### Fonction à compléter

```python
def apply_human_review(state, reviews: dict[str, str]) -> int:
    queue = {r.paper_id: r for r in state.human_queue()}     # fourni
    # TODO 9.1 - vérifier toutes les revues (ValueError si hors file ou choix invalide)
    # TODO 9.2 - appliquer : final = ScreeningDecision(..., confidence=1.0,
    #            reasons=[f"décision humaine ({escalation})"], decider="humain")
    # TODO 9.3 - retourner le nombre de décisions appliquées
```

### Vérification

```bash
python -m pytest -v -k "todo_9_"
python main.py --llm demo --etape sujet --accepter --rq RQ1
```

Répondez `i` pour le premier article de la file, `e` pour le second, Entrée
pour le troisième. Le bilan affiche 2 décisions humaines appliquées et
1 article en attente.

## TODO 10 - Choisir le seuil à partir des données

### Objectif

Le seuil n'est pas une constante magique : on le choisit sur le jeu étiqueté.
Critère : **aucune fausse exclusion**, puis la charge humaine la plus faible. À
charge égale, on garde le seuil le plus élevé, le plus prudent.

### Fonction à compléter

```python
def choose_threshold(rows: list[dict]) -> float | None:
    # TODO 10 - Garder les lignes sans fausse exclusion ; aucune -> None.
    #           Parmi elles, la plus petite charge "humain" ; à égalité, le "seuil" le plus élevé.
```

### Vérification

```bash
python -m pytest -v -k "todo_10_"
python main.py --llm demo --etape jeu
```

Attendu : « Seuil recommandé : 0.75 ».

---

## 6. Validation finale

```bash
python -m pytest -v
python main.py
python -m streamlit run streamlit_app.py
```

Résultat attendu :

```text
49 passed
```

## 7. Résultats attendus (LLM de démonstration)

Jeu étiqueté, seuil 0.75 :

| Politique | Auto | Humain | Fausses exclusions | Fausses inclusions |
|---|---|---|---|---|
| règles seules, sans humain | 15 | 0 | 4 | 1 |
| LLM seul, sans humain | 15 | 0 | 4 | 0 |
| double évaluation + humain | 6 | 9 | **0** | **0** |

Effet du seuil :

| Seuil | Auto | Humain | Fausses exclusions |
|---|---|---|---|
| 0.5 | 7 | 8 | 1 (`G15`) |
| 0.6 | 7 | 8 | 1 (`G15`) |
| 0.75 | 6 | 9 | 0 |
| 0.9 | 1 | 14 | 0 |

Sujet de référence, RQ1, recherche `perturbe` niveau 2 (4 articles) :

| Article | Règles | LLM | Issue |
|---|---|---|---|
| `W1001` | include 0.90 | include 0.90 | automatique : inclusion |
| `W1002` | uncertain 0.50 | include 0.65 | humain : désaccord |
| `crossref:…2022.207` | uncertain 0.50 | exclude 0.80 | humain : désaccord |
| `s2-7f3a` | uncertain 0.00 | uncertain 0.30 | humain : abstention (résumé absent) |

## 8. Erreurs fréquentes

| Symptôme | Cause probable |
|---|---|
| `NotImplementedError: TODO n` | le TODO n n'est pas terminé |
| « LLM » trouvé dans « LLMOps » | `\b` oublié ou pattern non utilisé (TODO 2) |
| `G03` inclus | inclusion testée avant l'exclusion (TODO 3) |
| le programme plante sur `G09` | `parse_judgment` laisse passer la `ValueError` (TODO 4) |
| `G07` décidé automatiquement | citation inventée non détectée (TODO 5) |
| `G15` exclu automatiquement au seuil 0.75 | marge d'exclusion oubliée (TODO 6) |
| `G06` inclus automatiquement | désaccord non escaladé (TODO 7) |
| `faux_exclus` et `faux_inclus` inversés | comparaison avec la mauvaise étiquette (TODO 8) |
| une revue humaine écrase une décision automatique | vérification de la file oubliée (TODO 9) |
| « Seuil recommandé : aucun » | TODO 10 non terminé |
| mode réel : tout part à l'humain | résumés indisponibles, LLM injoignable ou hors contrat : l'agent s'abstient, c'est voulu |

## 9. Questions de synthèse

1. Pourquoi le screening est-il une décision sous incertitude, et non une simple
   classification ?
2. Pourquoi une décision doit-elle porter des raisons et des preuves, et pas
   seulement un libellé ?
3. Dans les règles, la confiance est fixée ; dans le LLM, elle est déclarée.
   Est-ce une probabilité ? Qu'en déduire pour le choix du seuil ?
4. Quelles erreurs commet chaque évaluateur (règles, LLM) sur le jeu étiqueté ?
   Pourquoi sont-ils complémentaires ?
5. Pourquoi une citation inventée invalide-t-elle toute la décision du LLM ?
6. Pourquoi une exclusion automatique exige-t-elle une confiance plus élevée
   qu'une inclusion ?
7. Quelle différence faites-vous entre s'abstenir et escalader ?
8. Pourquoi agir seulement en cas d'accord des deux évaluateurs ? Quel est le
   coût de ce choix ?
9. Que montre la comparaison « automatisation complète » vs « humain dans la
   boucle » ? Que faudrait-il pour réduire la charge humaine sans perdre d'études ?
10. Pourquoi choisir le seuil sur un jeu étiqueté plutôt qu'au jugé ? Quelle est
    la limite d'un jeu de 15 articles ?
11. Pourquoi une revue humaine ne doit-elle jamais écraser silencieusement une
    décision automatique ?
12. Quels articles le TD05 devra-t-il télécharger, et pourquoi pas seulement
    les articles inclus ?
