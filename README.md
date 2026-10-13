# SLRResearchAI - TD04 : ScreeningAgent

Le TD04 prolonge le TD03 : les articles trouvés à partir de votre sujet sont
triés sur leur titre et leur résumé, sous incertitude.

```text
sujet → TD02 (concepts validés = critères d'inclusion) → TD03 (articles) → résumés
      → deux évaluateurs : règles du protocole · LLM contrôlé (preuves vérifiées)
      → agir · s'abstenir · escalader vers un humain
      → erreurs mesurées sur un jeu étiqueté
```

Fichier à compléter : `slrresearch/screening_agent.py` (TODO 1 à 10).
Lisez d'abord `ENONCE_TD04.md`.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env      # seulement pour le mode « LLM réel »
```

## Tests

```bash
python -m pytest -v -k "fourni"     # code fourni : vert dès le départ
python -m pytest -v -k "todo_1_"    # un TODO
python -m pytest -v                 # validation complète
```

Résultat final attendu : `49 passed`.

## Exécution

```bash
python main.py                                            # mode guidé
python main.py --llm demo --etape jeu                     # mesurer la politique
python main.py --llm demo --etape sujet --accepter --rq RQ1
python -m streamlit run streamlit_app.py
```

Le mode `demo` fonctionne sans clé ni réseau. Le mode `reel` utilise le
fournisseur LLM de `.env` ; avec `--sources reel`, la recherche interroge les
API ouvertes et les résumés viennent d'OpenAlex.
