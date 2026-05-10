# SmartSupport RAG

> Plateforme RAG modulaire pour l'automatisation du support client.  
> Transforme la documentation interne et l'historique des tickets en assistant IA capable de répondre instantanément aux questions des agents support, avec sources citées et score de confiance explicite.

---

## Problème résolu

Les équipes support passent en moyenne **40% de leur temps** à rechercher manuellement des réponses dans :
- Des FAQs dispersées en plusieurs documents
- Des milliers de tickets résolus non structurés
- Des manuels utilisateurs de plusieurs centaines de pages
- Des échanges emails archivés

**SmartSupport RAG** centralise toutes ces sources et permet à un agent de poser une question en langage naturel pour obtenir une réponse précise, sourcée et fiable en quelques secondes.

---

## Principe fondamental

> Le système ne devine jamais.  
> Si la réponse n'est pas dans les documents, il le dit explicitement.  
> Chaque réponse est accompagnée de ses sources et d'un score de confiance.

---

## Architecture

```
Sources de données
├── FAQ produit (PDF)
├── Tickets résolus (CSV)
├── Manuel utilisateur (DOCX)
├── Procédures internes (HTML)
└── Échanges support (Email)
          │
          ▼
┌─────────────────────────┐
│   Ingestion             │
│   loader.py             │  Détecte le format → charge → retourne Documents
│   cleaner.py            │  Nettoie → normalise → filtre les vides
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Chunking              │
│   chunker.py            │  Découpe intelligemment selon le type de document
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Embedding             │
│   embedder.py           │  Vectorise les chunks — OpenAI ou HuggingFace local
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Vector Store          │
│   store.py              │  Indexe et persiste — FAISS (dev) / ChromaDB (prod)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   RAG                   │
│   retriever.py          │  Recherche les chunks pertinents
│   generator.py          │  Génère la réponse avec le LLM
│   confidence.py         │  Score de confiance + fallback si insuffisant
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Évaluation            │
│   metrics.py            │  RAGAS — mesure la qualité en continu
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Interface             │
│   app.py                │  Streamlit — interface agent support
└─────────────────────────┘
```

---

## Stack technique

| Composant | Technologie | Justification |
|---|---|---|
| Orchestration | LangChain + LangGraph + LCEL | Orchestration robuste et flexible |
| LLM principal | Claude 3 Sonnet | Excellent sur documents longs, 200K contexte |
| LLM rapide | Claude 3 Haiku | Réponses courtes, faible latence |
| LLM souverain | Mistral 7B via Ollama | Données sensibles, on-premise |
| Embeddings | text-embedding-3-small | Meilleur rapport qualité/coût |
| Vector store dev | FAISS | Simple, sans installation |
| Vector store prod | ChromaDB | Persistance, scalabilité |
| Évaluation | RAGAS + LangSmith | Métriques qualité en continu |
| Interface | Streamlit | Démo rapide et intuitive |
| Conteneurisation | Docker + Docker Compose | Déploiement reproductible |
| CI/CD | GitHub Actions | Tests automatisés |

---

## Sources de données supportées

| Format | Type | Contenu typique |
|---|---|---|
| **PDF** | Document | FAQ produit, politique support |
| **DOCX** | Document | Manuel utilisateur, procédures |
| **CSV** | Données | Historique tickets résolus |
| **JSON** | Données | Base de connaissances structurée |
| **HTML** | Web | Documentation en ligne, procédures internes |
| **EML** | Email | Échanges support archivés |
| **TXT** | Texte | Notes internes, logs |

---

## Chunking adaptatif

Chaque type de document est découpé différemment selon sa nature :

| Type | Chunk size | Overlap | Raison |
|---|---|---|---|
| PDF FAQ | 600 tokens | 100 | Questions/réponses courtes |
| DOCX manuel | 800 tokens | 150 | Texte dense et structuré |
| CSV tickets | 300 tokens | 50 | Entrées courtes et indépendantes |
| HTML procédures | 500 tokens | 100 | Contenu mixte |
| Email | 400 tokens | 80 | Messages courts |

---

## Choix LLM selon le contexte

| Situation | Modèle | Pourquoi |
|---|---|---|
| Question sur document long | Claude 3 Sonnet | 200K tokens de contexte |
| Réponse rapide et courte | Claude 3 Haiku | Rapide et économique |
| Données confidentielles | Mistral 7B local | Aucune donnée externe |
| Évaluation qualité | GPT-4o | Meilleur raisonnement |

---

## Scoring de confiance

Chaque réponse est accompagnée d'un score entre 0 et 100% calculé sur :

- La similarité sémantique entre la question et les chunks retrouvés
- Le nombre de sources concordantes disponibles
- La cohérence interne de la réponse générée

**Comportement selon le score :**

| Score | Comportement |
|---|---|
| >= 75% | Réponse affichée avec sources |
| 50-75% | Réponse affichée avec avertissement |
| < 50% | Fallback — message d'incertitude explicite |

---

## Métriques d'évaluation RAGAS

| Métrique | Ce qu'elle mesure |
|---|---|
| **Faithfulness** | La réponse est-elle fidèle aux documents sources ? |
| **Answer Relevancy** | La réponse répond-elle vraiment à la question posée ? |
| **Context Precision** | Les chunks retrouvés sont-ils tous pertinents ? |
| **Context Recall** | Toutes les informations nécessaires ont-elles été retrouvées ? |

---

## Structure du projet

```
smartsupport-rag/
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py          # Loader universel — PDF, DOCX, CSV, JSON, HTML, EML
│   │   └── cleaner.py         # Nettoyage et normalisation des documents
│   ├── chunking/
│   │   ├── __init__.py
│   │   └── chunker.py         # Chunking adaptatif selon le type de document
│   ├── embedding/
│   │   ├── __init__.py
│   │   └── embedder.py        # OpenAI text-embedding-3-small / HuggingFace
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   └── store.py           # FAISS (dev) / ChromaDB (prod)
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── retriever.py       # Recherche vectorielle top-k
│   │   ├── generator.py       # Génération réponse + prompt engineering
│   │   └── confidence.py      # Score de confiance + logique fallback
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py         # RAGAS + monitoring LangSmith
├── interface/
│   └── app.py                 # Interface Streamlit agent support
├── config/
│   └── config.yaml            # Paramètres LLM, chunking, seuils
├── data/
│   ├── pdf_files/             # FAQ, politiques support
│   ├── word_files/            # Manuels utilisateur
│   ├── csv_files/             # Historique tickets
│   ├── json_files/            # Base de connaissances
│   ├── html_files/            # Procédures internes
│   └── email_files/           # Échanges support archivés
├── tests/
│   ├── test_loader.py
│   ├── test_chunker.py
│   └── test_rag.py
├── notebooks/
│   └── exploration.ipynb      # Prototypage et analyse
├── .env.example
├── .gitignore
├── pyproject.toml
├── docker-compose.yml
└── README.md
```

---

## Installation

### Prérequis

- Python 3.11+
- uv — gestionnaire de paquets moderne
- Docker Desktop (optionnel)
- Ollama (optionnel — pour Mistral local)

### 1. Cloner le repo

```bash
git clone https://github.com/oumoukanfana/smartsupport-rag.git
cd smartsupport-rag
```

### 2. Créer et activer l'environnement virtuel

```bash
uv venv .venv
source .venv/bin/activate    # Mac / Linux / Codespaces
.venv\Scripts\activate       # Windows
```

### 3. Installer les dépendances

```bash
uv pip install -r requirements.txt
```

### 4. Configurer les variables d'environnement

```bash
cp .env.example .env
# Renseigner vos clés API dans .env
```

### 5. (Optionnel) Mistral local pour la souveraineté des données

```bash
ollama pull mistral
ollama serve
```

### 6. Lancer l'interface

```bash
streamlit run interface/app.py
```

### 7. Lancer avec Docker

```bash
docker-compose up --build
```

---

## Scénario de démo

**Étape 1** — Un agent support dépose les documents dans l'interface
- FAQ produit (PDF)
- Manuel utilisateur (DOCX)
- Historique tickets résolus (CSV)

**Étape 2** — Le pipeline indexe automatiquement toutes les sources

**Étape 3** — L'agent pose une question en langage naturel
> "Comment réinitialiser le mot de passe d'un client bloqué ?"

**Étape 4** — Le système retourne
- La réponse précise avec les étapes à suivre
- Les sources citées avec numéro de page ou ligne
- Le score de confiance — ex : 89%
- Un rapport de synthèse si besoin

**Étape 5** — Si le score est insuffisant
> "Je ne trouve pas d'information suffisamment fiable sur ce sujet. Consultez directement le manuel section 4.2 ou escaladez au niveau 2."

---

## Roadmap

- [ ] Connecteur base de données tickets (PostgreSQL)
- [ ] Support multilingue FR / EN / ES
- [ ] Feedback utilisateur pour améliorer le golden dataset
- [ ] Déploiement Kubernetes
- [ ] Authentification agents (SSO)
- [ ] Tableau de bord métriques RAGAS en temps réel

---

## Auteur

**Oumou Kanfana** — Data Scientist & AI Engineer Freelance  
Toulouse | [oumoukanfana.com](https://oumoukanfana.com) | [LinkedIn](https://linkedin.com/in/oumoukanfana)

---

*Projet développé comme démonstration d'architecture RAG industrialisable — données fictives uniquement.*# smartsupport-rag
