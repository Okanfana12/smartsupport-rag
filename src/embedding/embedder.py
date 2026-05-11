"""
src/embedding/embedder.py
═════════════════════════
Vectorisation des documents et questions — SmartSupport RAG

Rôle :
    Transformer le texte en vecteurs numériques (embeddings)
    pour permettre la recherche sémantique dans ChromaDB.

    Un embedding = représentation mathématique du sens d'un texte.
    Deux textes sémantiquement proches → vecteurs proches.

Modèle choisi :
    OpenAI text-embedding-3-small
    - 1536 dimensions
    - Meilleur rapport qualité/coût OpenAI
    - 5x moins cher que ada-002 avec de meilleures performances

Coût estimé pour ce projet :
    248 chunks × ~250 chars ≈ 15,500 tokens
    → 15,500 / 1,000,000 × 0.02$ ≈ 0.0003$ (moins d'1 centime)

Auteur : Oumou Kanfana
"""

import os
from typing import List
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Chargement des variables d'environnement (.env)
# Nécessaire pour récupérer OPENAI_API_KEY
load_dotenv()


# =============================================================
# Constantes — lues depuis config ou valeurs par défaut
# =============================================================

# Modèle d'embeddings OpenAI
# text-embedding-3-small : meilleur rapport qualité/coût
EMBEDDING_MODEL = "text-embedding-3-small"

# Nombre de dimensions du vecteur
# 1536 = standard pour text-embedding-3-small
EMBEDDING_DIMENSIONS = 1536

# Taille des batches pour l'envoi à l'API OpenAI
# OpenAI accepte jusqu'à 2048 inputs par requête
# On reste à 100 pour éviter les timeouts
BATCH_SIZE = 100


# =============================================================
# Fonctions principales
# =============================================================

def get_embedding_model() -> OpenAIEmbeddings:
    """
    Initialise et retourne le modèle d'embeddings OpenAI.

    Lit la clé API depuis la variable d'environnement OPENAI_API_KEY.
    Le modèle est partagé par embed_documents() et embed_query()
    pour éviter de le réinitialiser à chaque appel.

    Returns:
        Instance OpenAIEmbeddings configurée

    Raises:
        ValueError : si OPENAI_API_KEY n'est pas définie dans .env

    Exemple:
        >>> model = get_embedding_model()
        >>> print(model.model)  # text-embedding-3-small
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY non trouvée.\n"
            "Vérifiez votre fichier .env : OPENAI_API_KEY=sk-proj-..."
        )

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=api_key,
        dimensions=EMBEDDING_DIMENSIONS,
    )


def embed_documents(chunks: List[Document]) -> List[List[float]]:
    """
    Vectorise une liste de chunks (Documents LangChain).

    Envoie les chunks à l'API OpenAI par batches de BATCH_SIZE
    pour éviter les timeouts sur de grands volumes.

    Args:
        chunks : liste de Documents LangChain sortis du chunker

    Returns:
        Liste de vecteurs — 1 vecteur par chunk
        Chaque vecteur = liste de 1536 nombres flottants

    Exemple:
        >>> chunks = chunk_documents(cleaned_docs)
        >>> vectors = embed_documents(chunks)
        >>> print(len(vectors))        # 248 (1 par chunk)
        >>> print(len(vectors[0]))     # 1536 (dimensions)
        >>> print(type(vectors[0][0])) # float
    """
    if not chunks:
        return []

    model = get_embedding_model()

    # Extraction du texte de chaque chunk
    # On vectorise uniquement le contenu textuel — pas les métadonnées
    texts = [chunk.page_content for chunk in chunks]

    all_vectors = []

    # Traitement par batches pour éviter les timeouts API
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]

        # Appel à l'API OpenAI pour vectoriser le batch
        batch_vectors = model.embed_documents(batch)
        all_vectors.extend(batch_vectors)

        print(
            f"[Embedder] Batch {i // BATCH_SIZE + 1}/"
            f"{(len(texts) - 1) // BATCH_SIZE + 1} "
            f"— {len(batch)} chunks vectorisés"
        )

    print(
        f"[Embedder] ✅ {len(all_vectors)} vecteurs générés "
        f"({EMBEDDING_DIMENSIONS} dimensions chacun)"
    )

    return all_vectors


def embed_query(query: str) -> List[float]:
    """
    Vectorise une question utilisateur pour la recherche RAG.

    Utilise le même modèle que embed_documents() pour garantir
    que les vecteurs sont comparables dans le même espace vectoriel.

    Important : on utilise embed_query() et non embed_documents()
    car OpenAI optimise différemment les embeddings selon l'usage
    (documents longs vs questions courtes).

    Args:
        query : question posée par l'utilisateur en langage naturel

    Returns:
        Vecteur de 1536 nombres représentant le sens de la question

    Exemple:
        >>> vector = embed_query("Comment réinitialiser un mot de passe ?")
        >>> print(len(vector))  # 1536
    """
    if not query or not query.strip():
        raise ValueError("La question ne peut pas être vide.")

    model = get_embedding_model()

    # embed_query() optimisé pour les requêtes courtes
    vector = model.embed_query(query)

    print(f"[Embedder] ✅ Question vectorisée ({len(vector)} dimensions)")

    return vector


def get_embedding_info() -> dict:
    """
    Retourne les informations sur le modèle d'embeddings.

    Utile pour la documentation et le debugging.

    Returns:
        Dictionnaire avec les infos du modèle
    """
    return {
        "model":      EMBEDDING_MODEL,
        "dimensions": EMBEDDING_DIMENSIONS,
        "provider":   "OpenAI",
        "cost_per_million_tokens": 0.02,
        "batch_size": BATCH_SIZE,
    }