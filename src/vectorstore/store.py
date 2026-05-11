"""
src/vectorstore/store.py
════════════════════════
Gestion du Vector Store — SmartSupport RAG

Rôle :
    Stocker les chunks vectorisés dans ChromaDB
    et les retrouver lors d'une question utilisateur.

    Phase 1 — Indexation :
        Reçoit les chunks + vecteurs → les stocke dans ChromaDB
        ChromaDB persiste sur disque → survit aux redémarrages

    Phase 2 — Recherche :
        Reçoit une question vectorisée → ChromaDB compare
        → Retourne les k chunks les plus proches sémantiquement

Pourquoi ChromaDB ?
    - Persistance automatique sur disque
    - Recherche par similarité cosinus rapide
    - Stocke texte + vecteurs + métadonnées ensemble
    - Compatible LangChain natif

Auteur : Oumou Kanfana
"""

import os
import yaml
from pathlib import Path
from typing import List, Tuple, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.embedding.embedder import get_embedding_model


# =============================================================
# Chargement de la configuration
# =============================================================

def load_config() -> dict:
    """
    Charge la configuration depuis config/config.yaml.

    Returns:
        Dictionnaire de configuration

    Raises:
        FileNotFoundError : si config.yaml est introuvable
    """
    possible_paths = [
        "config/config.yaml",
        "../config/config.yaml",
        "/workspaces/smartsupport-rag/config/config.yaml",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)

    raise FileNotFoundError("config/config.yaml introuvable.")


# =============================================================
# Gestion du Vector Store
# =============================================================

# Instance globale — initialisée une seule fois
# Évite de recharger ChromaDB à chaque appel
_vector_store: Optional[Chroma] = None


def get_vector_store(force_reload: bool = False) -> Chroma:
    """
    Initialise ou retourne l'instance ChromaDB existante.

    Utilise le pattern Singleton — ChromaDB est chargé une seule fois
    et réutilisé pour tous les appels suivants.

    Args:
        force_reload : si True, recrée l'instance même si elle existe

    Returns:
        Instance Chroma configurée et prête à l'emploi

    Exemple:
        >>> store = get_vector_store()
        >>> print(store._collection.name)  # smartsupport_docs
    """
    global _vector_store

    # Retourne l'instance existante si disponible
    if _vector_store is not None and not force_reload:
        return _vector_store

    # Chargement de la configuration
    config = load_config()
    chroma_config = config["vector_store"]["chroma"]

    collection_name  = chroma_config["collection_name"]
    persist_directory = chroma_config["persist_directory"]

    # Création du répertoire de persistance si inexistant
    Path(persist_directory).mkdir(parents=True, exist_ok=True)

    # Initialisation du modèle d'embeddings
    embeddings = get_embedding_model()

    # Initialisation de ChromaDB
    # Si la collection existe déjà → charge les données existantes
    # Si elle n'existe pas → crée une nouvelle collection vide
    _vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    print(f"[Store] ✅ ChromaDB initialisé — collection '{collection_name}'")
    print(f"[Store] 📁 Persistance : {persist_directory}")

    return _vector_store


def add_documents(chunks: List[Document]) -> int:
    """
    Indexe une liste de chunks dans ChromaDB.

    Pour chaque chunk :
    1. ChromaDB appelle get_embedding_model() pour vectoriser
    2. Stocke le vecteur + le texte + les métadonnées
    3. Persiste sur disque automatiquement

    Args:
        chunks : liste de Documents LangChain sortis du chunker

    Returns:
        Nombre de chunks indexés

    Exemple:
        >>> n = add_documents(chunks)
        >>> print(f"{n} chunks indexés")
    """
    if not chunks:
        print("[Store] ⚠️ Aucun chunk à indexer")
        return 0

    store = get_vector_store()

    # Ajout des chunks dans ChromaDB
    # ChromaDB gère lui-même la vectorisation via embedding_function
    store.add_documents(chunks)

    print(f"[Store] ✅ {len(chunks)} chunks indexés dans ChromaDB")

    return len(chunks)


def similarity_search(
    query: str,
    k: int = 4
) -> List[Tuple[Document, float]]:
    """
    Recherche les k chunks les plus proches sémantiquement.

    Processus :
    1. Vectorise la question (via embedding_function)
    2. Calcule la similarité cosinus avec tous les vecteurs stockés
    3. Retourne les k chunks avec les scores les plus élevés

    Args:
        query : question posée par l'utilisateur
        k     : nombre de chunks à retourner (défaut : 4 depuis config)

    Returns:
        Liste de tuples (Document, score)
        - Document : le chunk retrouvé avec son texte et métadonnées
        - score    : score de similarité entre 0.0 et 1.0
                     Plus proche de 1.0 = plus pertinent

    Exemple:
        >>> results = similarity_search("Comment réinitialiser un mot de passe ?")
        >>> for doc, score in results:
        ...     print(f"Score: {score:.2f} — {doc.page_content[:100]}")
    """
    store = get_vector_store()

    # Recherche avec scores de similarité
    # similarity_search_with_relevance_scores retourne des scores
    # normalisés entre 0 et 1 (contrairement aux distances brutes)
    results = store.similarity_search_with_relevance_scores(query, k=k)

    print(f"[Store] 🔍 {len(results)} chunks retrouvés pour : '{query[:50]}...'")

    return results


def get_collection_stats() -> dict:
    """
    Retourne les statistiques de la collection ChromaDB.

    Utile pour vérifier combien de chunks sont indexés
    et diagnostiquer les problèmes d'indexation.

    Returns:
        Dictionnaire avec les stats de la collection

    Exemple:
        >>> stats = get_collection_stats()
        >>> print(f"{stats['count']} chunks dans ChromaDB")
    """
    store = get_vector_store()

    # Accès direct à la collection ChromaDB sous-jacente
    collection = store._collection
    count = collection.count()

    config = load_config()
    chroma_config = config["vector_store"]["chroma"]

    return {
        "collection_name":  chroma_config["collection_name"],
        "chunks_indexed":   count,
        "persist_directory": chroma_config["persist_directory"],
        "embedding_model":  "text-embedding-3-small",
        "dimensions":       1536,
    }


def reset_collection() -> None:
    """
    Supprime tous les chunks de la collection ChromaDB.

    Utile pour réindexer proprement depuis zéro
    quand les documents ont changé.

    ⚠️ Action irréversible — tous les chunks sont supprimés.

    Exemple:
        >>> reset_collection()
        >>> print("Collection vidée — prêt pour réindexation")
    """
    global _vector_store

    store = get_vector_store()
    collection = store._collection

    # Récupération de tous les IDs pour suppression
    all_ids = collection.get()["ids"]

    if all_ids:
        collection.delete(ids=all_ids)
        print(f"[Store] 🗑️ {len(all_ids)} chunks supprimés de ChromaDB")
    else:
        print("[Store] ℹ️ Collection déjà vide")

    # Reset de l'instance globale
    _vector_store = None