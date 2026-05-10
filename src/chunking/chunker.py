"""
src/chunking/chunker.py
═══════════════════════
Chunking adaptatif — SmartSupport RAG

Rôle :
    Découper les Documents nettoyés en chunks de taille optimale
    pour le RAG. Chaque chunk doit représenter une unité sémantique
    cohérente — une question/réponse, un paragraphe, un ticket.

Stratégie :
    - Paramètres lus depuis config/config.yaml
    - chunk_size adapté selon le type de document
    - RecursiveCharacterTextSplitter — respecte la structure du texte
    - Overlap pour préserver le contexte entre chunks

Pourquoi RecursiveCharacterTextSplitter ?
    Coupe dans cet ordre de priorité :
    1. Entre paragraphes (\n\n)  → préserve la structure
    2. Entre lignes (\n)         → préserve les phrases
    3. Entre phrases (". ")      → préserve le sens
    4. Entre mots (" ")          → en dernier recours
    5. Entre caractères ("")     → uniquement si nécessaire

Auteur : Oumou Kanfana
"""

import os
import yaml
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =============================================================
# Chargement de la configuration
# =============================================================

def load_config() -> dict:
    """
    Charge la configuration depuis config/config.yaml.

    Cherche le fichier config.yaml dans :
    1. Le répertoire courant
    2. Le répertoire racine du projet

    Returns:
        Dictionnaire de configuration

    Raises:
        FileNotFoundError : si config.yaml est introuvable
    """
    # Chemins possibles pour config.yaml
    possible_paths = [
        "config/config.yaml",
        "../config/config.yaml",
        "/workspaces/smartsupport-rag/config/config.yaml",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)

    raise FileNotFoundError(
        "config/config.yaml introuvable.\n"
        "Vérifiez que vous êtes bien à la racine du projet."
    )


# =============================================================
# Mapping extension → clé config.yaml
# =============================================================

# Permet de retrouver les bons paramètres dans config.yaml
# selon le type de fichier source
EXTENSION_TO_CONFIG_KEY = {
    ".pdf":  "pdf",
    ".txt":  "txt",
    ".md":   "txt",
    ".docx": "docx",
    ".csv":  "csv",
    ".json": "json",
    ".html": "html",
    ".htm":  "html",
    ".pptx": "pptx",
    ".eml":  "email",
}


def get_chunking_config(file_type: str, config: dict) -> dict:
    """
    Retourne les paramètres de chunking pour un type de fichier.

    Si le type n'est pas reconnu → utilise les valeurs par défaut.

    Args:
        file_type : extension du fichier (.pdf, .csv...)
        config    : dictionnaire de configuration complet

    Returns:
        Dictionnaire avec chunk_size, chunk_overlap, separators
    """
    # Recherche de la clé config correspondante
    config_key = EXTENSION_TO_CONFIG_KEY.get(file_type, "default")

    # Lecture des paramètres depuis config.yaml
    chunking_config = config.get("chunking", {})
    params = chunking_config.get(config_key, chunking_config.get("default", {}))

    return {
        "chunk_size":    params.get("chunk_size", 300),
        "chunk_overlap": params.get("chunk_overlap", 60),
        "separators":    params.get("separators", ["\n\n", "\n", ". ", " ", ""]),
    }


def get_splitter(file_type: str, config: dict) -> RecursiveCharacterTextSplitter:
    """
    Crée et retourne un RecursiveCharacterTextSplitter configuré
    selon le type de fichier.

    Args:
        file_type : extension du fichier (.pdf, .csv...)
        config    : dictionnaire de configuration complet

    Returns:
        Instance de RecursiveCharacterTextSplitter configurée
    """
    params = get_chunking_config(file_type, config)

    return RecursiveCharacterTextSplitter(
        chunk_size=params["chunk_size"],
        chunk_overlap=params["chunk_overlap"],
        separators=params["separators"],
        length_function=len,          # mesure en caractères
        is_separator_regex=False,     # séparateurs littéraux, pas regex
    )


def add_chunk_metadata(
    chunks: List[Document],
    original_doc: Document,
    chunk_start_index: int = 0
) -> List[Document]:
    """
    Enrichit chaque chunk avec des métadonnées sur sa position.

    Les métadonnées permettent de :
    - Citer la source précise dans les réponses RAG
    - Retrouver le contexte d'un chunk
    - Déboguer le pipeline

    Args:
        chunks            : liste de chunks générés
        original_doc      : Document source avant chunking
        chunk_start_index : index de départ pour numéroter les chunks

    Returns:
        Chunks enrichis avec métadonnées de position
    """
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        # Copie des métadonnées du document source
        chunk.metadata.update(original_doc.metadata)

        # Ajout des métadonnées spécifiques au chunk
        chunk.metadata["chunk_index"]  = chunk_start_index + i  # position globale
        chunk.metadata["chunk_total"]  = total_chunks            # total chunks du doc
        chunk.metadata["chunk_size"]   = len(chunk.page_content) # taille réelle
        chunk.metadata["is_chunk"]     = True                    # flag chunké

    return chunks


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Découpe une liste de Documents en chunks optimisés pour le RAG.

    Pour chaque Document :
    1. Détecte le type de fichier depuis les métadonnées
    2. Charge les paramètres de chunking depuis config.yaml
    3. Crée le splitter approprié
    4. Découpe le Document en chunks
    5. Enrichit les chunks avec leurs métadonnées

    C'est la fonction principale du module.

    Args:
        documents : liste de Documents nettoyés sortis du cleaner

    Returns:
        Liste de chunks prêts pour l'embedding

    Exemple:
        >>> from src.ingestion.loader import load_file
        >>> from src.ingestion.cleaner import clean_documents
        >>> from src.chunking.chunker import chunk_documents
        >>> docs    = load_file("data/pdf_files/FAQ_Support.txt")
        >>> cleaned = clean_documents(docs)
        >>> chunks  = chunk_documents(cleaned)
        >>> print(f"{len(chunks)} chunks générés")
        >>> print(chunks[0].page_content[:200])
    """
    if not documents:
        return []

    # Chargement de la configuration une seule fois
    config = load_config()

    all_chunks = []
    chunk_index = 0

    for doc in documents:
        # Détection du type de fichier depuis les métadonnées
        file_type = doc.metadata.get("file_type", ".txt")

        # Création du splitter adapté au type de fichier
        splitter = get_splitter(file_type, config)

        # Découpage du Document en chunks
        chunks = splitter.split_documents([doc])

        # Enrichissement des métadonnées des chunks
        chunks = add_chunk_metadata(chunks, doc, chunk_index)

        all_chunks.extend(chunks)
        chunk_index += len(chunks)

    # Récupération des paramètres utilisés pour le log
    sample_type = documents[0].metadata.get("file_type", ".txt") if documents else ".txt"
    params = get_chunking_config(sample_type, config)

    print(
        f"[Chunker] ✅ {len(all_chunks)} chunks générés "
        f"depuis {len(documents)} document(s) "
        f"(chunk_size={params['chunk_size']}, "
        f"overlap={params['chunk_overlap']})"
    )

    return all_chunks


def chunk_summary(chunks: List[Document]) -> None:
    """
    Affiche un résumé statistique des chunks générés.

    Utile pour vérifier que le chunking est optimal :
    - Distribution des tailles
    - Chunks trop courts ou trop longs
    - Répartition par source

    Args:
        chunks : liste de chunks à analyser
    """
    if not chunks:
        print("[Chunker] Aucun chunk à analyser")
        return

    sizes = [len(c.page_content) for c in chunks]

    print("\n=== RÉSUMÉ CHUNKING ===")
    print(f"Total chunks    : {len(chunks)}")
    print(f"Taille moyenne  : {sum(sizes) // len(sizes)} caractères")
    print(f"Taille min      : {min(sizes)} caractères")
    print(f"Taille max      : {max(sizes)} caractères")

    # Répartition par source
    sources = {}
    for chunk in chunks:
        filename = chunk.metadata.get("filename", "inconnu")
        sources[filename] = sources.get(filename, 0) + 1

    print(f"\nRépartition par source :")
    for filename, count in sorted(sources.items()):
        print(f"  {filename:<50} {count:>4} chunks")