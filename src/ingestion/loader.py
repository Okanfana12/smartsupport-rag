"""
src/ingestion/loader.py
═══════════════════════
Loader universel — SmartSupport RAG

Rôle :
    Détecter le format d'un fichier et charger son contenu
    en utilisant le loader LangChain approprié.
    Retourne toujours List[Document] — même interface pour tous les formats.

Formats supportés :
    PDF, DOCX, CSV, JSON, HTML, TXT, PPTX, Email (.eml, .txt)

Choix technique JSON :
    On utilise json natif Python au lieu de JSONLoader (qui requiert jq)
    Plus simple, pas de dépendance externe, même résultat.

Auteur : Oumou Kanfana
"""

import os
import json
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,                    # PDF  — extrait texte page par page
    Docx2txtLoader,                 # DOCX — extrait texte des paragraphes Word
    CSVLoader,                      # CSV  — une ligne = un Document
    BSHTMLLoader,                   # HTML — retire les balises, garde le texte
    TextLoader,                     # TXT  — lit le texte brut ligne par ligne
    UnstructuredPowerPointLoader,   # PPTX — une slide = un Document
    UnstructuredEmailLoader,        # EML  — extrait sujet + corps
)


# =============================================================
# Mapping extension → loader LangChain
# JSON est géré séparément via json natif Python (pas de jq requis)
# =============================================================
LOADER_MAP = {
    ".pdf":  PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".csv":  CSVLoader,
    ".html": BSHTMLLoader,
    ".htm":  BSHTMLLoader,
    ".txt":  TextLoader,
    ".md":   TextLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".eml":  UnstructuredEmailLoader,
}

# Extensions supportées — inclut JSON séparément
SUPPORTED_EXTENSIONS = set(LOADER_MAP.keys()) | {".json"}


def detect_extension(file_path: str) -> str:
    """
    Détecte l'extension d'un fichier et la retourne en minuscules.

    Args:
        file_path : chemin vers le fichier

    Returns:
        Extension en minuscules — ex: ".pdf", ".csv", ".json"

    Raises:
        ValueError : si le fichier n'a pas d'extension
    """
    ext = Path(file_path).suffix.lower()

    if not ext:
        raise ValueError(
            f"Impossible de détecter le format du fichier : {file_path}\n"
            f"Assurez-vous que le fichier a une extension (.pdf, .csv...)"
        )

    return ext


def load_json(file_path: str) -> List[Document]:
    """
    Charge un fichier JSON sans dépendance externe (pas de jq).

    Stratégie :
        - Si le JSON est une liste → 1 Document par élément
        - Si le JSON est un dict  → 1 Document par clé de premier niveau
        - Sinon                   → 1 seul Document avec tout le contenu

    Args:
        file_path : chemin vers le fichier JSON

    Returns:
        List[Document] avec le contenu JSON converti en texte lisible
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []

    if isinstance(data, list):
        # JSON tableau → 1 Document par élément
        # Ex: [{"ticket": 1, ...}, {"ticket": 2, ...}]
        for i, item in enumerate(data):
            content = json.dumps(item, ensure_ascii=False, indent=2)
            documents.append(Document(
                page_content=content,
                metadata={"source": file_path, "index": i}
            ))

    elif isinstance(data, dict):
        # JSON objet → 1 Document par clé de premier niveau
        # Ex: {"plans": {...}, "politique": {...}}
        for key, value in data.items():
            content = f"{key}:\n{json.dumps(value, ensure_ascii=False, indent=2)}"
            documents.append(Document(
                page_content=content,
                metadata={"source": file_path, "key": key}
            ))

    else:
        # Autre type → 1 seul Document
        documents.append(Document(
            page_content=str(data),
            metadata={"source": file_path}
        ))

    return documents


def get_loader(file_path: str, ext: str):
    """
    Retourne le loader LangChain approprié selon l'extension.

    JSON est traité séparément via load_json() — retourne None
    pour signaler à load_file() d'utiliser load_json() directement.

    Args:
        file_path : chemin vers le fichier
        ext       : extension du fichier

    Returns:
        Instance du loader LangChain ou None si JSON

    Raises:
        ValueError : si le format n'est pas supporté
    """
    # JSON — retourne None pour utiliser load_json() dans load_file()
    if ext == ".json":
        return None

    # Autres formats — lookup dans LOADER_MAP
    loader_class = LOADER_MAP.get(ext)

    if loader_class is None:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Format non supporté : {ext}\n"
            f"Formats acceptés : {supported}"
        )

    return loader_class(file_path)


def add_metadata(documents: List[Document], file_path: str) -> List[Document]:
    """
    Enrichit chaque Document avec des métadonnées sur le fichier source.

    Les métadonnées permettent de citer la source dans les réponses RAG.
    Ex: "Source : FAQ_Support_Client.pdf — Page 3"

    Args:
        documents : liste de Documents chargés
        file_path : chemin vers le fichier source

    Returns:
        Documents enrichis avec les métadonnées
    """
    path = Path(file_path)

    for i, doc in enumerate(documents):
        # On ne remplace pas les métadonnées existantes — on enrichit
        doc.metadata["filename"]  = path.name            # FAQ_Support.pdf
        doc.metadata["file_path"] = str(path)            # /data/pdf_files/FAQ...
        doc.metadata["file_type"] = path.suffix.lower()  # .pdf
        doc.metadata["doc_index"] = i                    # position dans le fichier

    return documents


def load_file(file_path: str) -> List[Document]:
    """
    Charge un fichier et retourne une liste de Documents LangChain.

    C'est la fonction principale du module — un seul appel
    pour tous les formats supportés.

    Args:
        file_path : chemin vers le fichier à charger

    Returns:
        List[Document] — même format quelle que soit la source

    Raises:
        FileNotFoundError : si le fichier n'existe pas
        ValueError        : si le format n'est pas supporté

    Exemple:
        >>> docs = load_file("data/csv_files/historique_tickets_2026.csv")
        >>> print(len(docs))
        >>> print(docs[0].page_content[:200])
        >>> print(docs[0].metadata)
    """
    # Vérification que le fichier existe
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Fichier introuvable : {file_path}\n"
            f"Vérifiez le chemin et réessayez."
        )

    # Détection du format
    ext = detect_extension(file_path)

    # ── Cas spécial JSON — chargement natif Python ────────────
    if ext == ".json":
        documents = load_json(file_path)
        documents = add_metadata(documents, file_path)
        print(f"[Loader] ✅ {len(documents)} document(s) chargé(s) depuis '{Path(file_path).name}'")
        return documents

    # ── Autres formats — loader LangChain ────────────────────
    loader = get_loader(file_path, ext)
    documents = loader.load()
    documents = add_metadata(documents, file_path)

    print(f"[Loader] ✅ {len(documents)} document(s) chargé(s) depuis '{Path(file_path).name}'")

    return documents


def load_directory(directory_path: str) -> List[Document]:
    """
    Charge tous les fichiers supportés d'un dossier.

    Parcourt le dossier, détecte les formats supportés
    et charge chaque fichier automatiquement.
    Les erreurs sur un fichier n'arrêtent pas le traitement des autres.

    Args:
        directory_path : chemin vers le dossier

    Returns:
        List[Document] — tous les documents du dossier combinés

    Raises:
        FileNotFoundError : si le dossier n'existe pas

    Exemple:
        >>> docs = load_directory("data/csv_files/")
        >>> print(f"{len(docs)} documents chargés")
    """
    directory = Path(directory_path)

    if not directory.exists():
        raise FileNotFoundError(
            f"Dossier introuvable : {directory_path}"
        )

    all_documents = []

    # Parcours de tous les fichiers du dossier par ordre alphabétique
    for file_path in sorted(directory.iterdir()):

        # Ignorer les sous-dossiers
        if not file_path.is_file():
            continue

        # Ignorer les formats non supportés
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            docs = load_file(str(file_path))
            all_documents.extend(docs)
        except Exception as e:
            # On log l'erreur mais on continue avec les autres fichiers
            print(f"[Loader] ⚠️ Erreur sur '{file_path.name}' : {e}")
            continue

    print(f"[Loader] 📁 Total : {len(all_documents)} document(s) depuis '{directory.name}'")

    return all_documents