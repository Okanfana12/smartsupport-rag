"""
src/ingestion/cleaner.py
════════════════════════
Nettoyage et normalisation des Documents — SmartSupport RAG

Rôle :
    Prendre les Documents bruts sortis du loader et les nettoyer
    avant le découpage en chunks.

    Un texte propre → des embeddings plus précis → un RAG plus fiable.

Pipeline de nettoyage (dans l'ordre) :
    1. Suppression des caractères invisibles et de contrôle
    2. Normalisation de l'encodage Unicode
    3. Normalisation des espaces et sauts de ligne
    4. Suppression des répétitions (headers, footers, watermarks)
    5. Suppression du bruit spécifique par type de document
    6. Filtrage des documents vides ou trop courts

Auteur : Oumou Kanfana
"""

import re
import unicodedata
from typing import List

from langchain_core.documents import Document


# =============================================================
# Constantes
# =============================================================

# Longueur minimale d'un document pour être conservé
# Un document de moins de 20 caractères est considéré vide
MIN_CONTENT_LENGTH = 20

# Patterns de bruit fréquents dans les documents d'entreprise
# Ces patterns sont supprimés car ils n'apportent aucune info utile
NOISE_PATTERNS = [
    r'Page\s+\d+\s+(of|sur|/)\s+\d+',      # "Page 1 of 10", "Page 1 sur 10"
    r'©\s*\d{4}.*',                          # "© 2024 TechVision SAS"
    r'Confidentiel\s*[-—].*',               # "Confidentiel — Usage interne"
    r'www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # URLs simples
    r'\[.*?\]\s*\(http.*?\)',               # Liens Markdown [texte](url)
    r'={3,}',                               # Séparateurs === ou ===...
    r'-{3,}',                               # Séparateurs --- ou ---...
    r'_{3,}',                               # Séparateurs ___ ou ___...
]


# =============================================================
# Fonctions de nettoyage
# =============================================================

def remove_invisible_characters(text: str) -> str:
    """
    Supprime les caractères invisibles et de contrôle.

    Ces caractères apparaissent souvent dans les PDF scannés
    ou les fichiers exportés depuis des systèmes legacy.

    Ex: \x00 (null), \x01 (SOH), \x08 (backspace)...

    Args:
        text : texte brut à nettoyer

    Returns:
        Texte sans caractères invisibles
    """
    # Supprime les caractères de contrôle (catégorie Unicode "Cc")
    # sauf \n (newline=0x0A) et \t (tab=0x09) qui sont utiles
    cleaned = ''.join(
        char for char in text
        if unicodedata.category(char) != 'Cc' or char in '\n\t'
    )
    return cleaned


def normalize_unicode(text: str) -> str:
    """
    Normalise l'encodage Unicode en forme NFC.

    Uniformise les caractères accentués qui peuvent avoir
    plusieurs représentations Unicode différentes.

    Ex: é peut être représenté comme 1 ou 2 caractères Unicode.
    NFC garantit une représentation unique.

    Args:
        text : texte à normaliser

    Returns:
        Texte avec encodage Unicode normalisé
    """
    return unicodedata.normalize('NFC', text)


def normalize_whitespace(text: str) -> str:
    """
    Normalise les espaces et les sauts de ligne.

    Transformations :
        - Espaces multiples → un seul espace
        - Tabulations → espace
        - Plus de 2 sauts de ligne consécutifs → 2 sauts de ligne max
        - Espaces en début/fin de ligne supprimés

    Args:
        text : texte à normaliser

    Returns:
        Texte avec espaces normalisés
    """
    # Remplace les tabulations par des espaces
    text = text.replace('\t', ' ')

    # Supprime les espaces en début et fin de chaque ligne
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    # Remplace les espaces multiples par un seul espace
    # (?![\n]) → ne touche pas aux sauts de ligne
    text = re.sub(r' {2,}', ' ', text)

    # Remplace 3+ sauts de ligne consécutifs par 2 maximum
    # Préserve la structure paragraphes sans trop d'espacement
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def remove_noise_patterns(text: str) -> str:
    """
    Supprime les patterns de bruit récurrents dans les documents d'entreprise.

    Ces patterns n'apportent aucune information utile pour le RAG
    et polluent les embeddings.

    Args:
        text : texte à nettoyer

    Returns:
        Texte sans patterns de bruit
    """
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)

    return text


def remove_duplicate_lines(text: str) -> str:
    """
    Supprime les lignes dupliquées consécutives.

    Utile pour les documents avec headers/footers répétés
    sur chaque page (ex: "TechVision SAS — Confidentiel").

    Args:
        text : texte à nettoyer

    Returns:
        Texte sans lignes dupliquées consécutives
    """
    lines = text.split('\n')
    cleaned_lines = []
    previous_line = None

    for line in lines:
        # Ne garde la ligne que si elle est différente de la précédente
        if line.strip() != previous_line:
            cleaned_lines.append(line)
            previous_line = line.strip()

    return '\n'.join(cleaned_lines)


def clean_text(text: str) -> str:
    """
    Pipeline complet de nettoyage d'un texte.

    Applique toutes les transformations dans l'ordre optimal :
    1. Caractères invisibles    → texte lisible
    2. Unicode                  → encodage uniforme
    3. Patterns de bruit        → texte sans parasites
    4. Lignes dupliquées        → texte sans répétitions
    5. Espaces                  → texte bien formaté

    Args:
        text : texte brut à nettoyer

    Returns:
        Texte propre et normalisé
    """
    if not text or not text.strip():
        return ""

    # Étape 1 — Suppression des caractères invisibles
    text = remove_invisible_characters(text)

    # Étape 2 — Normalisation Unicode
    text = normalize_unicode(text)

    # Étape 3 — Suppression des patterns de bruit
    text = remove_noise_patterns(text)

    # Étape 4 — Suppression des lignes dupliquées
    text = remove_duplicate_lines(text)

    # Étape 5 — Normalisation des espaces (en dernier)
    text = normalize_whitespace(text)

    return text


def is_valid_document(document: Document) -> bool:
    """
    Vérifie qu'un Document contient suffisamment de contenu utile.

    Un Document est invalide si :
    - Son contenu est vide ou None
    - Son contenu est trop court (< MIN_CONTENT_LENGTH caractères)
    - Son contenu ne contient que des espaces ou caractères spéciaux

    Args:
        document : Document LangChain à vérifier

    Returns:
        True si le Document est valide, False sinon
    """
    if not document.page_content:
        return False

    content = document.page_content.strip()

    # Trop court pour être utile
    if len(content) < MIN_CONTENT_LENGTH:
        return False

    # Contient au moins quelques lettres ou chiffres
    # (pas seulement des caractères spéciaux)
    if not re.search(r'[a-zA-ZÀ-ÿ0-9]', content):
        return False

    return True


def clean_documents(documents: List[Document]) -> List[Document]:
    """
    Nettoie une liste de Documents LangChain.

    Pour chaque Document :
    1. Nettoie le contenu textuel
    2. Vérifie que le Document est valide après nettoyage
    3. Filtre les Documents vides ou trop courts

    C'est la fonction principale du module — appelée par le pipeline.

    Args:
        documents : liste de Documents bruts sortis du loader

    Returns:
        Liste de Documents propres, prêts pour le chunking

    Exemple:
        >>> from src.ingestion.loader import load_file
        >>> from src.ingestion.cleaner import clean_documents
        >>> docs = load_file("data/pdf_files/FAQ_Support.txt")
        >>> cleaned = clean_documents(docs)
        >>> print(f"{len(cleaned)} documents après nettoyage")
    """
    if not documents:
        return []

    cleaned_docs = []
    n_removed = 0

    for doc in documents:
        # Nettoyage du contenu
        cleaned_content = clean_text(doc.page_content)

        # Mise à jour du contenu dans le Document
        doc.page_content = cleaned_content

        # Ajout du flag de nettoyage dans les métadonnées
        doc.metadata["cleaned"] = True

        # Filtrage des Documents invalides après nettoyage
        if is_valid_document(doc):
            cleaned_docs.append(doc)
        else:
            n_removed += 1

    print(
        f"[Cleaner] ✅ {len(cleaned_docs)} document(s) propres "
        f"({n_removed} supprimé(s) car vides ou trop courts)"
    )

    return cleaned_docs