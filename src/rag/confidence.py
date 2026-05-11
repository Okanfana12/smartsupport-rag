"""
src/rag/confidence.py
═════════════════════
Scoring de confiance — SmartSupport RAG

Rôle :
    Calculer le score de confiance d'une réponse RAG
    et décider si elle est fiable ou si un fallback est nécessaire.

    C'est le mécanisme de sécurité du système :
    → Score élevé  : réponse fiable, affichée normalement
    → Score moyen  : réponse affichée avec avertissement
    → Score faible : fallback déclenché, pas de réponse risquée

Principe fondamental :
    Mieux vaut signaler l'incertitude que risquer une hallucination.
    Dans un contexte support client, une mauvaise réponse
    peut avoir des conséquences réelles pour le client.

Calcul du score :
    Le score final combine plusieurs signaux :
    1. Score de similarité ChromaDB (signal principal)
    2. Nombre de sources concordantes (signal secondaire)
    3. Diversité des sources (bonus)

Auteur : Oumou Kanfana
"""

import os
import yaml
from typing import List, Dict, Any
from enum import Enum


# =============================================================
# Niveaux de confiance
# =============================================================

class ConfidenceLevel(str, Enum):
    """
    Niveaux de confiance d'une réponse RAG.

    HIGH    → réponse fiable, affichée normalement
    MEDIUM  → réponse affichée avec avertissement
    LOW     → fallback déclenché, réponse non affichée
    """
    HIGH   = "high"     # Score ≥ seuil configuré (défaut 75%)
    MEDIUM = "medium"   # Score entre warning et seuil (50-75%)
    LOW    = "low"      # Score < seuil warning (< 50%)


# =============================================================
# Chargement de la configuration
# =============================================================

def load_config() -> dict:
    """
    Charge la configuration depuis config/config.yaml.

    Returns:
        Dictionnaire de configuration
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
# Calcul du score de confiance
# =============================================================

def compute_confidence_score(
    scores: List[float],
    sources: List[str] = None
) -> float:
    """
    Calcule le score de confiance global à partir des scores ChromaDB.

    Stratégie de calcul :
    1. Moyenne pondérée des scores — le meilleur score compte plus
    2. Bonus diversité — plusieurs sources concordantes = plus fiable
    3. Normalisation entre 0.0 et 1.0

    Pourquoi moyenne pondérée et pas simple ?
    → Le chunk le plus pertinent (score 1) doit peser plus
      que les chunks secondaires (scores 2, 3, 4)
    → Évite qu'un chunk peu pertinent tire le score vers le bas

    Args:
        scores  : liste des scores de similarité ChromaDB [0.0 - 1.0]
        sources : liste des noms de fichiers sources (pour diversité)

    Returns:
        Score de confiance entre 0.0 et 1.0

    Exemple:
        >>> score = compute_confidence_score([0.51, 0.50, 0.48, 0.38])
        >>> print(f"Score : {score:.2%}")  # Score : 48.23%
    """
    if not scores:
        return 0.0

    # ── Étape 1 : Moyenne pondérée ────────────────────────────
    # Poids décroissants : chunk 1 compte 4x, chunk 2 compte 3x...
    # Plus le chunk est haut dans le classement → plus il compte
    weights = [1 / (i + 1) for i in range(len(scores))]
    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    weighted_avg = weighted_sum / sum(weights)

    # ── Étape 2 : Bonus diversité des sources ────────────────
    # Si plusieurs fichiers différents sont retrouvés
    # → signal plus fort que si tout vient du même fichier
    diversity_bonus = 0.0
    if sources:
        unique_sources = len(set(sources))
        if unique_sources >= 2:
            # +3% par source supplémentaire, max +9%
            diversity_bonus = min((unique_sources - 1) * 0.03, 0.09)

    # ── Étape 3 : Score final normalisé ──────────────────────
    final_score = min(weighted_avg + diversity_bonus, 1.0)

    return round(final_score, 3)


def get_confidence_level(
    score: float,
    config: dict = None
) -> ConfidenceLevel:
    """
    Détermine le niveau de confiance selon le score et les seuils config.

    Seuils depuis config.yaml :
        confidence_threshold : seuil principal (défaut 0.75)
        confidence_warning   : seuil d'avertissement (défaut 0.50)

    Args:
        score  : score de confiance entre 0.0 et 1.0
        config : dictionnaire de configuration (None = charge depuis fichier)

    Returns:
        ConfidenceLevel.HIGH, MEDIUM ou LOW
    """
    if config is None:
        config = load_config()

    threshold = config["rag"]["confidence_threshold"]  # 0.75
    warning   = config["rag"]["confidence_warning"]    # 0.50

    if score >= threshold:
        return ConfidenceLevel.HIGH
    elif score >= warning:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.LOW


# =============================================================
# Fonction principale
# =============================================================

def assess_confidence(
    scores: List[float],
    sources: List[str] = None,
    config: dict = None
) -> Dict[str, Any]:
    """
    Évalue la confiance complète d'une réponse RAG.

    Combine le calcul du score et la détermination du niveau
    pour retourner un rapport de confiance complet.

    C'est la fonction principale du module — appelée par le pipeline
    après le retrieval pour décider si on génère une réponse ou un fallback.

    Args:
        scores  : scores de similarité ChromaDB
        sources : noms des fichiers sources
        config  : configuration (None = charge depuis fichier)

    Returns:
        Dictionnaire avec :
        - score          : score numérique [0.0 - 1.0]
        - score_pct      : score en pourcentage "51%"
        - level          : ConfidenceLevel (HIGH/MEDIUM/LOW)
        - is_reliable    : True si HIGH ou MEDIUM
        - should_fallback: True si LOW
        - message        : message explicatif pour l'utilisateur

    Exemple:
        >>> result = assess_confidence([0.51, 0.50, 0.48], ["FAQ.pdf", "email"])
        >>> print(result["score_pct"])      # "50%"
        >>> print(result["is_reliable"])    # True
        >>> print(result["should_fallback"])# False
    """
    if config is None:
        config = load_config()

    # Calcul du score
    score = compute_confidence_score(scores, sources)

    # Détermination du niveau
    level = get_confidence_level(score, config)

    # Messages selon le niveau
    messages = {
        ConfidenceLevel.HIGH: "Réponse fiable — sources pertinentes trouvées.",
        ConfidenceLevel.MEDIUM: (
            "⚠️ Confiance modérée — la réponse est basée sur des sources "
            "partiellement pertinentes. Vérification recommandée."
        ),
        ConfidenceLevel.LOW: (
            "❌ Confiance insuffisante — les documents disponibles ne permettent "
            "pas de répondre à cette question avec certitude. "
            "Consultez directement la documentation ou escaladez au niveau 2."
        ),
    }

    return {
        "score":           score,
        "score_pct":       f"{score:.0%}",
        "level":           level,
        "is_reliable":     level in {ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM},
        "should_fallback": level == ConfidenceLevel.LOW,
        "message":         messages[level],
        "scores_detail":   scores,
        "sources_count":   len(set(sources)) if sources else 0,
    }