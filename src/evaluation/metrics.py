"""
src/evaluation/metrics.py
═════════════════════════
Évaluation et benchmarking — SmartSupport RAG

Rôle :
    Mesurer la qualité des réponses RAG avec RAGAS
    et produire des métriques exploitables pour améliorer le système.

    Différence avec confidence.py :
    confidence.py → score de similarité ChromaDB (avant génération)
    metrics.py    → qualité de la réponse générée (après génération)

Métriques RAGAS :
    Faithfulness      → La réponse est-elle fidèle aux sources ?
                        Détecte les hallucinations
    Answer Relevancy  → La réponse répond-elle à la question ?
                        Détecte les réponses hors sujet
    Context Precision → Les chunks retrouvés sont-ils tous utiles ?
                        Mesure la précision du retrieval
    Context Recall    → Tous les chunks nécessaires ont-ils été retrouvés ?
                        Mesure la couverture du retrieval

Note sur l'installation RAGAS :
    RAGAS nécessite des packages lourds (datasets, transformers)
    On implémente une version simplifiée compatible avec notre env.

Auteur : Oumou Kanfana
"""

import os
import yaml
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


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
# Évaluation simplifiée — sans RAGAS complet
# =============================================================

def evaluate_faithfulness(
    answer: str,
    context: str
) -> float:
    """
    Évalue si la réponse est fidèle au contexte fourni.

    Approche simplifiée :
    On vérifie si les mots clés de la réponse apparaissent
    dans le contexte — une vraie implémentation RAGAS utilise
    un LLM juge pour une évaluation plus fine.

    Args:
        answer  : réponse générée par le LLM
        context : contexte documentaire fourni au LLM

    Returns:
        Score entre 0.0 et 1.0
        1.0 = réponse entièrement basée sur le contexte
        0.0 = réponse sans rapport avec le contexte
    """
    if not answer or not context:
        return 0.0

    # Extraction des mots significatifs de la réponse
    # (mots de plus de 4 caractères pour éviter les mots outils)
    answer_words = set(
        word.lower().strip('.,!?;:')
        for word in answer.split()
        if len(word) > 4
    )

    if not answer_words:
        return 0.0

    # Contexte en minuscules pour comparaison
    context_lower = context.lower()

    # Proportion de mots de la réponse présents dans le contexte
    words_in_context = sum(
        1 for word in answer_words
        if word in context_lower
    )

    score = words_in_context / len(answer_words)

    return round(min(score, 1.0), 3)


def evaluate_answer_relevancy(
    query: str,
    answer: str
) -> float:
    """
    Évalue si la réponse est pertinente par rapport à la question.

    Approche simplifiée :
    On vérifie si les mots clés de la question apparaissent
    dans la réponse.

    Args:
        query  : question posée par l'utilisateur
        answer : réponse générée par le LLM

    Returns:
        Score entre 0.0 et 1.0
    """
    if not query or not answer:
        return 0.0

    # Mots clés de la question
    query_words = set(
        word.lower().strip('.,!?;:')
        for word in query.split()
        if len(word) > 3
    )

    if not query_words:
        return 0.0

    answer_lower = answer.lower()

    # Proportion de mots de la question présents dans la réponse
    words_in_answer = sum(
        1 for word in query_words
        if word in answer_lower
    )

    score = words_in_answer / len(query_words)

    return round(min(score, 1.0), 3)


def evaluate_context_precision(
    query: str,
    chunks: List[Any],
    scores: List[float]
) -> float:
    """
    Évalue si les chunks retrouvés sont pertinents pour la question.

    Approche simplifiée :
    Score moyen des chunks retrouvés pondéré par leur position.
    Un chunk au top du classement compte plus qu'un chunk en bas.

    Args:
        query  : question posée
        chunks : chunks retrouvés par ChromaDB
        scores : scores de similarité correspondants

    Returns:
        Score entre 0.0 et 1.0
    """
    if not scores:
        return 0.0

    # Moyenne pondérée des scores ChromaDB
    # Les chunks en tête de liste comptent plus
    weights = [1 / (i + 1) for i in range(len(scores))]
    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    weighted_avg = weighted_sum / sum(weights)

    return round(weighted_avg, 3)


def evaluate_context_recall(
    query: str,
    chunks: List[Any],
    scores: List[float],
    threshold: float = 0.4
) -> float:
    """
    Évalue si suffisamment de chunks pertinents ont été retrouvés.

    Approche simplifiée :
    Proportion de chunks au-dessus du seuil de pertinence minimum.

    Args:
        query     : question posée
        chunks    : chunks retrouvés
        scores    : scores de similarité
        threshold : seuil minimum de pertinence (défaut 0.4)

    Returns:
        Score entre 0.0 et 1.0
    """
    if not scores:
        return 0.0

    # Proportion de chunks suffisamment pertinents
    relevant_chunks = sum(1 for s in scores if s >= threshold)
    recall = relevant_chunks / len(scores)

    return round(recall, 3)


# =============================================================
# Évaluation complète
# =============================================================

def evaluate_rag_response(
    query: str,
    answer: str,
    context: str,
    chunks: List[Any],
    scores: List[float],
    sources: List[str] = None
) -> Dict[str, Any]:
    """
    Évalue complètement une réponse RAG sur les 4 métriques.

    C'est la fonction principale du module — appelée après
    chaque génération pour mesurer la qualité.

    Args:
        query   : question posée par l'utilisateur
        answer  : réponse générée par Claude
        context : contexte fourni au LLM (chunks concaténés)
        chunks  : liste des Documents retrouvés
        scores  : scores de similarité ChromaDB
        sources : noms des fichiers sources

    Returns:
        Dictionnaire complet avec toutes les métriques et métadonnées

    Exemple:
        >>> metrics = evaluate_rag_response(
        ...     query="Comment réinitialiser un mot de passe ?",
        ...     answer="Pour réinitialiser...",
        ...     context="[Source 1 | FAQ.pdf]...",
        ...     chunks=chunks,
        ...     scores=[0.51, 0.50, 0.48]
        ... )
        >>> print(metrics["faithfulness"])       # 0.823
        >>> print(metrics["answer_relevancy"])   # 0.750
        >>> print(metrics["quality_grade"])      # "B"
    """
    config = load_config()
    thresholds = config["evaluation"]["thresholds"]

    # ── Calcul des 4 métriques ────────────────────────────────
    faithfulness     = evaluate_faithfulness(answer, context)
    answer_relevancy = evaluate_answer_relevancy(query, answer)
    context_precision = evaluate_context_precision(query, chunks, scores)
    context_recall   = evaluate_context_recall(query, chunks, scores)

    # ── Score global ──────────────────────────────────────────
    # Moyenne des 4 métriques avec pondération
    # Faithfulness et relevancy comptent plus (qualité réponse)
    # que precision et recall (qualité retrieval)
    overall_score = (
        faithfulness     * 0.35 +
        answer_relevancy * 0.35 +
        context_precision * 0.15 +
        context_recall   * 0.15
    )

    # ── Note qualité ──────────────────────────────────────────
    if overall_score >= 0.85:
        grade = "A"
    elif overall_score >= 0.75:
        grade = "B"
    elif overall_score >= 0.65:
        grade = "C"
    elif overall_score >= 0.50:
        grade = "D"
    else:
        grade = "F"

    # ── Vérification des seuils ───────────────────────────────
    below_threshold = {
        metric: value
        for metric, value in {
            "faithfulness":     faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall":   context_recall,
        }.items()
        if value < thresholds.get(metric, 0.0)
    }

    return {
        # Métriques principales
        "faithfulness":      faithfulness,
        "answer_relevancy":  answer_relevancy,
        "context_precision": context_precision,
        "context_recall":    context_recall,

        # Score global et note
        "overall_score":     round(overall_score, 3),
        "overall_pct":       f"{overall_score:.0%}",
        "quality_grade":     grade,

        # Alertes qualité
        "below_threshold":   below_threshold,
        "has_quality_issues": len(below_threshold) > 0,

        # Métadonnées
        "query":             query,
        "answer_length":     len(answer),
        "chunks_count":      len(chunks),
        "sources_count":     len(set(sources)) if sources else 0,
        "timestamp":         datetime.utcnow().isoformat(),
    }


def print_metrics_report(metrics: Dict[str, Any]) -> None:
    """
    Affiche un rapport lisible des métriques d'évaluation.

    Args:
        metrics : dictionnaire retourné par evaluate_rag_response()
    """
    grade_colors = {"A": "✅", "B": "✅", "C": "⚠️", "D": "⚠️", "F": "❌"}
    icon = grade_colors.get(metrics["quality_grade"], "❓")

    print(f"\n{'='*50}")
    print(f"RAPPORT QUALITÉ RAG {icon} Note : {metrics['quality_grade']}")
    print(f"{'='*50}")
    print(f"Score global      : {metrics['overall_pct']}")
    print(f"{'─'*50}")
    print(f"Faithfulness      : {metrics['faithfulness']:.0%}  "
          f"(réponse fidèle aux sources ?)")
    print(f"Answer Relevancy  : {metrics['answer_relevancy']:.0%}  "
          f"(réponse à la question ?)")
    print(f"Context Precision : {metrics['context_precision']:.0%}  "
          f"(bons chunks retrouvés ?)")
    print(f"Context Recall    : {metrics['context_recall']:.0%}  "
          f"(couverture suffisante ?)")
    print(f"{'─'*50}")
    print(f"Chunks utilisés   : {metrics['chunks_count']}")
    print(f"Sources uniques   : {metrics['sources_count']}")
    print(f"Longueur réponse  : {metrics['answer_length']} caractères")

    if metrics["has_quality_issues"]:
        print(f"\n⚠️  Métriques sous le seuil :")
        for metric, value in metrics["below_threshold"].items():
            print(f"   {metric}: {value:.0%}")

    print(f"{'='*50}\n")