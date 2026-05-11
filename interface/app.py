"""
interface/app.py
════════════════
Interface Streamlit — SmartSupport RAG

Assistant : ARIA — Assistant de Recherche Intelligente et d'Analyse
Auto-indexation au démarrage si la base est vide.

Auteur : Oumou Kanfana
"""

import sys
import os
import tempfile
from pathlib import Path

import streamlit as st

# ── Setup du path projet ─────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.ingestion.loader import load_file, load_directory
from src.ingestion.cleaner import clean_documents
from src.chunking.chunker import chunk_documents
from src.vectorstore.store import add_documents, get_collection_stats, reset_collection
from src.rag.generator import generate
from src.rag.confidence import assess_confidence, ConfidenceLevel

# =============================================================
# Configuration de la page
# =============================================================

st.set_page_config(
    page_title="ARIA — Assistant Support",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================
# Auto-indexation au démarrage
# =============================================================

@st.cache_resource(show_spinner=False)
def initialize_database():
    """
    Initialise la base documentaire au démarrage.

    Vérifie si la base ChromaDB est vide.
    Si oui → indexe automatiquement tous les documents du dossier data/.
    Si non → utilise la base existante.

    @st.cache_resource garantit que cette fonction
    ne s'exécute qu'une seule fois par session Streamlit.
    """
    try:
        stats = get_collection_stats()
        if stats["chunks_indexed"] > 0:
            return stats["chunks_indexed"]

        # Base vide → indexation automatique
        folders = [
            'data/pdf_files/',
            'data/word_files/',
            'data/csv_files/',
            'data/json_files/',
            'data/html_files/',
            'data/email_files/',
        ]

        all_chunks = []
        for folder in folders:
            if os.path.exists(folder):
                docs    = load_directory(folder)
                cleaned = clean_documents(docs)
                chunks  = chunk_documents(cleaned)
                all_chunks.extend(chunks)

        if all_chunks:
            n = add_documents(all_chunks)
            return n

        return 0

    except Exception:
        return 0


# Lancement de l'initialisation
n_chunks_indexed = initialize_database()

# =============================================================
# CSS — Design minimaliste et professionnel
# =============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    :root {
        --blue: #2563EB;
        --blue-light: #EFF6FF;
        --blue-border: #DBEAFE;
        --green: #059669;
        --green-light: #ECFDF5;
        --orange: #D97706;
        --orange-light: #FFFBEB;
        --red: #DC2626;
        --red-light: #FEF2F2;
        --gray-50: #F9FAFB;
        --gray-100: #F3F4F6;
        --gray-200: #E5E7EB;
        --gray-500: #6B7280;
        --gray-900: #111827;
    }

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .aria-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1.5rem 0;
        border-bottom: 1px solid var(--gray-200);
        margin-bottom: 2rem;
    }

    .aria-avatar {
        width: 48px; height: 48px;
        background: var(--blue);
        border-radius: 14px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.4rem; flex-shrink: 0;
    }

    .aria-name {
        font-size: 1.4rem; font-weight: 600;
        color: var(--gray-900); letter-spacing: -0.3px; margin: 0;
    }

    .aria-role {
        font-size: 0.82rem; color: var(--gray-500);
        margin: 2px 0 0 0;
    }

    .status-badge {
        display: inline-flex; align-items: center; gap: 0.4rem;
        background: var(--green-light); color: var(--green);
        border: 1px solid #A7F3D0; border-radius: 20px;
        padding: 0.25rem 0.75rem; font-size: 0.75rem; font-weight: 500;
    }

    .status-dot {
        width: 6px; height: 6px;
        background: var(--green); border-radius: 50%;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.4; }
    }

    .confidence-card {
        border-radius: 14px; padding: 1.25rem 1.5rem;
        margin: 1rem 0; display: flex; align-items: center; gap: 1rem;
    }

    .conf-high   { background: var(--green-light);  border: 1px solid #A7F3D0; }
    .conf-medium { background: var(--orange-light); border: 1px solid #FDE68A; }
    .conf-low    { background: var(--red-light);    border: 1px solid #FECACA; }

    .conf-icon { font-size: 1.5rem; }

    .conf-score {
        font-size: 1.8rem; font-weight: 700;
        font-family: 'DM Mono', monospace; line-height: 1;
    }

    .conf-high .conf-score   { color: var(--green); }
    .conf-medium .conf-score { color: var(--orange); }
    .conf-low .conf-score    { color: var(--red); }

    .conf-label { font-size: 0.78rem; font-weight: 500; margin-top: 2px; }
    .conf-high .conf-label   { color: var(--green); }
    .conf-medium .conf-label { color: var(--orange); }
    .conf-low .conf-label    { color: var(--red); }

    .conf-message { font-size: 0.85rem; color: var(--gray-500); flex: 1; }

    .answer-box {
        background: white; border: 1px solid var(--gray-200);
        border-radius: 14px; padding: 1.5rem 1.75rem;
        line-height: 1.75; color: var(--gray-900);
        font-size: 0.95rem; margin: 1rem 0;
    }

    .source-tag {
        display: inline-block;
        background: var(--blue-light); color: var(--blue);
        border: 1px solid var(--blue-border); border-radius: 6px;
        padding: 0.2rem 0.65rem; font-size: 0.78rem; font-weight: 500;
        margin: 0.2rem; font-family: 'DM Mono', monospace;
    }

    .section-title {
        font-size: 0.7rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.08em;
        color: var(--gray-500); margin-bottom: 0.75rem;
    }

    .stTextArea textarea {
        border-radius: 12px !important;
        border: 1.5px solid var(--gray-200) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.95rem !important; padding: 1rem !important;
    }

    .stTextArea textarea:focus {
        border-color: var(--blue) !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
    }

    .stButton > button[kind="primary"] {
        background: var(--blue) !important;
        border: none !important; border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important; padding: 0.6rem 1.5rem !important;
        transition: all 0.2s !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: #1D4ED8 !important;
        box-shadow: 0 4px 12px rgba(37,99,235,0.3) !important;
        transform: translateY(-1px) !important;
    }

    .sidebar-label {
        font-size: 0.7rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.08em;
        color: var(--gray-500); margin-bottom: 0.5rem;
    }

    .welcome {
        text-align: center; padding: 4rem 2rem; color: var(--gray-500);
    }

    .welcome-icon  { font-size: 3rem; margin-bottom: 1rem; }
    .welcome-title { font-size: 1.1rem; font-weight: 600; color: #374151; margin-bottom: 0.5rem; }
    .welcome-sub   { font-size: 0.85rem; line-height: 1.6; max-width: 420px; margin: 0 auto; }

    .suggestion {
        background: var(--gray-50); border: 1px solid var(--gray-200);
        border-radius: 10px; padding: 0.75rem 1rem;
        font-size: 0.85rem; color: #374151; margin: 0.3rem 0;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# =============================================================
# Fonctions utilitaires
# =============================================================

def index_document(uploaded_file) -> bool:
    """Indexe un document uploadé dans la base documentaire."""
    try:
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        with st.spinner(f"Lecture de {uploaded_file.name}..."):
            docs    = load_file(tmp_path)
            cleaned = clean_documents(docs)
            chunks  = chunk_documents(cleaned)
            add_documents(chunks)
        os.unlink(tmp_path)
        return True
    except Exception as e:
        st.error(f"Impossible de traiter ce document : {e}")
        return False


def get_conf_class(level) -> str:
    val = level.value if hasattr(level, 'value') else str(level)
    return {"high": "conf-high", "medium": "conf-medium", "low": "conf-low"}.get(val, "conf-low")


def get_conf_icon(level) -> str:
    val = level.value if hasattr(level, 'value') else str(level)
    return {"high": "✅", "medium": "⚠️", "low": "❌"}.get(val, "❌")


def get_conf_label(level) -> str:
    val = level.value if hasattr(level, 'value') else str(level)
    return {
        "high":   "Réponse fiable",
        "medium": "Fiabilité modérée — vérification recommandée",
        "low":    "Information insuffisante dans les documents"
    }.get(val, "Non défini")


# =============================================================
# Sidebar
# =============================================================

with st.sidebar:

    st.markdown("""
    <div style="padding:1rem 0 1.5rem; border-bottom:1px solid #E5E7EB; margin-bottom:1.5rem;">
        <div style="font-size:1rem; font-weight:600; color:#111827;">🔵 ARIA</div>
        <div style="font-size:0.75rem; color:#6B7280; margin-top:2px;">Assistant Support TechVision</div>
    </div>
    """, unsafe_allow_html=True)

    # Statut base
    st.markdown('<div class="sidebar-label">📚 Base documentaire</div>', unsafe_allow_html=True)
    try:
        stats = get_collection_stats()
        n = stats["chunks_indexed"]
        if n > 0:
            st.markdown(f"""
            <div style="background:#ECFDF5; border:1px solid #A7F3D0; border-radius:8px;
                        padding:0.75rem; font-size:0.85rem; color:#065F46; margin-bottom:1rem;">
                ✅ {n} passages disponibles
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#FEF2F2; border:1px solid #FECACA; border-radius:8px;
                        padding:0.75rem; font-size:0.85rem; color:#991B1B; margin-bottom:1rem;">
                ⚠️ Aucun document indexé
            </div>
            """, unsafe_allow_html=True)
    except Exception:
        st.info("Base non initialisée")

    st.markdown("---")

    # Upload
    st.markdown('<div class="sidebar-label">➕ Ajouter un document</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "PDF, Word, CSV, JSON, HTML, Email",
        type=["pdf", "docx", "csv", "json", "html", "eml", "txt"],
        label_visibility="collapsed"
    )

    if uploaded:
        st.markdown(f"""
        <div style="background:#F9FAFB; border:1px solid #E5E7EB; border-radius:8px;
                    padding:0.6rem 0.8rem; font-size:0.82rem; color:#374151; margin-bottom:0.5rem;">
            📄 {uploaded.name}
        </div>
        """, unsafe_allow_html=True)

        if st.button("Ajouter à la base", type="primary", use_container_width=True):
            if index_document(uploaded):
                st.success(f"✅ **{uploaded.name}** ajouté !")
                st.rerun()

    st.markdown("---")

    # Gestion
    st.markdown('<div class="sidebar-label">🗑️ Gestion</div>', unsafe_allow_html=True)
    if st.button("Vider la base documentaire", use_container_width=True):
        reset_collection()
        # Reset du cache pour forcer la réindexation au prochain chargement
        st.cache_resource.clear()
        st.success("Base vidée")
        st.rerun()


# =============================================================
# Zone principale
# =============================================================

# Header ARIA
st.markdown("""
<div class="aria-header">
    <div class="aria-avatar">🔵</div>
    <div style="flex:1;">
        <p class="aria-name">ARIA</p>
        <p class="aria-role">
            Assistant de Recherche Intelligente et d'Analyse —
            Je réponds à vos questions à partir de vos documents internes.
            Chaque réponse est sourcée et accompagnée d'un niveau de fiabilité.
        </p>
    </div>
    <div>
        <div class="status-badge">
            <div class="status-dot"></div>
            En ligne
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Zone de question
query = st.text_area(
    "question",
    placeholder="Posez votre question… Ex : Comment réinitialiser le mot de passe d'un client ?",
    height=110,
    label_visibility="collapsed"
)

col1, col2, _ = st.columns([1.5, 1, 5])
with col1:
    run = st.button("🔍 Rechercher", type="primary", use_container_width=True)
with col2:
    if st.button("Effacer", use_container_width=True):
        st.rerun()

# =============================================================
# Pipeline RAG
# =============================================================

if run and query.strip():

    with st.spinner("ARIA analyse votre question..."):
        result = generate(query)
        conf   = assess_confidence(result["scores"], result["sources"])

    st.markdown("---")

    # Carte de confiance
    conf_class = get_conf_class(conf["level"])
    conf_icon  = get_conf_icon(conf["level"])
    conf_label = get_conf_label(conf["level"])

    st.markdown(f"""
    <div class="confidence-card {conf_class}">
        <div class="conf-icon">{conf_icon}</div>
        <div>
            <div class="conf-score">{conf['score_pct']}</div>
            <div class="conf-label">{conf_label}</div>
        </div>
        <div class="conf-message">{conf['message']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Réponse ou fallback
    if not result["is_fallback"]:

        st.markdown('<div class="section-title">Réponse</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="answer-box">
            {result['answer'].replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)

        if result["sources"]:
            st.markdown('<div class="section-title" style="margin-top:1rem;">Documents consultés</div>', unsafe_allow_html=True)
            sources_html = "".join(
                f'<span class="source-tag">📄 {s}</span>'
                for s in result["sources"]
            )
            st.markdown(sources_html, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="answer-box" style="border-color:#FECACA; background:#FEF2F2; color:#991B1B;">
            {conf['message']}
        </div>
        """, unsafe_allow_html=True)

elif run and not query.strip():
    st.warning("Veuillez saisir une question.")

else:
    # Écran d'accueil
    st.markdown("""
    <div class="welcome">
        <div class="welcome-icon">🔵</div>
        <div class="welcome-title">Bonjour, je suis ARIA</div>
        <div class="welcome-sub">
            Votre assistant de support intelligent.
            Posez-moi une question sur vos procédures, contrats,
            tickets ou politiques internes — je retrouve la réponse
            dans vos documents et vous indique ma fiabilité.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title" style="text-align:center;">Exemples de questions</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    suggestions = [
        "Comment réinitialiser le mot de passe d'un client ?",
        "Que faire si un client conteste une facture ?",
        "Procédure en cas d'accès non autorisé ?",
        "Comment exporter les données d'un compte ?",
    ]
    with col1:
        for s in suggestions[:2]:
            st.markdown(f'<div class="suggestion">💬 {s}</div>', unsafe_allow_html=True)
    with col2:
        for s in suggestions[2:]:
            st.markdown(f'<div class="suggestion">💬 {s}</div>', unsafe_allow_html=True)