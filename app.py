"""
Step 5: Streamlit UI for the TN Agriculture Scheme KG-RAG chatbot.

Same chat-interface pattern as the Vector RAG project, but each answer also
shows the Cypher query that was generated and run — transparency into how
the graph was queried, similar to showing "Source: scheme_name" before.
"""
import streamlit as st
from dotenv import load_dotenv
import os
from neo4j import GraphDatabase
from kg_pipeline.text_to_cypher import answer_question

load_dotenv()

st.set_page_config(
    page_title="TN Agriculture Scheme Assistant (KG-RAG)",
    page_icon="🌾",
    layout="centered"
)


@st.cache_resource
def get_driver():
    """Cached so we don't reconnect to Neo4j on every single message."""
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    )


driver = get_driver()


@st.cache_data(ttl=600)
def get_graph_stats():
    """Quick counts shown in the sidebar — refreshes every 10 min, not on
    every keystroke, since the graph data doesn't change during a session."""
    with driver.session() as session:
        scheme_count = session.run("MATCH (s:Scheme) RETURN count(s) AS c").single()["c"]
        rel_count = session.run("MATCH ()-->() RETURN count(*) AS c").single()["c"]
    return scheme_count, rel_count


# ---------------------------------------------------------------------------
# Custom CSS — header banner with an inline SVG farm illustration.
# Drawn ourselves (not a stock photo) so there's no copyright concern and
# nothing that can break as a dead external image link.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
    }

    .header-banner {
        background: linear-gradient(135deg, #2E7D32 0%, #66BB6A 100%);
        border-radius: 14px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.25);
        overflow: hidden;
        position: relative;
    }
    .header-banner-content {
        padding: 26px 24px 10px 24px;
        position: relative;
        z-index: 2;
    }
    .header-banner h1 {
        color: white;
        font-size: 26px;
        margin: 0;
    }
    .header-banner p {
        color: #E8F5E9;
        margin-top: 6px;
        font-size: 14px;
    }
    .gov-badge {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 10px;
        letter-spacing: 0.3px;
    }

    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 4px 8px;
    }
    [data-testid="stChatInput"] textarea {
        border-radius: 10px !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #E8F5E9;
    }
</style>

<div class="header-banner">
    <div class="header-banner-content">
        <span class="gov-badge">🏛️ Government of Tamil Nadu</span>
        <h1>🌾 TN Agriculture Scheme Assistant</h1>
        <p>Knowledge Graph RAG · Ask about Agriculture & Farmers Welfare Department schemes</p>
    </div>
    <svg viewBox="0 0 800 70" xmlns="http://www.w3.org/2000/svg" style="display:block; width:100%; height:60px;">
        <rect width="800" height="70" fill="#1B5E20" opacity="0.25"/>
        <circle cx="60" cy="15" r="22" fill="#FFF59D" opacity="0.85"/>
        <path d="M0,55 Q40,30 80,55 T160,55 T240,55 T320,55 T400,55 T480,55 T560,55 T640,55 T720,55 T800,55 V70 H0 Z" fill="#388E3C"/>
        <g stroke="#1B5E20" stroke-width="3">
            <line x1="40" y1="55" x2="40" y2="40"/>
            <line x1="90" y1="58" x2="90" y2="42"/>
            <line x1="140" y1="55" x2="140" y2="38"/>
            <line x1="190" y1="58" x2="190" y2="42"/>
            <line x1="240" y1="55" x2="240" y2="40"/>
            <line x1="290" y1="58" x2="290" y2="42"/>
            <line x1="340" y1="55" x2="340" y2="38"/>
            <line x1="390" y1="58" x2="390" y2="42"/>
            <line x1="440" y1="55" x2="440" y2="40"/>
            <line x1="490" y1="58" x2="490" y2="42"/>
            <line x1="540" y1="55" x2="540" y2="38"/>
            <line x1="590" y1="58" x2="590" y2="42"/>
            <line x1="640" y1="55" x2="640" y2="40"/>
            <line x1="690" y1="58" x2="690" y2="42"/>
            <line x1="740" y1="55" x2="740" y2="38"/>
            <line x1="780" y1="58" x2="780" y2="42"/>
        </g>
    </svg>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar — Tamil Nadu Government branding + project info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏛️ Government of Tamil Nadu")
    st.caption("Department of Agriculture & Farmers Welfare")
    st.divider()

    st.markdown("#### 🌿 About this Assistant")
    st.write(
        "This chatbot answers questions about Tamil Nadu Agriculture schemes "
        "using a **Knowledge Graph** built with Neo4j — instead of plain text "
        "search, it understands real connections between schemes, crops, "
        "districts, and farmer groups."
    )
    st.divider()

    st.markdown("#### 📊 Knowledge Graph Stats")
    try:
        scheme_count, rel_count = get_graph_stats()
        col1, col2 = st.columns(2)
        col1.metric("Schemes", scheme_count)
        col2.metric("Relationships", rel_count)
    except Exception:
        st.caption("Stats unavailable — check Neo4j connection.")
    st.divider()

    st.markdown("#### 🔗 Official Source")
    st.markdown(
        "[tn.gov.in – Agriculture Schemes](https://www.tn.gov.in/scheme_list.php?dep_id=Mg==)"
    )
    st.caption("Built as a Knowledge Graph RAG learning project.")


# ---------------------------------------------------------------------------
# Chat memory (same pattern as the Vector RAG project)
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "cypher" in message:
            with st.expander("🔍 View generated Cypher query"):
                st.code(message["cypher"], language="cypher")

user_question = st.chat_input("Ask about a farming scheme...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Querying the knowledge graph..."):
            result = answer_question(driver, user_question)
            st.markdown(result["answer"])
            with st.expander("🔍 View generated Cypher query"):
                st.code(result["cypher"], language="cypher")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "cypher": result["cypher"],
    })