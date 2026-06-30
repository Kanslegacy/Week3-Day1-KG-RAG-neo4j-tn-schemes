"""
TN Agriculture Scheme Assistant — KG-RAG
Supports two graph storage backends, switchable from the sidebar:
  • Neo4j AuraDB  — cloud graph database, LLM writes Cypher queries
  • NetworkX      — in-memory Python graph, LLM extracts JSON intent
Same question, two different retrieval engines — good for comparing both.
"""
import streamlit as st
from dotenv import load_dotenv
import os
from neo4j import GraphDatabase
from kg_pipeline.text_to_cypher import answer_question as answer_neo4j
from kg_pipeline.nx_answer import answer_question_nx

load_dotenv()

st.set_page_config(
    page_title="TN Agriculture Scheme Assistant (KG-RAG)",
    page_icon="🌾",
    layout="centered"
)


@st.cache_resource
def get_driver():
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    )


driver = get_driver()


@st.cache_data(ttl=600)
def get_graph_stats():
    with driver.session() as session:
        scheme_count = session.run("MATCH (s:Scheme) RETURN count(s) AS c").single()["c"]
        rel_count    = session.run("MATCH ()-->() RETURN count(*) AS c").single()["c"]
    return scheme_count, rel_count


# ---------------------------------------------------------------------------
# CSS + Header banner
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
    .header-banner {
        background: linear-gradient(135deg, #2E7D32 0%, #66BB6A 100%);
        border-radius: 14px; margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(46,125,50,0.25); overflow: hidden;
    }
    .header-banner-content { padding: 26px 24px 10px 24px; }
    .header-banner h1  { color: white; font-size: 26px; margin: 0; }
    .header-banner p   { color: #E8F5E9; margin-top: 6px; font-size: 14px; }
    .gov-badge {
        display: inline-block; background: rgba(255,255,255,0.18);
        color: white; padding: 4px 12px; border-radius: 20px;
        font-size: 12px; font-weight: 600; margin-bottom: 10px;
    }
    .backend-badge {
        display: inline-block; padding: 3px 10px; border-radius: 12px;
        font-size: 12px; font-weight: 600; margin-bottom: 8px;
    }
    .neo4j-badge { background: #E8F5E9; color: #2E7D32; }
    .nx-badge    { background: #E3F2FD; color: #1565C0; }
    [data-testid="stChatMessage"] { border-radius: 14px; padding: 4px 8px; }
    [data-testid="stChatInput"] textarea { border-radius: 10px !important; }
    section[data-testid="stSidebar"] { background-color: #E8F5E9; }
</style>

<div class="header-banner">
  <div class="header-banner-content">
    <span class="gov-badge">🏛️ Government of Tamil Nadu</span>
    <h1>🌾 TN Agriculture Scheme Assistant</h1>
    <p>Knowledge Graph RAG · Agriculture & Farmers Welfare Department</p>
  </div>
  <svg viewBox="0 0 800 70" xmlns="http://www.w3.org/2000/svg"
       style="display:block;width:100%;height:60px;">
    <rect width="800" height="70" fill="#1B5E20" opacity="0.25"/>
    <circle cx="60" cy="15" r="22" fill="#FFF59D" opacity="0.85"/>
    <path d="M0,55 Q40,30 80,55 T160,55 T240,55 T320,55 T400,55
             T480,55 T560,55 T640,55 T720,55 T800,55 V70 H0 Z"
          fill="#388E3C"/>
    <g stroke="#1B5E20" stroke-width="3">
      <line x1="40"  y1="55" x2="40"  y2="40"/>
      <line x1="90"  y1="58" x2="90"  y2="42"/>
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
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏛️ Government of Tamil Nadu")
    st.caption("Department of Agriculture & Farmers Welfare")
    st.divider()

    st.markdown("#### ⚙️ Graph Storage Backend")
    backend = st.radio(
        label="Choose retrieval engine:",
        options=["Neo4j AuraDB", "NetworkX"],
        index=0,
        help=(
            "Neo4j AuraDB — cloud graph DB. LLM writes a Cypher query.\n\n"
            "NetworkX — in-memory Python graph. LLM extracts JSON intent, "
            "Python traverses the graph locally."
        )
    )

    if "active_backend" not in st.session_state:
        st.session_state.active_backend = backend

    if st.session_state.active_backend != backend:
        st.session_state.active_backend = backend
        st.session_state.messages = []
        st.rerun()

    if backend == "Neo4j AuraDB":
        st.markdown('<span class="backend-badge neo4j-badge">🟢 Neo4j AuraDB active</span>',
                    unsafe_allow_html=True)
        st.caption("Retrieval: LLM → Cypher → Neo4j server")
    else:
        st.markdown('<span class="backend-badge nx-badge">🔵 NetworkX active</span>',
                    unsafe_allow_html=True)
        st.caption("Retrieval: LLM → JSON intent → Python traversal")

    st.divider()

    st.markdown("#### 🌿 About this Assistant")
    st.write(
        "Answers questions about TN Agriculture schemes using a "
        "**Knowledge Graph** — understands real connections between "
        "schemes, crops, districts, and farmer groups."
    )
    st.divider()

    st.markdown("#### 📊 Knowledge Graph Stats")
    try:
        scheme_count, rel_count = get_graph_stats()
        col1, col2 = st.columns(2)
        col1.metric("Schemes", scheme_count)
        col2.metric("Relationships", rel_count)
    except Exception:
        st.caption("Stats unavailable.")
    st.divider()

    st.markdown("#### 🔗 Official Source")
    st.markdown(
        "[tn.gov.in – Agriculture Schemes]"
        "(https://www.tn.gov.in/scheme_list.php?dep_id=Mg==)"
    )
    st.caption("Built as a KG-RAG learning project.")


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "cypher" in message:
            with st.expander("🔍 Cypher query (Neo4j)"):
                st.code(message["cypher"], language="cypher")
        if "intent" in message:
            with st.expander("🔍 Extracted intent (NetworkX)"):
                st.json(message["intent"])

user_question = st.chat_input("Ask about a farming scheme...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Querying the knowledge graph..."):

            if backend == "Neo4j AuraDB":
                result = answer_neo4j(driver, user_question)
                st.markdown(result["answer"])
                with st.expander("🔍 Cypher query (Neo4j)"):
                    st.code(result["cypher"], language="cypher")
                st.session_state.messages.append({
                    "role"   : "assistant",
                    "content": result["answer"],
                    "cypher" : result["cypher"],
                })

            else:
                result = answer_question_nx(user_question)
                st.markdown(result["answer"])
                with st.expander("🔍 Extracted intent (NetworkX)"):
                    st.json(result["intent"])
                st.session_state.messages.append({
                    "role"   : "assistant",
                    "content": result["answer"],
                    "intent" : result["intent"],
                })
