import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="NeuroSim v4 | Cinematic 4D Spatial Explorer",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# ADVANCED CINEMATIC DARK SPACE THEME + GLOWING MOON MESH
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Poppins:wght@300;400;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
    color: #f1f0fa !important;
}

h1, h2, h3, .hero-title, .section-header {
    font-family: 'Orbitron', sans-serif;
}

/* Deep Space Background with a massive glowing Vector Moon simulation */
.stApp {
    background: radial-gradient(circle at 50% 120%, #150d2a 0%, #070714 60%, #020205 100%);
    overflow-x: hidden;
}

/* Pseudo-element for the giant cinematic rotating moon / portal effect behind everything */
.stApp::before {
    content: "";
    position: fixed;
    bottom: -300px;
    left: 50%;
    transform: translateX(-50%);
    width: 900px;
    height: 900px;
    background: radial-gradient(circle, rgba(168,85,247,0.12) 0%, rgba(56,189,248,0.05) 50%, transparent 70%);
    border-radius: 50%;
    box-shadow: 0 0 120px rgba(168,85,247,0.15), inset 0 0 80px rgba(56,189,248,0.1);
    z-index: 0;
    pointer-events: none;
    animation: moonPulse 12s ease-in-out infinite alternate;
}

/* Ambient Floating Star Dust */
.stApp::after {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image: 
        radial-gradient(1.5px 1.5px at 15% 20%, rgba(255,255,255,0.6), transparent),
        radial-gradient(2px 2px at 80% 40%, rgba(168,85,247,0.4), transparent),
        radial-gradient(1.5px 1.5px at 40% 70%, rgba(56,189,248,0.5), transparent),
        radial-gradient(2.5px 2.5px at 70% 80%, rgba(236,72,153,0.4), transparent);
    background-size: 150% 150%;
    animation: celestialDrift 40s linear infinite;
    z-index: 0;
    pointer-events: none;
}

@keyframes moonPulse {
    0% { transform: translateX(-50%) scale(1); filter: drop-shadow(0 0 40px rgba(168,85,247,0.1)); }
    100% { transform: translateX(-50%) scale(1.08); filter: drop-shadow(0 0 90px rgba(56,189,248,0.25)); }
}

@keyframes celestialDrift {
    0% { background-position: 0% 0%; }
    100% { background-position: 100% 100%; }
}

/* Sci-fi Interactive Clickable Hologram Boxes (Accordions / Details) */
.holo-box {
    background: rgba(10, 8, 28, 0.65);
    border-radius: 16px;
    border: 1px solid rgba(168, 85, 247, 0.25);
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5), inset 0 0 15px rgba(168, 85, 247, 0.1);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    padding: 10px 18px;
    margin-bottom: 14px;
    transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}

.holo-box:hover {
    border-color: #38bdf8;
    box-shadow: 0 0 25px rgba(56,189,248,0.35);
    transform: translateY(-2px);
}

.hero-title {
    font-weight: 900;
    font-size: 3.5rem;
    background: linear-gradient(135deg, #f3e8ff, #a855f7, #38bdf8);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    text-align: center;
    letter-spacing: 2px;
    filter: drop-shadow(0 0 15px rgba(168,85,247,0.3));
}

.hero-sub {
    text-align: center;
    color: #93c5fd;
    font-size: 1.1rem;
    margin-bottom: 30px;
    text-transform: uppercase;
    letter-spacing: 3px;
}

.section-header {
    font-size: 1.3rem;
    color: #38bdf8;
    text-shadow: 0 0 10px rgba(56,189,248,0.5);
    border-bottom: 1px solid rgba(56,189,248,0.2);
    padding-bottom: 8px;
    margin-top: 25px;
    margin-bottom: 15px;
}

/* Textarea Optimization */
textarea, .stTextArea textarea {
    background: rgba(5, 3, 15, 0.85) !important;
    color: #ffffff !important;
    font-size: 1.1rem !important;
    border-radius: 12px !important;
    border: 1px solid rgba(168, 85, 247, 0.5) !important;
}

/* Action Button Cyber Styling */
div.stButton > button {
    background: linear-gradient(135deg, #a855f7, #ec4899);
    color: white;
    font-family: 'Orbitron', sans-serif;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0 0 15px rgba(168,85,247,0.4);
    letter-spacing: 1px;
}
div.stButton > button:hover {
    box-shadow: 0 0 30px rgba(236,72,153,0.7);
    transform: scale(1.02);
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# HEADER & INTERACTIVE INTERFACE TITLE
# ----------------------------------------------------------------------------
st.markdown('<div class="hero-title">NEUROSIM v4</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">// 4D Spatial Core & Neural Matrix Terminal</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# CORE MODEL CACHE
# ----------------------------------------------------------------------------
@st.cache_resource()
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ----------------------------------------------------------------------------
# TEXT INPUT CONTAINER
# ----------------------------------------------------------------------------
st.markdown('<div class="section-header">📡 [01] INJECT DATA VECTOR</div>', unsafe_allow_html=True)

default_text = """I love natural language processing
Machine learning is fascinating
I enjoy studying artificial intelligence
The weather is nice today
Deep learning models are powerful
I went to the market to buy vegetables
Neural networks can understand language"""

with st.container():
    user_text = st.text_area(
        "Enter query sequence (Line 1) followed by target matrix nodes:",
        value=default_text,
        height=180,
    )
    run = st.button("CORE ANALYZE MATRIX")

# ----------------------------------------------------------------------------
# CALCULATION & GRID GENERATION
# ----------------------------------------------------------------------------
if run and user_text.strip():
    lines = [l.strip() for l in user_text.split("\n") if l.strip()]
    if len(lines) < 2:
        st.warning("Terminal requires at least 1 query node and 1 target mesh.")
        st.stop()

    query = lines[0]
    candidates = lines[1:]
    all_items = [query] + candidates

    # Embeddings Processing
    embeddings = model.encode(all_items)
    query_emb = embeddings[0]
    cand_embs = embeddings[1:]

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sims = np.array([cosine_sim(query_emb, c) for c in cand_embs])
    order = np.argsort(-sims)
    
    # Split interface layout into two high-tech columns
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<div class="section-header">🔮 [02] MULTIDIMENSIONAL ORBIT (4D HYPER-SPIN)</div>', unsafe_allow_html=True)
        
        # 4D PCA Reduction Engine
        n_components = min(3, len(all_items))
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(embeddings)
        if reduced.shape[1] < 3:
            pad = np.zeros((reduced.shape[0], 3 - reduced.shape[1]))
            reduced = np.hstack([reduced, pad])

        short_labels = [t if len(t) <= 22 else t[:20] + "…" for t in all_items]
        colors = ["#ec4899"] + ["#38bdf8"] * len(candidates)
        sizes = [24] + [15] * len(candidates)

        # Generating 4D complex orbital projection path frames
        frames = []
        steps = 90 
        for i in range(steps):
            angle = (i / steps) * 2 * np.pi
            frames.append(go.Frame(
                layout=dict(
                    scene_camera=dict(
                        eye=dict(
                            x=1.9 * np.cos(angle),
                            y=1.9 * np.sin(angle),
                            # 4D wave simulation: z axis tilts smoothly down and up during orbit rotation
                            z=0.8 + 0.4 * np.sin(2 * angle) 
                        )
                    )
                ),
                name=f"f4d_{i}"
            ))

        fig4d = go.Figure(
            data=[go.Scatter3d(
                x=reduced[:, 0], y=reduced[:, 1], z=reduced[:, 2],
                mode="markers+text",
                text=short_labels,
                textposition="top center",
                marker=dict(size=sizes, color=colors, opacity=0.95, line=dict(width=1.5, color="#ffffff")),
            )],
            frames=frames
        )
        
        fig4d.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            scene=dict(
                xaxis=dict(title="DIM_X", showgrid=False, backgroundcolor="rgba(0,0,0,0)"),
                yaxis=dict(title="DIM_Y", showgrid=False, backgroundcolor="rgba(0,0,0,0)"),
                zaxis=dict(title="DIM_Z", showgrid=False, backgroundcolor="rgba(0,0,0,0)"),
            ),
            height=600,
            margin=dict(l=0, r=0, t=0, b=0),
            scene_camera=dict(eye=dict(x=1.9, y=0.0, z=0.8)),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                x=0.02, y=0.02,
                buttons=[
                    dict(
                        label="⚡ INITIATE 4D HYPER-SPIN",
                        method="animate",
                        args=[None, dict(
                            frame=dict(duration=40, redraw=False),
                            fromcurrent=True,
                            transition=dict(duration=0),
                            loop=True
                        )]
                    )
                ]
            )]
        )
        st.plotly_chart(fig4d, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">🎛️ [03] INTERACTIVE MATRIX NODES (CLICK TO OPEN)</div>', unsafe_allow_html=True)
        st.caption("Boxes ke header par click karo details aur neural calculations toggle karne ke liye:")
        
        # Clickable Boxes Engine using HTML Details tags styled with glassmorphism CSS
        for rank, idx in enumerate(order, start=1):
            match_text = candidates[idx]
            score = sims[idx]
            
            # Interactive Cyber Boxes HTML template
            box_html = f"""
            <div class="holo-box">
                <details>
                    <summary style="cursor: pointer; font-weight: 600; color: #f1f0fa; font-family: 'Orbitron', sans-serif;">
                        NODE #{rank}: {match_text[:35]}... — <span style="color: #38bdf8;">SIM: {score:.4f}</span>
                    </summary>
                    <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.1); color: #c4b5fd; font-size: 0.9rem;">
                        <strong>Full String:</strong> "{match_text}"<br>
                        <strong>Vector Cluster Address:</strong> PCA_Coordinate_{[round(x, 2) for x in reduced[idx+1].tolist()]}<br>
                        <strong>Status:</strong> Quantum Computed Semantic Connection Verified.
                    </div>
                </details>
            </div>
            """
            st.markdown(box_html, unsafe_allow_html=True)

    # Full Matrix Heatmap at the bottom over the deep dark moon core
    st.markdown('<div class="section-header">🔥 [04] PAIRWISE QUANTUM MATRIX INTEGRATION</div>', unsafe_allow_html=True)
    sim_matrix = np.zeros((len(all_items), len(all_items)))
    for i in range(len(all_items)):
        for j in range(len(all_items)):
            sim_matrix[i, j] = cosine_sim(embeddings[i], embeddings[j])

    fig_heat = go.Figure(data=go.Heatmap(
        z=sim_matrix, x=short_labels, y=short_labels,
        colorscale=[[0, "#05030f"], [0.5, "#a855f7"], [1, "#38bdf8"]],
        zmin=0, zmax=1,
    ))
    fig_heat.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=450, margin=dict(l=40, r=40, t=10, b=10)
    )
    st.plotly_chart(fig_heat, use_container_width=True)

else:
    st.info("🌌 Terminal Waiting... Click **CORE ANALYZE MATRIX** to synthesize the space environment.")
