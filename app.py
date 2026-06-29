import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WordWeave · NLP Similarity Explorer",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #0D0D1A;
    --surface: #13132A;
    --surface2: #1C1C3A;
    --border: #2A2A5A;
    --accent: #7C6FFF;
    --accent2: #FF6FD8;
    --accent3: #6FFFE9;
    --accent4: #FFD06F;
    --text: #E8E8FF;
    --muted: #888AAA;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'Space Grotesk', sans-serif;
    color: var(--text);
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

[data-testid="stHeader"] { background: transparent !important; }

.stTextArea textarea, .stTextInput input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    padding: 12px !important;
}

.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(124,111,255,0.25) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 12px 32px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 24px rgba(124,111,255,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(124,111,255,0.55) !important;
}

.hero-title {
    font-size: 3.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #7C6FFF 0%, #FF6FD8 50%, #6FFFE9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
}

.hero-sub {
    font-size: 1.05rem;
    color: var(--muted);
    font-weight: 300;
    letter-spacing: 0.04em;
}

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 24px 28px;
    margin-bottom: 20px;
}

.card-accent {
    border-left: 4px solid var(--accent);
}

.metric-pill {
    display: inline-block;
    background: linear-gradient(135deg, rgba(124,111,255,0.15), rgba(255,111,216,0.15));
    border: 1px solid rgba(124,111,255,0.4);
    border-radius: 999px;
    padding: 4px 16px;
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--accent);
    font-family: 'JetBrains Mono', monospace;
}

.score-high { color: #6FFFE9; font-weight: 700; }
.score-mid  { color: #FFD06F; font-weight: 700; }
.score-low  { color: #FF6F6F; font-weight: 700; }

.section-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.badge {
    background: rgba(124,111,255,0.2);
    border: 1px solid rgba(124,111,255,0.5);
    color: #A89FFF;
    border-radius: 8px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
}

.ct-box {
    background: linear-gradient(135deg, rgba(124,111,255,0.08), rgba(111,255,233,0.05));
    border: 1px solid rgba(124,111,255,0.3);
    border-radius: 16px;
    padding: 20px 24px;
    margin-top: 8px;
}

.ct-standard {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    margin-bottom: 12px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}

.ct-icon {
    font-size: 1.2rem;
    flex-shrink: 0;
    margin-top: 2px;
}

.ct-content strong {
    display: block;
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--accent3);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 2px;
}

.ct-content p {
    font-size: 0.9rem;
    color: var(--muted);
    margin: 0;
    line-height: 1.5;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 28px 0;
}

.model-tag {
    background: rgba(111,255,233,0.1);
    border: 1px solid rgba(111,255,233,0.3);
    color: var(--accent3);
    border-radius: 8px;
    padding: 3px 12px;
    font-size: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    display: inline-block;
}

.warning-box {
    background: rgba(255,208,111,0.08);
    border: 1px solid rgba(255,208,111,0.3);
    border-radius: 12px;
    padding: 12px 18px;
    color: var(--accent4);
    font-size: 0.88rem;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 20px'>
      <div style='font-size:1.8rem; font-weight:700; background:linear-gradient(135deg,#7C6FFF,#FF6FD8);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>🔮 WordWeave</div>
      <div style='color:#888AAA; font-size:0.82rem; margin-top:4px;'>NLP Similarity Explorer</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='model-tag'>all-MiniLM-L6-v2</div>", unsafe_allow_html=True)
    st.caption("Sentence Transformers · Free · No API Key · 384-dim embeddings")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.markdown("**⚙️ Settings**")
    top_n = st.slider("Top N similar pairs to show", 3, 10, 5)
    chart_theme = st.selectbox("Color Palette", ["Cosmic", "Aurora", "Neon", "Sunset"])
    show_scores = st.toggle("Show raw score values", value=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color:#888AAA; font-size:0.8rem; line-height:1.6;'>
    <strong style='color:#E8E8FF;'>How it works</strong><br>
    Enter words or sentences separated by newlines. The model converts each into a
    384-dimensional vector and computes cosine similarity between all pairs.
    <br><br>
    <strong style='color:#E8E8FF;'>No preprocessing.</strong> Text is passed directly
    to the model — no tokenization, no stopword removal, no stemming.
    </div>
    """, unsafe_allow_html=True)

# ─── Color palettes ───────────────────────────────────────────────────────────────
PALETTES = {
    "Cosmic":  ["#7C6FFF","#FF6FD8","#6FFFE9","#FFD06F","#FF6F6F","#6FC8FF","#AEFF6F","#FF9B6F"],
    "Aurora":  ["#00F5A0","#00D9F5","#7B61FF","#F56AE4","#F5A623","#6AFFA8","#F56161","#61D4F5"],
    "Neon":    ["#00FFAA","#AA00FF","#FF00AA","#FFAA00","#00AAFF","#FF4400","#44FF00","#FF0044"],
    "Sunset":  ["#FF6B6B","#FFA751","#FFE259","#4ECDC4","#A78BFA","#F59E0B","#EC4899","#10B981"],
}

# ─── Model loader ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ─── Main content ────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom: 2rem;'>
  <div class='hero-title'>WordWeave</div>
  <div class='hero-sub'>Explore semantic similarity using free pretrained NLP embeddings</div>
</div>
""", unsafe_allow_html=True)

# ─── Input section ───────────────────────────────────────────────────────────────
col_in, col_info = st.columns([2, 1])

with col_in:
    st.markdown("<div class='section-title'>📝 Your Text <span class='badge'>one per line</span></div>", unsafe_allow_html=True)
    user_input = st.text_area(
        label="",
        value="artificial intelligence\nmachine learning\ndeep learning\nnatural language processing\ncomputer vision\ndata science\ncooking recipes\nfootball match\nocean waves",
        height=220,
        placeholder="Enter words, phrases, or sentences — one per line...",
        label_visibility="collapsed"
    )

with col_info:
    st.markdown("""
    <div class='card' style='height: 100%; margin-top: 0;'>
      <div style='font-size:0.85rem; color:#888AAA; line-height:1.7;'>
        <div style='color:#E8E8FF; font-weight:600; margin-bottom:8px;'>💡 Try examples</div>
        <div style='margin-bottom:6px;'>🔬 <em>Science terms</em> — watch related concepts cluster together</div>
        <div style='margin-bottom:6px;'>🌍 <em>Country names</em> — geography shapes similarity</div>
        <div style='margin-bottom:6px;'>📖 <em>Full sentences</em> — model handles context too</div>
        <div style='margin-top:14px; padding-top:14px; border-top:1px solid rgba(255,255,255,0.08);'>
          <strong style='color:#A89FFF;'>Min: 3 items</strong><br>
          <span style='font-size:0.8rem;'>For best heatmap & PCA</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

run_btn = st.button("✨  Analyse Similarity", use_container_width=False)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ─── Analysis ────────────────────────────────────────────────────────────────────
if run_btn:
    texts = [t.strip() for t in user_input.strip().split("\n") if t.strip()]

    if len(texts) < 2:
        st.markdown("<div class='warning-box'>⚠️ Please enter at least 2 lines of text to compare.</div>", unsafe_allow_html=True)
        st.stop()

    palette = PALETTES[chart_theme]

    with st.spinner("🔮 Encoding with all-MiniLM-L6-v2..."):
        model = load_model()
        embeddings = model.encode(texts)
        sim_matrix = cosine_similarity(embeddings)

    # ── Score pairs ─────────────────────────────────────────────────────────────
    pairs = []
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            pairs.append({
                "A": texts[i], "B": texts[j],
                "score": float(sim_matrix[i][j]),
                "label": f"{texts[i][:20]}  ↔  {texts[j][:20]}"
            })
    pairs.sort(key=lambda x: x["score"], reverse=True)

    top_pairs = pairs[:top_n]
    top_scores = [p["score"] for p in top_pairs]
    top_labels = [p["label"] for p in top_pairs]

    # ── KPIs ────────────────────────────────────────────────────────────────────
    avg_sim = np.mean([p["score"] for p in pairs])
    max_pair = pairs[0]
    min_pair = pairs[-1]

    k1, k2, k3, k4 = st.columns(4)
    for col, icon, label, val, color in [
        (k1, "🔗", "Most Similar", f"{max_pair['score']:.4f}", "#6FFFE9"),
        (k2, "📉", "Least Similar", f"{min_pair['score']:.4f}", "#FF6F6F"),
        (k3, "📊", "Average Score", f"{avg_sim:.4f}", "#FFD06F"),
        (k4, "🔢", "Pairs Compared", str(len(pairs)), "#A89FFF"),
    ]:
        col.markdown(f"""
        <div class='card' style='text-align:center; padding:20px;'>
          <div style='font-size:1.6rem;'>{icon}</div>
          <div style='font-size:0.78rem; color:#888AAA; margin:4px 0;'>{label}</div>
          <div style='font-size:1.5rem; font-weight:700; color:{color};'>{val}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # GRAPH 1 — Bar Chart
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-title'>📊 Top Similar Pairs <span class='badge'>Bar Chart</span></div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#888AAA; font-size:0.88rem; margin-bottom:14px;'>Cosine similarity scores for the most semantically related pairs</div>", unsafe_allow_html=True)

    bar_colors = [palette[i % len(palette)] for i in range(len(top_pairs))]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=top_scores,
        y=top_labels,
        orientation="h",
        marker=dict(
            color=bar_colors,
            opacity=0.9,
            line=dict(width=0),
        ),
        text=[f"{s:.4f}" for s in top_scores] if show_scores else None,
        textposition="outside",
        textfont=dict(color="#E8E8FF", size=12, family="JetBrains Mono"),
    ))
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[0, 1.05],
            gridcolor="rgba(255,255,255,0.06)",
            color="#888AAA",
            tickfont=dict(family="JetBrains Mono", size=11),
            title=dict(text="Cosine Similarity Score", font=dict(color="#888AAA", size=12))
        ),
        yaxis=dict(
            color="#E8E8FF",
            tickfont=dict(family="Space Grotesk", size=12),
            automargin=True,
        ),
        margin=dict(l=10, r=80, t=20, b=40),
        height=max(300, top_n * 55),
        bargap=0.35,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # GRAPH 2 — Heatmap
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-title'>🌡️ Pairwise Similarity Heatmap <span class='badge'>Heatmap</span></div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#888AAA; font-size:0.88rem; margin-bottom:14px;'>Every cell shows the cosine similarity between two inputs — brighter = more similar</div>", unsafe_allow_html=True)

    short_labels = [t[:22] + ("…" if len(t) > 22 else "") for t in texts]

    colorscale_map = {
        "Cosmic":  [[0,"#13132A"],[0.5,"#7C6FFF"],[1,"#6FFFE9"]],
        "Aurora":  [[0,"#0D1A2A"],[0.5,"#00D9F5"],[1,"#00F5A0"]],
        "Neon":    [[0,"#0D0D1A"],[0.5,"#AA00FF"],[1,"#00FFAA"]],
        "Sunset":  [[0,"#1A0D0D"],[0.5,"#FF6B6B"],[1,"#FFE259"]],
    }

    fig_heat = go.Figure(go.Heatmap(
        z=sim_matrix,
        x=short_labels,
        y=short_labels,
        colorscale=colorscale_map[chart_theme],
        text=np.round(sim_matrix, 3) if show_scores else None,
        texttemplate="%{text}" if show_scores else None,
        textfont=dict(size=10, family="JetBrains Mono", color="#E8E8FF"),
        hovertemplate="<b>%{y}</b><br>vs <b>%{x}</b><br>Score: %{z:.4f}<extra></extra>",
        zmin=0, zmax=1,
        colorbar=dict(
            tickfont=dict(color="#888AAA", family="JetBrains Mono"),
            title=dict(text="Score", font=dict(color="#888AAA")),
            bgcolor="rgba(0,0,0,0)",
        )
    ))
    fig_heat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(color="#888AAA", tickfont=dict(size=11, family="Space Grotesk"), tickangle=-35),
        yaxis=dict(color="#888AAA", tickfont=dict(size=11, family="Space Grotesk")),
        margin=dict(l=10, r=10, t=20, b=10),
        height=max(360, len(texts) * 48),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # GRAPH 3 — PCA 2D Embedding Plot
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-title'>🌌 2D Embedding Space <span class='badge'>PCA</span></div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#888AAA; font-size:0.88rem; margin-bottom:14px;'>384 dimensions collapsed to 2D via PCA — nearby points are semantically related</div>", unsafe_allow_html=True)

    if len(texts) >= 2:
        n_components = min(2, len(texts))
        pca = PCA(n_components=n_components)
        coords = pca.fit_transform(embeddings)

        if n_components == 1:
            coords = np.column_stack([coords, np.zeros(len(coords))])

        variance = pca.explained_variance_ratio_

        # Compute avg similarity per point for coloring
        avg_sims = [np.mean([sim_matrix[i][j] for j in range(len(texts)) if j != i]) for i in range(len(texts))]

        point_colors = [palette[i % len(palette)] for i in range(len(texts))]

        fig_pca = go.Figure()

        # Draw connecting lines for highly similar pairs
        for p in pairs:
            if p["score"] > 0.5:
                i = texts.index(p["A"])
                j = texts.index(p["B"])
                opacity = (p["score"] - 0.5) * 1.2
                fig_pca.add_trace(go.Scatter(
                    x=[coords[i,0], coords[j,0]],
                    y=[coords[i,1], coords[j,1]],
                    mode="lines",
                    line=dict(color=f"rgba(124,111,255,{opacity:.2f})", width=1.5),
                    showlegend=False,
                    hoverinfo="skip",
                ))

        # Points
        fig_pca.add_trace(go.Scatter(
            x=coords[:,0],
            y=coords[:,1],
            mode="markers+text",
            marker=dict(
                size=18,
                color=point_colors,
                line=dict(width=2, color="rgba(255,255,255,0.3)"),
                opacity=0.9,
            ),
            text=short_labels,
            textposition="top center",
            textfont=dict(size=11, family="Space Grotesk", color="#E8E8FF"),
            customdata=[[f"{avg_sims[i]:.4f}"] for i in range(len(texts))],
            hovertemplate="<b>%{text}</b><br>Avg similarity: %{customdata[0]}<extra></extra>",
            showlegend=False,
        ))

        fig_pca.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,13,26,0.6)",
            xaxis=dict(
                title=dict(text=f"PC 1 ({variance[0]*100:.1f}% variance)", font=dict(color="#888AAA", size=11)),
                gridcolor="rgba(255,255,255,0.05)",
                color="#888AAA",
                tickfont=dict(family="JetBrains Mono", size=10),
                zeroline=False,
            ),
            yaxis=dict(
                title=dict(text=f"PC 2 ({variance[1]*100:.1f}% variance)" if len(variance) > 1 else "PC 2", font=dict(color="#888AAA", size=11)),
                gridcolor="rgba(255,255,255,0.05)",
                color="#888AAA",
                tickfont=dict(family="JetBrains Mono", size=10),
                zeroline=False,
            ),
            margin=dict(l=10, r=10, t=30, b=10),
            height=480,
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # GRAPH 4 — Radar / Distribution (Bonus)
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-title'>📡 Similarity Distribution <span class='badge'>Violin + Box</span></div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#888AAA; font-size:0.88rem; margin-bottom:14px;'>Distribution of all pairwise similarity scores — shows how tightly your inputs cluster</div>", unsafe_allow_html=True)

    all_scores = [p["score"] for p in pairs]

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Violin(
        y=all_scores,
        box_visible=True,
        meanline_visible=True,
        fillcolor="rgba(124,111,255,0.25)",
        line_color="#7C6FFF",
        opacity=0.8,
        name="All Pairs",
        points="all",
        pointpos=0,
        marker=dict(color=palette[:len(all_scores)], size=8, opacity=0.7),
        hovertemplate="Score: %{y:.4f}<extra></extra>",
    ))
    fig_dist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(
            title=dict(text="Cosine Similarity", font=dict(color="#888AAA")),
            gridcolor="rgba(255,255,255,0.06)",
            color="#888AAA",
            tickfont=dict(family="JetBrains Mono", size=11),
            range=[0, 1.05],
        ),
        xaxis=dict(color="#888AAA"),
        margin=dict(l=10, r=10, t=20, b=10),
        height=340,
        showlegend=False,
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # RESULTS TABLE
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-title'>📋 Full Similarity Table <span class='badge'>All Pairs</span></div>", unsafe_allow_html=True)

    df = pd.DataFrame([{
        "Rank": i+1,
        "Text A": p["A"],
        "Text B": p["B"],
        "Similarity": round(p["score"], 6),
        "Strength": "🟢 High" if p["score"] >= 0.7 else ("🟡 Medium" if p["score"] >= 0.4 else "🔴 Low")
    } for i, p in enumerate(pairs)])

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Similarity": st.column_config.ProgressColumn(
                "Similarity",
                format="%.4f",
                min_value=0,
                max_value=1,
            )
        }
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # PAUL'S CRITICAL THINKING STANDARDS
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-title'>🧠 Critical Thinking Analysis <span class='badge'>Paul's Standards</span></div>", unsafe_allow_html=True)

    top_p = pairs[0]
    bot_p = pairs[-1]
    score_label = lambda s: "High" if s >= 0.7 else ("Medium" if s >= 0.4 else "Low")

    ct_data = [
        ("🔍", "Clarity", f"The input consisted of {len(texts)} text items. The model encoded each into a 384-dimensional vector. "
                          f"Cosine similarity was measured between all {len(pairs)} unique pairs. Scores range from 0 (unrelated) to 1 (identical)."),
        ("✅", "Accuracy", f"Model used: <strong>all-MiniLM-L6-v2</strong> (Sentence Transformers, free, open-source). "
                          f"No claims are made beyond what cosine similarity in embedding space can support."),
        ("📏", "Precision", f"Top pair: <em>{top_p['A']}</em> ↔ <em>{top_p['B']}</em> scored exactly <strong>{top_p['score']:.6f}</strong>. "
                           f"Lowest pair scored <strong>{bot_p['score']:.6f}</strong>. All scores are reported to 6 decimal places."),
        ("🎯", "Relevance", f"All four graphs directly reflect the computed similarity values — the bar chart ranks pairs, "
                           f"the heatmap shows all pairwise scores, PCA visualises clustering in embedding space, and the violin plot shows score distribution."),
        ("💡", "Logic", f"The top result (<em>{top_p['A']}</em> ↔ <em>{top_p['B']}</em>, score {top_p['score']:.4f}) is rated '{score_label(top_p['score'])}' "
                        f"because the model's training data contains many co-occurrences of these concepts, placing their vectors close together in high-dimensional space."),
        ("⭐", "Significance", f"The most important finding is that average similarity across all pairs is {avg_sim:.4f}. "
                              f"The PCA plot reveals natural clusters — nearby items share conceptual domains, confirming the model captures domain semantics."),
        ("⚖️", "Fairness", f"Limitation: <strong>all-MiniLM-L6-v2</strong> was trained primarily on English text and may not capture nuances in "
                           f"multilingual, highly technical, or domain-specific content. Similarity reflects training data patterns, not absolute semantic truth."),
    ]

    st.markdown("<div class='ct-box'>", unsafe_allow_html=True)
    for icon, title, body in ct_data:
        st.markdown(f"""
        <div class='ct-standard'>
          <div class='ct-icon'>{icon}</div>
          <div class='ct-content'>
            <strong>{title}</strong>
            <p>{body}</p>
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

else:
    # Empty state
    st.markdown("""
    <div style='text-align:center; padding: 60px 20px; color:#888AAA;'>
      <div style='font-size:4rem; margin-bottom:16px;'>🔮</div>
      <div style='font-size:1.3rem; font-weight:600; color:#E8E8FF; margin-bottom:8px;'>Ready to explore semantic space</div>
      <div style='font-size:0.95rem; max-width:420px; margin:0 auto;'>
        Enter your words or sentences above — one per line — then hit <strong style='color:#A89FFF;'>Analyse Similarity</strong>.
        <br><br>The default example compares AI/tech terms with unrelated ones, which nicely demonstrates semantic clustering.
      </div>
    </div>
    """, unsafe_allow_html=True)
