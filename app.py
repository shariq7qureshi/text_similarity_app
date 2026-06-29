import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="Semantic Explorer", page_icon="🌊", layout="wide")

# --- Custom Glassmorphism CSS ---
st.markdown("""
    <style>
    /* Beautiful Ocean/Sunset Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
        color: #1a1a2e;
    }
    /* Stylish Headers */
    h1, h2, h3 {
        color: #1a1a2e !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 800 !important;
    }
    /* Glass Effect Box for Paul's Standards */
    .glass-box {
        background: rgba(255, 255, 255, 0.45);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 25px; 
        border-radius: 15px; 
        border: 1px solid rgba(255, 255, 255, 0.6);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15); 
        color: #1a1a2e;
        font-size: 1.1em;
    }
    .highlight {
        color: #d63031;
        font-weight: bold;
        background-color: rgba(255,255,255,0.7);
        padding: 2px 6px;
        border-radius: 4px;
    }
    /* Make standard Streamlit text dark for readability */
    .stMarkdown, .stText {
        color: #1a1a2e;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🌊 Semantic Explorer: The Glass UI")
st.markdown("**Free Pretrained Model:** `all-MiniLM-L6-v2` | **Zero Preprocessing**")
st.write("Dive into the meaning of words with this interactive matrix.")

# --- Load Model ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- User Input ---
st.sidebar.markdown("### 📝 Input Dashboard")
default_text = "The ocean breeze is incredibly calming today.\nWater bodies have a soothing effect on the mind.\nArtificial intelligence algorithms are getting smarter.\nMachine learning makes automation easy.\nI love taking a walk on the beach."
user_input = st.sidebar.text_area("Input Sentences (One per line):", value=default_text, height=250)
sentences = [s.strip() for s in user_input.split('\n') if s.strip()]

if len(sentences) < 3:
    st.error("⚠️ Enter at least 3 sentences to unlock the visualizations.")
else:
    # --- Generate Embeddings ---
    embeddings = model.encode(sentences)
    similarity_matrix = cosine_similarity(embeddings)
    
    st.divider()
    
    # --- GRAPHS SECTION ---
    st.header("📊 Interactive Visualizations")
    col1, col2 = st.columns(2)
    
    # 1. Bar Chart
    with col1:
        st.subheader("1. Proximity to Base Sentence")
        base_sentence = sentences[0]
        scores = similarity_matrix[0][1:]
        comparison_sentences = sentences[1:]
        
        df_bar = pd.DataFrame({'Sentence': comparison_sentences, 'Similarity': scores})
        df_bar = df_bar.sort_values('Similarity', ascending=False)
        
        fig_bar = px.bar(df_bar, x='Similarity', y='Sentence', orientation='h', 
                         color='Similarity', color_continuous_scale='Purp',
                         title=f"Target: '{base_sentence}'")
        fig_bar.update_layout(plot_bgcolor='rgba(255,255,255,0.5)', paper_bgcolor='rgba(255,255,255,0.5)', font_color='#1a1a2e')
        st.plotly_chart(fig_bar, use_container_width=True)

    # 2. Heatmap
    with col2:
        st.subheader("2. Pairwise Heatmap")
        fig_heat = px.imshow(similarity_matrix,
                             x=[f"S{i+1}" for i in range(len(sentences))],
                             y=[f"S{i+1}" for i in range(len(sentences))],
                             color_continuous_scale='Mint',
                             text_auto=".2f", aspect="auto",
                             title="Full Similarity Matrix")
        fig_heat.update_layout(plot_bgcolor='rgba(255,255,255,0.5)', paper_bgcolor='rgba(255,255,255,0.5)', font_color='#1a1a2e')
        st.plotly_chart(fig_heat, use_container_width=True)

    # 3. Embedding Plots (2D & 3D)
    st.divider()
    st.subheader("3. Embedding Projections (The Fun Part)")
    
    # PCA Calculation
    n_comp = 3 if len(sentences) >= 3 else 2
    pca = PCA(n_components=n_comp)
    embeddings_pca = pca.fit_transform(embeddings)
    
    df_pca = pd.DataFrame({
        'Sentence': sentences,
        'Label': [f"S{i+1}" for i in range(len(sentences))],
        'PCA1': embeddings_pca[:, 0],
        'PCA2': embeddings_pca[:, 1]
    })
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("**2D Projection (Quiz Requirement)**")
        fig_pca2d = px.scatter(df_pca, x='PCA1', y='PCA2', text='Label', color='Label',
                             color_discrete_sequence=px.colors.qualitative.Bold,
                             size_max=60, title="2D Semantic Map")
        fig_pca2d.update_traces(marker=dict(size=14, line=dict(width=2, color='DarkSlateGrey')), textposition='top center')
        fig_pca2d.update_layout(plot_bgcolor='rgba(255,255,255,0.5)', paper_bgcolor='rgba(255,255,255,0.5)', font_color='#1a1a2e', showlegend=False)
        st.plotly_chart(fig_pca2d, use_container_width=True)
        
    with col4:
        st.markdown("**3D Interactive Projection (Rotate Me!)**")
        if n_comp == 3:
            df_pca['PCA3'] = embeddings_pca[:, 2]
            fig_pca3d = px.scatter_3d(df_pca, x='PCA1', y='PCA2', z='PCA3', text='Label', color='Label',
                                      color_discrete_sequence=px.colors.qualitative.Bold,
                                      title="3D Semantic Space")
            fig_pca3d.update_traces(marker=dict(size=10, line=dict(width=2, color='DarkSlateGrey')))
            fig_pca3d.update_layout(plot_bgcolor='rgba(255,255,255,0)', paper_bgcolor='rgba(255,255,255,0.5)', font_color='#1a1a2e', showlegend=False)
            st.plotly_chart(fig_pca3d, use_container_width=True)

    st.divider()

    # --- PAUL'S CRITICAL THINKING STANDARDS ---
    st.header("🧠 Paul’s Critical Thinking Standards Analysis")
    st.markdown('<div class="glass-box">', unsafe_allow_html=True)
    
    st.markdown(f"**1. Clarity:** The inputs are {len(sentences)} basic sentences. The output clearly shows how the AI translates text into mathematical distance to determine if they mean the same thing.")
    st.markdown("**2. Accuracy:** We utilized the `all-MiniLM-L6-v2` model from HuggingFace to ensure precise, unmodified cosine similarity calculations, exactly as per the strict rules.")
    
    np.fill_diagonal(similarity_matrix, -1)
    max_idx = np.unravel_index(np.argmax(similarity_matrix, axis=None), similarity_matrix.shape)
    max_score = similarity_matrix[max_idx]
    
    st.markdown(f"**3. Precision:** The app avoids ambiguity. S{max_idx[0]+1} and S{max_idx[1]+1} don't just 'look alike'; they share a statistically exact similarity score of <span class='highlight'>{max_score:.4f}</span>.", unsafe_allow_html=True)
    st.markdown("**4. Relevance:** Each graph serves a distinct, relevant purpose: Bar charts for 1-to-many comparisons, Heatmaps for all-to-all relationships, and PCA plots for spatial grouping.")
    st.markdown(f"**5. Logic:** It is logical that the highest score ({max_score:.4f}) belongs to contextually similar sentences, proving the model maps deeper semantics rather than just matching alphabets.")
    st.markdown("**6. Significance:** The PCA scatter plot reveals the most significant pattern: AI can naturally cluster completely distinct topics (e.g., nature vs. technology) into opposite corners of a graph.")
    st.markdown("**7. Fairness:** As a limitation, this specific pretrained model is relatively small. It handles standard English perfectly but would not evaluate domain-specific technical jargon or non-English phrases fairly without fine-tuning.")
    
    st.markdown('</div>', unsafe_allow_html=True)
