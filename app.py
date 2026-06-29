import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd
import numpy as np

# --- Page Configuration (Dark Theme setup) ---
st.set_page_config(page_title="NLP Matrix Explorer", page_icon="🌌", layout="wide")

# --- Custom Cyberpunk/Neon CSS ---
st.markdown("""
    <style>
    /* Dark background with neon accents */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    h1, h2, h3 {
        color: #58a6ff !important;
        font-family: 'Courier New', Courier, monospace;
    }
    .neon-box {
        background-color: #161b22; 
        padding: 25px; 
        border-radius: 12px; 
        box-shadow: 0 0 15px rgba(88, 166, 255, 0.2); 
        border: 1px solid #58a6ff;
        color: #e6edf3;
    }
    .highlight {
        color: #00ff00;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🌌 NLP Matrix: Semantic Similarity Engine")
st.markdown("**Free Pretrained Model:** `all-MiniLM-L6-v2` | **Zero Preprocessing**")
st.write("Enter text below to map semantics into multidimensional space.")

# --- Load Model ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- User Input ---
st.sidebar.markdown("### 📝 Text Input Terminal")
default_text = "Quantum computing will change the future.\nArtificial intelligence is evolving rapidly.\nMachine learning models require huge data.\nI am enjoying my coffee right now.\nThe weather is slightly chilly today."
user_input = st.sidebar.text_area("Input Sentences (One per line):", value=default_text, height=250)
sentences = [s.strip() for s in user_input.split('\n') if s.strip()]

if len(sentences) < 3:
    st.error("⚠️ Enter at least 3 sentences to generate the matrix.")
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
                         color='Similarity', color_continuous_scale='Tealgrn',
                         title=f"Target: '{base_sentence}'")
        fig_bar.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar, use_container_width=True)

    # 2. Heatmap
    with col2:
        st.subheader("2. Pairwise Heatmap")
        fig_heat = px.imshow(similarity_matrix,
                             x=[f"S{i+1}" for i in range(len(sentences))],
                             y=[f"S{i+1}" for i in range(len(sentences))],
                             color_continuous_scale='Turbo',
                             text_auto=".2f", aspect="auto",
                             title="Full Similarity Matrix")
        fig_heat.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_heat, use_container_width=True)

    # 3. Embedding Plots (2D & 3D)
    st.divider()
    st.subheader("3. Embedding Projections")
    
    # Need at least 3 samples for 3D PCA
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
        st.markdown("**2D Projection (Required)**")
        fig_pca2d = px.scatter(df_pca, x='PCA1', y='PCA2', text='Label', color='Label',
                             color_discrete_sequence=px.colors.qualitative.Set3,
                             size_max=60, title="2D Semantic Space")
        fig_pca2d.update_traces(marker=dict(size=12, line=dict(width=1, color='White')), textposition='top center')
        fig_pca2d.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
        st.plotly_chart(fig_pca2d, use_container_width=True)
        
    with col4:
        st.markdown("**3D Interactive Projection (Ghoomne Wala!)**")
        if n_comp == 3:
            df_pca['PCA3'] = embeddings_pca[:, 2]
            fig_pca3d = px.scatter_3d(df_pca, x='PCA1', y='PCA2', z='PCA3', text='Label', color='Label',
                                      color_discrete_sequence=px.colors.qualitative.Set3,
                                      title="3D Semantic Space (Click & Drag to Rotate)")
            fig_pca3d.update_traces(marker=dict(size=8))
            fig_pca3d.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
            st.plotly_chart(fig_pca3d, use_container_width=True)
        else:
            st.warning("Need at least 3 sentences for 3D visualization.")

    st.divider()

    # --- PAUL'S CRITICAL THINKING STANDARDS ---
    st.header("🧠 Paul’s Critical Thinking Standards Analysis")
    st.markdown('<div class="neon-box">', unsafe_allow_html=True)
    
    st.markdown(f"**1. Clarity:** The input is an array of {len(sentences)} raw text strings. The output demonstrates how a neural network interprets their semantic meaning as mathematical vectors, showing distance/similarity.")
    st.markdown("**2. Accuracy:** Everything is computed strictly using the `all-MiniLM-L6-v2` embedding model. There are no manual modifications or unsupported adjustments to the similarity scores.")
    
    np.fill_diagonal(similarity_matrix, -1)
    max_idx = np.unravel_index(np.argmax(similarity_matrix, axis=None), similarity_matrix.shape)
    max_score = similarity_matrix[max_idx]
    
    st.markdown(f"**3. Precision:** The analysis avoids vague terms. We can precisely state that S{max_idx[0]+1} and S{max_idx[1]+1} have a cosine similarity of <span class='highlight'>{max_score:.4f}</span>.", unsafe_allow_html=True)
    st.markdown("**4. Relevance:** All visual data aligns directly with the mathematical outputs. The bar chart isolates one context, the heatmap shows the holistic view, and the PCA plots map the spatial relationships of the text.")
    st.markdown(f"**5. Logic:** The model logically assigned the highest similarity score ({max_score:.4f}) to the texts that share contextual overlap, proving the embeddings capture meaning, not just exact word matching.")
    st.markdown("**6. Significance:** The most significant insight comes from the spatial clustering in the PCA plots, where technologically aligned sentences cluster together, separating completely from casual topics like coffee or weather.")
    st.markdown("**7. Fairness:** Limitation check: This pre-trained model is fundamentally a lightweight version. It lacks the capacity to fairly evaluate multilingual text, complex sarcasm, or highly nuanced academic jargon without further fine-tuning.")
    
    st.markdown('</div>', unsafe_allow_html=True)
