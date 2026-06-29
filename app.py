import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# --- Page Configuration (For that eye-catching look!) ---
st.set_page_config(page_title="Text Similarity Explorer", page_icon="✨", layout="wide")

# --- Custom CSS for some extra beauty ---
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    h1, h2, h3 {color: #ff4b4b;}
    .paul-box {background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-left: 5px solid #ff4b4b;}
    </style>
    """, unsafe_allow_html=True)

st.title("✨ Beautiful NLP: Text Similarity Explorer")
st.markdown("**Free Pretrained Model:** `all-MiniLM-L6-v2` | **No Preprocessing Used**")
st.write("Enter your sentences below to see the magic of embeddings and interactive graphs! 🚀")

# --- Load Model (Free & Pretrained) ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- User Input ---
st.sidebar.header("📝 Input Section")
default_text = "I love learning about artificial intelligence.\nMachine learning is fascinating.\nDeep learning creates smart systems.\nThe weather is very pleasant today.\nIt is raining outside."
user_input = st.sidebar.text_area("Enter sentences (one per line):", value=default_text, height=200)
sentences = [s.strip() for s in user_input.split('\n') if s.strip()]

if len(sentences) < 3:
    st.warning("⚠️ Please enter at least 3 sentences to generate meaningful graphs.")
else:
    # --- Generate Embeddings (No manual preprocessing!) ---
    embeddings = model.encode(sentences)
    similarity_matrix = cosine_similarity(embeddings)
    
    st.divider()
    
    # --- GRAPHS SECTION ---
    st.header("📊 Interactive Visualizations")
    col1, col2 = st.columns(2)
    
    # 1. Bar Chart: Top Similarities to the First Sentence
    with col1:
        st.subheader("1. Similarity to Base Sentence")
        base_sentence = sentences[0]
        scores = similarity_matrix[0][1:] # Skip comparing with itself
        comparison_sentences = sentences[1:]
        
        df_bar = pd.DataFrame({'Sentence': comparison_sentences, 'Similarity Score': scores})
        df_bar = df_bar.sort_values('Similarity Score', ascending=False)
        
        fig_bar = px.bar(df_bar, x='Similarity Score', y='Sentence', orientation='h', 
                         color='Similarity Score', color_continuous_scale='Sunsetdark',
                         title=f"Compared to: '{base_sentence}'")
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)

    # 2. Heatmap: Pairwise Similarity
    with col2:
        st.subheader("2. Pairwise Similarity Heatmap")
        fig_heat = px.imshow(similarity_matrix,
                             x=[f"S{i+1}" for i in range(len(sentences))],
                             y=[f"S{i+1}" for i in range(len(sentences))],
                             color_continuous_scale='Plasma',
                             text_auto=".2f", aspect="auto",
                             title="All-vs-All Sentence Similarity")
        # Hover info
        fig_heat.update_traces(hovertemplate="Sentence X: %{x}<br>Sentence Y: %{y}<br>Similarity: %{z}")
        st.plotly_chart(fig_heat, use_container_width=True)

    # 3. 2D Embedding Plot (PCA)
    st.subheader("3. 2D Embedding Plot (PCA)")
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    df_pca = pd.DataFrame({
        'PCA1': embeddings_2d[:, 0],
        'PCA2': embeddings_2d[:, 1],
        'Sentence': sentences,
        'Label': [f"S{i+1}" for i in range(len(sentences))]
    })
    
    fig_pca = px.scatter(df_pca, x='PCA1', y='PCA2', text='Label', color='Label',
                         color_discrete_sequence=px.colors.qualitative.Pastel,
                         size_max=60, title="Sentences in 2D Space (Closer dots = Similar meaning)",
                         hover_data={'PCA1':False, 'PCA2':False, 'Sentence':True, 'Label':False})
    
    fig_pca.update_traces(marker=dict(size=15, line=dict(width=2, color='DarkSlateGrey')), 
                          textposition='top center')
    st.plotly_chart(fig_pca, use_container_width=True)

    st.divider()

    # --- PAUL'S CRITICAL THINKING STANDARDS ---
    st.header("🧠 Paul’s Critical Thinking Standards Analysis")
    st.markdown('<div class="paul-box">', unsafe_allow_html=True)
    
    st.markdown(f"**1. Clarity:** The input provided is a list of {len(sentences)} distinct sentences. The output represents how mathematically similar these sentences are in meaning, based on vector embeddings.")
    st.markdown("**2. Accuracy:** The results are generated using the `all-MiniLM-L6-v2` pretrained model from Hugging Face. No unsupported claims are made; the scores are direct cosine similarity calculations from the model's embeddings.")
    
    # Getting the most similar pair dynamically for Precision & Logic
    np.fill_diagonal(similarity_matrix, -1) # Ignore self-similarity
    max_idx = np.unravel_index(np.argmax(similarity_matrix, axis=None), similarity_matrix.shape)
    max_score = similarity_matrix[max_idx]
    
    st.markdown(f"**3. Precision:** Instead of saying sentences are just 'highly similar', the model provides an exact mathematical score. For example, Sentence {max_idx[0]+1} and Sentence {max_idx[1]+1} have an exact similarity score of **{max_score:.4f}** (on a scale of -1 to 1).")
    st.markdown("**4. Relevance:** The three graphs directly support the results. The Bar Chart shows top matches to a single text, the Heatmap provides a complete similarity matrix, and the PCA plot visually groups related topics together.")
    st.markdown(f"**5. Logic:** The top matching result (Score: {max_score:.4f}) makes logical sense because the model maps words with similar contextual meanings (e.g., AI and Machine Learning) to similar vector spaces.")
    st.markdown("**6. Significance:** The most significant result is seen in the PCA scatter plot, where sentences discussing similar topics cluster tightly together, while unrelated sentences (e.g., about weather) are pushed far away.")
    st.markdown("**7. Fairness:** It is important to note a limitation: This pretrained model is small and optimized for English. It might not understand highly complex idioms, domain-specific jargon, or non-English text accurately without further training (which was omitted per assignment rules).")
    
    st.markdown('</div>', unsafe_allow_html=True)
