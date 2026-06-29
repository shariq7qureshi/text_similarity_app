# 🧬 Semantic Matrix Explorer: High-Dimensional NLP Visualizer

A cutting-edge, modern Streamlit web application designed to map and evaluate textual semantics using state-of-the-art transformer embeddings. This application evaluates word and sentence proximities through mathematical matrix operations and visualizes the results against **Paul's Critical Thinking Standards**.

---

## 🌍 Deployed Application Link
👉 https://textsimilarityapp-tmzd5ftpeuk9cjt8qtltfa.streamlit.app/

---

## 🛠️ Project Architecture & Rules Compliance
As per the strict evaluation criteria of the assignment, this project strictly adheres to the following rules:
* **Zero Preprocessing:** No manual tokenization, stopword removal, stemming, lemmatization, or text cleaning is performed. Text inputs are fed directly into the model pipeline.
* **No Training:** Powered entirely by a free, open-source pretrained model.
* **No Paid APIs:** Completely functional without requiring OpenAI, Anthropic, or any other paid subscription keys.

### Core Model Specifications
* **Model Name:** `all-MiniLM-L6-v2` (via Hugging Face `sentence-transformers`)
* **Embedding Space Dimensions:** 384-dimensional dense vectors mapped to output semantic contexts natively.

---

## 🚀 Advanced Features

### 1. 🎯 Paul's Standards Performance Radar Chart
An innovative implementation tailored to display mathematical compliance metrics for **Clarity, Accuracy, Precision, Relevance, Logic, Significance, and Fairness**. Instead of generic graphs, this custom radar plot provides visual proof of how structurally optimized the text data is.

### 2. 🤖 Smart AI Insights Engine
An added algorithmic layer that instantly scans the computed matrix to output the **Highest Semantic Match** and the **Lowest Semantic Match** inside prominent, high-contrast visual alert boxes.

### 3. 🛰️ 3D Rotatable Semantic Space
A dimensionality reduction technique using Principal Component Analysis (PCA) that transforms high-dimensional embeddings into a fully interactive, 3-dimensional spatial map. Users can click, drag, scroll, and rotate the graph to analyze sentence clustering.

### 4. 🎛️ Real-Time Proximity Threshold Slider
An interactive control slider that allows users to filter out low-matching items dynamically from the relative distance horizontal bar chart.

### 5. 💾 Mathematical CSV Matrix Exporter
A professional data feature allowing users to immediately compile and export their dynamic similarity matrix data as a local `.csv` file.

---

## 📂 Repository Structure

The repository must contain the following core components to ensure smooth deployment on Streamlit Community Cloud:
```text
├── app.py             # Main Streamlit script containing UI and layout configuration
├── requirements.txt   # Complete set of application dependencies and framework constraints
└── README.md          # Comprehensive documentation and project evaluation guide
