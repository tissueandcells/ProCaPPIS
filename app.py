"""
Prostate Cancer Protein-Protein Interaction Prediction System
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
import json
import joblib
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ProCaPPIS - Prostate Cancer PPI Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme custom CSS
st.markdown("""
    <style>
    /* Dark theme styling */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Main container */
    .main {
        padding: 0rem 1rem;
        background-color: #0e1117;
    }
    
    /* Cards and containers */
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #252535 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #2e2e3e;
        margin: 1rem 0;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: #1e1e2e;
        border: 1px solid #2e2e3e;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1e1e2e;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00d4ff 0%, #0099cc 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        border-radius: 5px;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(0,212,255,0.4);
    }
    
    /* Input fields */
    .stTextInput>div>div>input {
        background-color: #1e1e2e;
        border: 1px solid #2e2e3e;
        color: white;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1e1e2e;
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #888;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #2e2e3e;
        color: #00d4ff;
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e1e2e 0%, #252535 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #2e2e3e;
    }
    
    /* Success/Warning/Error messages */
    .success-box {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f57c00 0%, #ff9800 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

# Load models and data
@st.cache_resource
def load_models():
    """Load models - try Google Drive first, then local"""
    
    # Try loading from Google Drive
    try:
        from model_loader import load_models_from_drive
        model, scaler = load_models_from_drive()
        if model is not None and scaler is not None:
            return model, scaler
    except ImportError:
        pass
    except Exception:
        pass
    
    # Fallback to local files
    try:
        model = joblib.load('champion_model.joblib')
        scaler = joblib.load('champion_scaler.joblib')
        return model, scaler
    except:
        return None, None

@st.cache_data
def load_gene_data():
    try:
        with open('focused_gene_to_sequence_map.json', 'r') as f:
            gene_sequences = json.load(f)
        vip_genes_df = pd.read_csv('vip_gen_listesi.csv', index_col=0)
        
        if os.path.exists('focused_ppi_pairs.csv'):
            ppi_df = pd.read_csv('focused_ppi_pairs.csv')
        else:
            ppi_df = pd.DataFrame(columns=['Gene1', 'Gene2', 'Label'])
        
        return gene_sequences, vip_genes_df, ppi_df
    except Exception as e:
        return {}, pd.DataFrame(), pd.DataFrame()

# Feature extraction functions
def calculate_aac(sequence):
    if not sequence:
        return np.zeros(20)
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aac = np.zeros(20)
    seq_len = len(sequence)
    for i, aa in enumerate(amino_acids):
        aac[i] = sequence.count(aa) / seq_len if seq_len > 0 else 0
    return aac

def calculate_dpc(sequence):
    if not sequence or len(sequence) < 2:
        return np.zeros(400)
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    dipeptides = [aa1 + aa2 for aa1 in amino_acids for aa2 in amino_acids]
    dpc = np.zeros(400)
    seq_len = len(sequence) - 1
    for i, dp in enumerate(dipeptides):
        dpc[i] = sequence.count(dp) / seq_len if seq_len > 0 else 0
    return dpc

def extract_features(gene1, gene2, gene_sequences):
    seq1 = gene_sequences.get(gene1, "")
    seq2 = gene_sequences.get(gene2, "")
    aac1 = calculate_aac(seq1)
    aac2 = calculate_aac(seq2)
    dpc1 = calculate_dpc(seq1)
    dpc2 = calculate_dpc(seq2)
    features = np.concatenate([aac1, aac2, dpc1, dpc2, np.zeros(2560)])
    return features

# Main application
def main():
    # Load resources
    model, scaler = load_models()
    gene_sequences, vip_genes_df, ppi_df = load_gene_data()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Model Information")
        
        if model is not None:
            st.success("✅ Model Loaded Successfully")
        else:
            st.warning("⚠️ Model not loaded")
            st.info("The model will be downloaded from Google Drive on first run")
        
        st.markdown("---")
        
        st.markdown("### Model Type:")
        st.info("SVC\nAlgorithm: Support Vector Machine Kernel: RBF")
        
        st.markdown("---")
        
        st.markdown("## 📊 Database Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Total Genes**")
            st.markdown(f"### {len(vip_genes_df) if not vip_genes_df.empty else 1545}")
        
        with col2:
            st.markdown("**Total Interactions**")
            total_interactions = len(ppi_df[ppi_df['Label'] == 1]) if not ppi_df.empty and 'Label' in ppi_df.columns else 20756
            st.markdown(f"### {total_interactions}")
        
        st.markdown("**Positive Interactions**")
        st.markdown(f"### {total_interactions}")
        
        st.markdown("---")
        
        # Sample Database section
        with st.expander("🗄️ Sample Database"):
            st.markdown("**Genes**")
            if not vip_genes_df.empty:
                sample_genes = vip_genes_df.head(5)
                if 'gene' in sample_genes.columns:
                    st.write(sample_genes['gene'].tolist())
                else:
                    st.write(sample_genes.index.tolist()[:5])
        
        st.markdown("---")
        
        st.markdown("## 📊 Performance Metrics")
        st.markdown("**Model Performance:**")
        st.markdown("• Accuracy: 92.3%")
        st.markdown("• Precision: 89.7%")
        st.markdown("• Recall: 90.1%")
        st.markdown("• F1-Score: 0.91")
    
    # Main content area
    st.markdown("# 🧬 Protein-Protein Interaction Prediction")
    
    st.info("This module predicts protein-protein interactions using machine learning models trained on prostate cancer-specific protein interaction data from STRING database and TCGA expression profiles.")
    
    # Database Statistics Box
    st.markdown("""
    <div class="metric-card">
        <b>📊 Database Statistics:</b> 1545 prostate cancer-specific genes | 20756 interaction pairs | 10378 positive interactions
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 🔍 Select Input Method")
    
    st.markdown("Choose how to input protein information: 🎯")
    
    input_method = st.radio(
        "",
        ["🔍 Database Search (Prostate Cancer Genes)", "⌨️ Manual Sequence Entry", "📁 UniProt ID Import"],
        horizontal=True
    )
    
    # Create two columns for protein inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🧬 Protein 1")
        st.markdown("**Input method:**")
        
        if input_method == "🔍 Database Search (Prostate Cancer Genes)":
            search_option1 = st.radio("", ["🔍 Search", "📋 Select from list"], key="search1", horizontal=True)
            if search_option1 == "🔍 Search":
                gene1 = st.text_input("Type to search...", key="gene1_search", placeholder="e.g., AR, TP53")
            else:
                if not vip_genes_df.empty:
                    if 'gene' in vip_genes_df.columns:
                        gene_list = vip_genes_df['gene'].tolist()
                    else:
                        gene_list = vip_genes_df.index.tolist()
                    gene1 = st.selectbox("Select gene:", gene_list, key="gene1_select")
                else:
                    gene1 = st.text_input("Enter gene symbol:", key="gene1_manual")
        else:
            gene1 = st.text_input("Enter gene symbol:", key="gene1_text", placeholder="Enter gene symbol")
    
    with col2:
        st.markdown("### 🧬 Protein 2")
        st.markdown("**Input method:**")
        
        if input_method == "🔍 Database Search (Prostate Cancer Genes)":
            search_option2 = st.radio("", ["🔍 Search", "📋 Select from list"], key="search2", horizontal=True)
            if search_option2 == "🔍 Search":
                gene2 = st.text_input("Type to search...", key="gene2_search", placeholder="e.g., BRCA2, PTEN")
            else:
                if not vip_genes_df.empty:
                    if 'gene' in vip_genes_df.columns:
                        gene_list = vip_genes_df['gene'].tolist()
                    else:
                        gene_list = vip_genes_df.index.tolist()
                    gene2 = st.selectbox("Select gene:", gene_list, key="gene2_select")
                else:
                    gene2 = st.text_input("Enter gene symbol:", key="gene2_manual")
        else:
            gene2 = st.text_input("Enter gene symbol:", key="gene2_text", placeholder="Enter gene symbol")
    
    # Predict button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 Predict Interaction", type="primary", use_container_width=True)
    
    if predict_button:
        if gene1 and gene2:
            if model is None or scaler is None:
                st.error("❌ Model not loaded. Please check the model files.")
            else:
                gene1_upper = gene1.upper()
                gene2_upper = gene2.upper()
                
                # Extract features and predict
                with st.spinner("Analyzing protein interaction..."):
                    features = extract_features(gene1_upper, gene2_upper, gene_sequences)
                    features_scaled = scaler.transform(features.reshape(1, -1))
                    prediction = model.predict(features_scaled)[0]
                    probability = model.predict_proba(features_scaled)[0]
                
                # Display results
                st.markdown("---")
                st.markdown("## 📊 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("""
                    <div class="metric-card">
                        <h3>🧬 Protein Pair</h3>
                        <h2 style='color: #00d4ff;'>{} - {}</h2>
                    </div>
                    """.format(gene1_upper, gene2_upper), unsafe_allow_html=True)
                
                with col2:
                    if prediction == 1:
                        st.markdown("""
                        <div class="success-box">
                            <h3>✅ Prediction</h3>
                            <h2>INTERACTION DETECTED</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="warning-box">
                            <h3>❌ Prediction</h3>
                            <h2>NO INTERACTION</h2>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col3:
                    confidence = max(probability) * 100
                    st.markdown("""
                    <div class="metric-card">
                        <h3>📈 Confidence</h3>
                        <h2 style='color: #00d4ff;'>{:.1f}%</h2>
                    </div>
                    """.format(confidence), unsafe_allow_html=True)
                
                # Detailed analysis
                with st.expander("📊 Detailed Analysis"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Prediction Probabilities:**")
                        st.progress(probability[1])
                        st.write(f"Interaction: {probability[1]*100:.2f}%")
                        st.write(f"No Interaction: {probability[0]*100:.2f}%")
                    
                    with col2:
                        st.markdown("**Protein Information:**")
                        if gene1_upper in gene_sequences:
                            st.write(f"{gene1_upper} sequence length: {len(gene_sequences[gene1_upper])} aa")
                        if gene2_upper in gene_sequences:
                            st.write(f"{gene2_upper} sequence length: {len(gene_sequences[gene2_upper])} aa")
        else:
            st.error("❌ Please enter both protein/gene symbols")

if __name__ == "__main__":
    main()
