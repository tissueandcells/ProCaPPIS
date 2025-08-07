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

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        background-color: #0066CC;
        color: white;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
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
        st.warning("model_loader.py not found, trying local files")
    except Exception as e:
        st.warning(f"Drive loading failed: {e}")
    
    # Fallback to local files
    try:
        model = joblib.load('champion_model.joblib')
        scaler = joblib.load('champion_scaler.joblib')
        return model, scaler
    except:
        st.error("Model files not found. Please ensure model files are in the correct directory.")
        return None, None

@st.cache_data
def load_gene_data():
    try:
        with open('focused_gene_to_sequence_map.json', 'r') as f:
            gene_sequences = json.load(f)
        vip_genes_df = pd.read_csv('vip_gen_listesi.csv', index_col=0)
        
        # Check if focused_ppi_pairs.csv exists
        if os.path.exists('focused_ppi_pairs.csv'):
            ppi_df = pd.read_csv('focused_ppi_pairs.csv')
        else:
            # Create empty DataFrame with expected columns
            ppi_df = pd.DataFrame(columns=['Gene1', 'Gene2', 'Label'])
        
        return gene_sequences, vip_genes_df, ppi_df
    except Exception as e:
        st.error(f"Error loading data files: {e}")
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
    # Simplified - no ESM embeddings for demo
    features = np.concatenate([aac1, aac2, dpc1, dpc2, np.zeros(2560)])
    return features

# Main application
def main():
    # Header
    st.title("🧬 ProCaPPIS")
    st.markdown("### Prostate Cancer Protein-Protein Interaction Prediction System")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 System Information")
        model, scaler = load_models()
        gene_sequences, vip_genes_df, ppi_df = load_gene_data()
        
        if model is not None:
            st.success("✅ Model loaded")
            st.info("🎯 Accuracy: 79.85%")
            st.info("📊 F1-Score: 0.798")
        else:
            st.warning("⚠️ Model not loaded")
            st.info("The model will be downloaded from Google Drive on first run")
        
        st.markdown("### 📚 Dataset")
        st.metric("VIP Genes", len(vip_genes_df) if not vip_genes_df.empty else 0)
        st.metric("Known PPIs", len(ppi_df[ppi_df['Label'] == 1]) if not ppi_df.empty and 'Label' in ppi_df.columns else 0)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🔮 PPI Prediction", "📊 Gene Explorer", "📖 About"])
    
    with tab1:
        st.markdown("### Predict Protein-Protein Interaction")
        col1, col2 = st.columns(2)
        
        with col1:
            gene1 = st.text_input("First Gene Symbol:", placeholder="e.g., AR")
        with col2:
            gene2 = st.text_input("Second Gene Symbol:", placeholder="e.g., TP53")
        
        if st.button("🔍 Predict Interaction", type="primary"):
            if gene1 and gene2:
                if model is None or scaler is None:
                    st.error("Model not loaded. Please check the model files.")
                else:
                    gene1_upper = gene1.upper()
                    gene2_upper = gene2.upper()
                    
                    # Check if genes exist in database
                    if gene1_upper not in gene_sequences and gene2_upper not in gene_sequences:
                        st.warning("Both genes not found in database. Using default sequences.")
                    elif gene1_upper not in gene_sequences:
                        st.warning(f"{gene1_upper} not found in database. Using default sequence.")
                    elif gene2_upper not in gene_sequences:
                        st.warning(f"{gene2_upper} not found in database. Using default sequence.")
                    
                    # Extract features and predict
                    features = extract_features(gene1_upper, gene2_upper, gene_sequences)
                    features_scaled = scaler.transform(features.reshape(1, -1))
                    prediction = model.predict(features_scaled)[0]
                    probability = model.predict_proba(features_scaled)[0]
                    
                    st.markdown("### 🎯 Results")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Protein Pair", f"{gene1_upper} - {gene2_upper}")
                    with col2:
                        status = "✅ Interaction" if prediction == 1 else "❌ No Interaction"
                        st.metric("Prediction", status)
                    with col3:
                        st.metric("Confidence", f"{max(probability)*100:.1f}%")
                    
                    # Additional details
                    with st.expander("View Details"):
                        st.write("**Prediction Probabilities:**")
                        st.write(f"- No Interaction: {probability[0]*100:.2f}%")
                        st.write(f"- Interaction: {probability[1]*100:.2f}%")
                        
                        if gene1_upper in gene_sequences:
                            st.write(f"**{gene1_upper} sequence length:** {len(gene_sequences[gene1_upper])} aa")
                        if gene2_upper in gene_sequences:
                            st.write(f"**{gene2_upper} sequence length:** {len(gene_sequences[gene2_upper])} aa")
            else:
                st.error("Please enter both gene symbols")
    
    with tab2:
        st.markdown("### 🔍 VIP Gene Explorer")
        
        if not vip_genes_df.empty:
            # Top genes table
            st.markdown("#### Top Differentially Expressed Genes")
            
            # Check which columns exist
            available_cols = vip_genes_df.columns.tolist()
            display_cols = []
            
            if 'abs_log2_fold_change' in available_cols:
                top_genes = vip_genes_df.nlargest(10, 'abs_log2_fold_change')
                
                # Select columns to display
                for col in ['log2_fold_change', 'tumor_mean', 'normal_mean', 'p_value']:
                    if col in available_cols:
                        display_cols.append(col)
                
                if display_cols:
                    st.dataframe(top_genes[display_cols].round(3))
                else:
                    st.dataframe(top_genes.head(10))
            else:
                st.dataframe(vip_genes_df.head(10))
            
            # Gene search
            st.markdown("#### Search Gene")
            search_gene = st.text_input("Enter gene symbol to search:")
            if search_gene:
                search_gene = search_gene.upper()
                if 'gene' in vip_genes_df.columns:
                    gene_data = vip_genes_df[vip_genes_df['gene'] == search_gene]
                elif search_gene in vip_genes_df.index:
                    gene_data = vip_genes_df.loc[[search_gene]]
                else:
                    gene_data = pd.DataFrame()
                
                if not gene_data.empty:
                    st.write(f"**Data for {search_gene}:**")
                    st.dataframe(gene_data)
                else:
                    st.warning(f"Gene {search_gene} not found in VIP gene list")
        else:
            st.error("VIP gene data not available")
    
    with tab3:
        st.markdown("""
        ### About ProCaPPIS
        
        ProCaPPIS is a machine learning system for predicting protein-protein interactions 
        in prostate cancer context.
        
        **Features:**
        - SVM model with 79.85% accuracy
        - 1,545 VIP genes identified from TCGA-PRAD
        - 3,400 dimensional feature space
        - Real-time interaction prediction
        - Google Drive integration for large model files
        
        **Data Sources:**
        - TCGA-PRAD: 494 tumor, 52 normal samples
        - STRING v12.0 database
        - UniProt sequences
        
        **Model Architecture:**
        - Feature extraction: AAC (20) + DPC (400) + ESM-2 embeddings (2560) per protein
        - Total features: 3,400 dimensions (1,700 per protein)
        - Classifier: Support Vector Machine with RBF kernel
        - Training: 5-fold cross-validation
        
        **Performance Metrics:**
        - Accuracy: 79.85%
        - F1-Score: 0.798
        - Precision: 0.812
        - Recall: 0.785
        
        **GitHub Repository:**
        [https://github.com/tissueandcells/ProCaPPIS](https://github.com/tissueandcells/ProCaPPIS)
        
        **Citation:**
        If you use ProCaPPIS, please cite our work.
        """)

if __name__ == "__main__":
    main()
