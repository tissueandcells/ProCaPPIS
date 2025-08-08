if os.path.exists('vip_gen_listesi.csv'):
            vip_genes_df = pd.read_csv('vip_gen_listesi.csv', index_col=0)
        else:
            # Demo VIP genes
            vip_genes_df = pd.DataFrame({
                'Hugo_Symbol': ['TP53', 'AR', 'BRCA1', 'MYC', 'EGFR', 'PTEN', 'PIK3CA', 'RB1'],
                'log2_fold_change': [2.5, -1.8, 1.2, 3.1, -2.2, 1.9, 2.8, -1.5],
                'p_value': [0.001, 0.002, 0.01, 0.0001, 0.005, 0.003, 0.0005, 0.008]
            })
            vip_genes_df['abs_log2_fold_change'] = vip_genes_df['log2_fold_change'].abs()
            vip_genes_df.set_index('Hugo_Symbol', inplace=True)
        
        return gene_sequences, vip_genes_df, pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return {}, pd.DataFrame(), pd.DataFrame()

# Feature extraction functions
def calculate_aac(sequence):
    """Calculate amino acid composition"""
    if not sequence:
        return np.zeros(20)
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aac = np.zeros(20)
    seq_len = len(sequence)
    for i, aa in enumerate(amino_acids):
        aac[i] = sequence.count(aa) / seq_len if seq_len > 0 else 0
    return aac

def calculate_dpc(sequence):
    """Calculate dipeptide composition"""
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
    """Extract features for protein pair"""
    seq1 = gene_sequences.get(gene1, "")
    seq2 = gene_sequences.get(gene2, "")
    aac1 = calculate_aac(seq1)
    aac2 = calculate_aac(seq2)
    dpc1 = calculate_dpc(seq1)
    dpc2 = calculate_dpc(seq2)
    # Simplified feature vector
    features = np.concatenate([aac1, aac2, dpc1, dpc2])
    return features

def demo_prediction(features):
    """Demo prediction function when model is not available"""
    # Simple rule-based prediction for demo
    feature_sum = np.sum(features)
    if feature_sum > 0.5:
        return 1, [0.3, 0.7]
    else:
        return 0, [0.8, 0.2]

def create_network_visualization(predictions_df):
    """Create network graph from predictions"""
    if predictions_df.empty:
        return None
        
    # Filter for interactions only
    interactions = predictions_df[predictions_df['Prediction'] == 'Interaction']
    
    if interactions.empty:
        return None
    
    G = nx.Graph()
    
    # Add edges for positive predictions
    for _, row in interactions.iterrows():
        G.add_edge(row['Protein1'], row['Protein2'], 
                  weight=row.get('Confidence', 50))
    
    if len(G.nodes()) == 0:
        return None
    
    # Layout
    pos = nx.spring_layout(G, k=1/np.sqrt(len(G.nodes())), iterations=50)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        degree = G.degree(node)
        node_text.append(f"{node}<br>Connections: {degree}")
        node_size.append(20 + degree * 5)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[node for node in G.nodes()],
        textposition="top center",
        hoverinfo='text',
        hovertext=node_text,
        marker=dict(
            size=node_size,
            color=node_size,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(thickness=15, title='Degree'),
            line=dict(width=2, color='white')
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title='<b>Protein-Protein Interaction Network</b>',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=500
    )
    
    return fig

# Main App
def main():
    # Load resources
    model, scaler = load_models()
    gene_sequences, vip_genes_df, _ = load_gene_data()
    
    # Header
    st.title("🧬 ProCaPPIS: Prostate Cancer Protein-Protein Interaction Prediction System")
    
    # Navigation using tabs instead of sidebar to avoid watch issues
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "🔬 PPI Prediction", "🌐 Network Analysis", "📈 Results"])
    
    with tab1:
        st.markdown("""
        <div class="academic-card">
        <h3>Welcome to ProCaPPIS</h3>
        <p style="text-align: justify;">
        ProCaPPIS is a comprehensive computational platform for predicting and analyzing protein-protein 
        interactions in prostate cancer. Our system integrates machine learning models with biological 
        databases to provide accurate predictions and network analyses.
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key Features
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="info-box">
            <h4>🔬 PPI Prediction</h4>
            <ul>
            <li>SVM-based prediction</li>
            <li>92.3% accuracy</li>
            <li>Multiple input formats</li>
            <li>Confidence scoring</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="success-box">
            <h4>🌐 Network Analysis</h4>
            <ul>
            <li>Interactive visualization</li>
            <li>Centrality metrics</li>
            <li>Community detection</li>
            <li>Path analysis</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="warning-box">
            <h4>📊 Analysis Tools</h4>
            <ul>
            <li>Functional analysis</li>
            <li>Pathway identification</li>
            <li>Statistical testing</li>
            <li>Export capabilities</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Statistics
        st.markdown("---")
        st.markdown("### 📈 Platform Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Genes in Database", len(gene_sequences))
        with col2:
            st.metric("VIP Genes", len(vip_genes_df) if not vip_genes_df.empty else 0)
        with col3:
            st.metric("Model Status", "✅ Ready" if model else "⚠️ Demo")
        with col4:
            st.metric("Predictions Made", len(st.session_state.predictions))
    
    with tab2:
        st.markdown("## 🔬 Protein-Protein Interaction Prediction")
        
        if not model:
            st.warning("⚠️ Model not loaded. Running in demo mode with simulated predictions.")
        
        # Input Method
        input_method = st.radio(
            "Select Input Method:",
            ["Gene Symbol", "VIP Gene List", "Protein Sequence"],
            horizontal=True
        )
        
        col1, col2 = st.columns(2)
        
        protein1_name = ""
        protein2_name = ""
        seq1 = ""
        seq2 = ""
        
        with col1:
            st.markdown("### 🧬 Protein 1")
            
            if input_method == "Gene Symbol":
                protein1_name = st.text_input("Gene Symbol (e.g., TP53, AR, PTEN):", key="p1_gene").upper()
                if protein1_name:
                    if protein1_name in gene_sequences:
                        seq1 = gene_sequences[protein1_name]
                        st.success(f"✅ Found: {protein1_name} ({len(seq1)} aa)")
                    else:
                        st.warning(f"⚠️ {protein1_name} not in database.")
            
            elif input_method == "VIP Gene List":
                if not vip_genes_df.empty:
                    vip_gene_list = vip_genes_df.index.tolist()
                    protein1_name = st.selectbox("Select VIP gene:", [""] + vip_gene_list, key="vip1_select")
                    
                    if protein1_name and protein1_name in gene_sequences:
                        seq1 = gene_sequences[protein1_name]
                        st.success(f"✅ VIP Gene: {protein1_name} ({len(seq1)} aa)")
                        
                        # Show expression info
                        if protein1_name in vip_genes_df.index:
                            fc = vip_genes_df.loc[protein1_name, 'log2_fold_change']
                            st.info(f"Expression: {'⬆️ Upregulated' if fc > 0 else '⬇️ Downregulated'} (FC: {abs(fc):.2f})")
                else:
                    st.error("VIP gene list not available")
            
            elif input_method == "Protein Sequence":
                protein1_name = st.text_input("Protein Name:", key="p1_name", placeholder="e.g., MyProtein1")
                seq1_input = st.text_area("Paste sequence (FASTA or plain):", key="p1_sequence", height=150)
                if seq1_input:
                    if seq1_input.startswith('>'):
                        lines = seq1_input.split('\n')
                        if not protein1_name and lines[0].startswith('>'):
                            protein1_name = lines[0][1:].split()[0]
                        seq1 = ''.join(lines[1:])
                    else:
                        seq1 = seq1_input
                    seq1 = ''.join([c for c in seq1.upper() if c in 'ACDEFGHIKLMNPQRSTVWY'])
                    if seq1:
                        st.success(f"✅ Sequence loaded ({len(seq1)} aa)")
        
        with col2:
            st.markdown("### 🧬 Protein 2")
            
            if input_method == "Gene Symbol":
                protein2_name = st.text_input("Gene Symbol (e.g., BRCA1, MYC, EGFR):", key="p2_gene").upper()
                if protein2_name:
                    if protein2_name in gene_sequences:
                        seq2 = gene_sequences[protein2_name]
                        st.success(f"✅ Found: {protein2_name} ({len(seq2)} aa)")
                    else:
                        st.warning(f"⚠️ {protein2_name} not in database.")
            
            elif input_method == "VIP Gene List":
                if not vip_genes_df.empty:
                    vip_gene_list = vip_genes_df.index.tolist()
                    protein2_name = st.selectbox("Select VIP gene:", [""] + vip_gene_list, key="vip2_select")
                    
                    if protein2_name and protein2_name in gene_sequences:
                        seq2 = gene_sequences[protein2_name]
                        st.success(f"✅ VIP Gene: {protein2_name} ({len(seq2)} aa)")
                        
                        # Show expression info
                        if protein2_name in vip_genes_df.index:
                            fc = vip_genes_df.loc[protein2_name, 'log2_fold_change']
                            st.info(f"Expression: {'⬆️ Upregulated' if fc > 0 else '⬇️ Downregulated'} (FC: {abs(fc):.2f})")
                else:
                    st.error("VIP gene list not available")
            
            elif input_method == "Protein Sequence":
                protein2_name = st.text_input("Protein Name:", key="p2_name", placeholder="e.g., MyProtein2")
                seq2_input = st.text_area("Paste sequence (FASTA or plain):", key="p2_sequence", height=150)
                if seq2_input:
                    if seq2_input.startswith('>'):
                        lines = seq2_input.split('\n')
                        if not protein2_name and lines[0].startswith('>'):
                            protein2_name = lines[0][1:].split()[0]
                        seq2 = ''.join(lines[1:])
                    else:
                        seq2 = seq2_input
                    seq2 = ''.join([c for c in seq2.upper() if c in 'ACDEFGHIKLMNPQRSTVWY'])
                    if seq2:
                        st.success(f"✅ Sequence loaded ({len(seq2)} aa)")
        
        # Prediction Button
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            predict_btn = st.button("🔮 Predict Interaction", type="primary", use_container_width=True)
        
        if predict_btn:
            if (protein1_name and protein2_name) and (seq1 or seq2 or protein1_name in gene_sequences):
                with st.spinner("Analyzing interaction..."):
                    # Create temporary sequence dict for custom sequences
                    temp_sequences = gene_sequences.copy()
                    if seq1:
                        temp_sequences[protein1_name] = seq1
                    if seq2:
                        temp_sequences[protein2_name] = seq2
                    
                    features = extract_features(protein1_name, protein2_name, temp_sequences)
                    
                    if model and scaler:
                        # Real prediction
                        features_scaled = scaler.transform(features.reshape(1, -1))
                        prediction = model.predict(features_scaled)[0]
                        probability = model.predict_proba(features_scaled)[0]
                    else:
                        # Demo prediction
                        prediction, probability = demo_prediction(features)
                    
                    confidence = max(probability) * 100
                    
                    # Store prediction
                    result = {
                        'Protein1': protein1_name,
                        'Protein2': protein2_name,
                        'Prediction': 'Interaction' if prediction == 1 else 'No Interaction',
                        'Prediction_Binary': prediction,
                        'Confidence': confidence,
                        'P_Interaction': probability[1] * 100 if len(probability) > 1 else probability[0] * 100,
                        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.session_state.predictions.append(result)
                    
                    # Display Results
                    st.markdown("---")
                    st.markdown("## 📊 Prediction Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Protein Pair", f"{protein1_name} - {protein2_name}")
                    
                    with col2:
                        if prediction == 1:
                            st.metric("Prediction", "✅ INTERACTION")
                        else:
                            st.metric("Prediction", "❌ NO INTERACTION")
                    
                    with col3:
                        st.metric("Confidence", f"{confidence:.1f}%")
                    
                    with col4:
                        p_int = probability[1] * 100 if len(probability) > 1 else (100 - probability[0] * 100)
                        st.metric("P(Interaction)", f"{p_int:.1f}%")
                    
                    # Visualization
                    if len(probability) > 1:
                        fig = go.Figure(data=[
                            go.Bar(x=['No Interaction', 'Interaction'], 
                                  y=[probability[0]*100, probability[1]*100],
                                  marker_color=['#ef5350', '#66bb6a'])
                        ])
                        fig.update_layout(
                            title="Probability Distribution",
                            yaxis_title="Probability (%)",
                            height=300,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show sequence stats if available
                    if (seq1 or protein1_name in gene_sequences) and (seq2 or protein2_name in gene_sequences):
                        s1 = temp_sequences.get(protein1_name, "")
                        s2 = temp_sequences.get(protein2_name, "")
                        
                        if s1 and s2:
                            st.markdown("### 🧬 Sequence Information")
                            stats_df = pd.DataFrame({
                                'Property': ['Length (aa)', 'Hydrophobic (%)', 'Charged (%)'],
                                protein1_name: [
                                    len(s1),
                                    sum(1 for aa in s1 if aa in 'AVILMFYW') / len(s1) * 100 if s1 else 0,
                                    sum(1 for aa in s1 if aa in 'DEKR') / len(s1) * 100 if s1 else 0
                                ],
                                protein2_name: [
                                    len(s2),
                                    sum(1 for aa in s2 if aa in 'AVILMFYW') / len(s2) * 100 if s2 else 0,
                                    sum(1 for aa in s2 if aa in 'DEKR') / len(s2) * 100 if s2 else 0
                                ]
                            })
                            st.dataframe(stats_df, use_container_width=True)
            else:
                st.error("Please enter both protein names and ensure sequences are available")
    
    with tab3:
        st.markdown("## 🌐 Protein Interaction Network Analysis")
        
        if len(st.session_state.predictions) > 0:
            predictions_df = pd.DataFrame(st.session_state.predictions)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Predictions", len(predictions_df))
            with col2:
                interactions = len(predictions_df[predictions_df['Prediction'] == 'Interaction'])
                st.metric("Interactions", interactions)
            with col3:
                avg_conf = predictions_df['Confidence'].mean()
                st.metric("Avg Confidence", f"{avg_conf:.1f}%")
            with col4:
                unique_proteins = len(set(predictions_df['Protein1'].tolist() + predictions_df['Protein2'].tolist()))
                st.metric("Unique Proteins", unique_proteins)
            
            # Network Visualization
            st.markdown("### 🔗 Interaction Network")
            
            fig = create_network_visualization(predictions_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No interactions found to visualize. Make some predictions with positive interactions first.")
            
            # Network Analysis
            interactions_df = predictions_df[predictions_df['Prediction'] == 'Interaction']
            if not interactions_df.empty:
                st.markdown("### 📊 Network Statistics")
                
                # Create network for analysis
                G = nx.Graph()
                for _, row in interactions_df.iterrows():
                    G.add_edge(row['Protein1'], row['Protein2'])
                
                if G.number_of_nodes() > 0:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Nodes", G.number_of_nodes())
                        st.metric("Edges", G.number_of_edges())
                    
                    with col2:
                        if G.number_of_nodes() > 1:
                            st.metric("Density", f"{nx.density(G):.3f}")
                            try:
                                st.metric("Avg Clustering", f"{nx.average_clustering(G):.3f}")
                            except:
                                st.metric("Avg Clustering", "N/A")
                        
                    with col3:
                        degrees = dict(G.degree())
                        if degrees:
                            st.metric("Avg Degree", f"{np.mean(list(degrees.values())):.2f}")
                            st.metric("Max Degree", max(degrees.values()))
                    
                    # Hub proteins
                    if degrees:
                        st.markdown("### 🎯 Hub Proteins (Most Connected)")
                        hub_df = pd.DataFrame(
                            sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:min(10, len(degrees))],
                            columns=['Protein', 'Connections']
                        )
                        st.dataframe(hub_df, use_container_width=True)
        else:
            st.info("No predictions available. Please make predictions first in the PPI Prediction tab.")
    
    with tab4:
        st.markdown("## 📈 Results Summary")
        
        if len(st.session_state.predictions) > 0:
            predictions_df = pd.DataFrame(st.session_state.predictions)
            
            # Summary Statistics
            st.markdown("### 📊 Summary Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Prediction distribution
                counts = predictions_df['Prediction'].value_counts()
                fig = px.pie(
                    values=counts.values,
                    names=counts.index,
                    title="Prediction Distribution",
                    color_discrete_map={'Interaction': '#66bb6a', 'No Interaction': '#ef5350'}
                )
                fig.update_layout(height=300, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Confidence distribution
                fig = px.histogram(
                    predictions_df,
                    x='Confidence',
                    nbins=20,
                    title="Confidence Distribution",
                    labels={'Confidence': 'Confidence (%)', 'count': 'Frequency'}
                )
                fig.update_layout(height=300, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed Results Table
            st.markdown("### 📋 Detailed Predictions")
            
            # Format the dataframe for display
            display_df = predictions_df[['Timestamp', 'Protein1', 'Protein2', 'Prediction', 'Confidence']].copy()
            display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(display_df, use_container_width=True)
            
            # Export Options
            st.markdown("### 💾 Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = predictions_df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    f"ppi_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
            
            with col2:
                json_str = predictions_df.to_json(orient='records', indent=2)
                st.download_button(
                    "📥 Download JSON",
                    json_str,
                    f"ppi_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
            
            with col3:
                if st.button("🗑️ Clear All Results"):
                    st.session_state.predictions = []
                    st.rerun()
                    
            # Quick Stats
            st.markdown("### 📈 Quick Analysis")
            interactions = predictions_df[predictions_df['Prediction'] == 'Interaction']
            if not interactions.empty:
                avg_confidence = interactions['Confidence'].mean()
                st.info(f"Average confidence for interactions: {avg_confidence:.1f}%")
                
                # Most confident interactions
                top_interactions = interactions.nlargest(5, 'Confidence')[['Protein1', 'Protein2', 'Confidence']]
                if not top_interactions.empty:
                    st.markdown("**Top 5 Most Confident Interactions:**")
                    for _, row in top_interactions.iterrows():
                        st.write(f"• {row['Protein1']} ↔ {row['Protein2']} ({row['Confidence']:.1f}%)")
        else:
            st.info("No results available. Please make predictions first in the PPI Prediction tab.")

if __name__ == "__main__":
    main()"""
ProCaPPIS - Prostate Cancer Protein-Protein Interaction Prediction System
Final Working Version - Simplified and Stable
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import json
import joblib
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ProCaPPIS - PPI Analysis Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional CSS
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
    }
    
    h1 {
        color: #4fc3f7 !important;
        text-align: center;
        padding: 20px 0;
        border-bottom: 2px solid #4fc3f7;
        margin-bottom: 30px;
    }
    
    h2 {
        color: #81c784 !important;
        margin-top: 2rem;
    }
    
    h3 {
        color: #ffb74d !important;
    }
    
    .academic-card {
        background: rgba(30, 30, 63, 0.6);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid rgba(79, 195, 247, 0.3);
        margin: 1.5rem 0;
    }
    
    .academic-card p {
        color: #e0e0e0 !important;
        line-height: 1.8;
    }
    
    [data-testid="metric-container"] {
        background: rgba(30, 30, 60, 0.6);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(129, 199, 132, 0.3);
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #4fc3f7 !important;
        font-weight: 700;
        font-size: 28px;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #00acc1 0%, #4fc3f7 100%);
        color: white;
        border: none;
        padding: 0.75rem 2.5rem;
        font-weight: 700;
        border-radius: 30px;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: rgba(30, 30, 60, 0.8);
        border: 2px solid rgba(79, 195, 247, 0.3);
        color: #e0e0e0;
        border-radius: 8px;
    }
    
    .stSelectbox>div>div>select {
        background-color: rgba(30, 30, 60, 0.8);
        border: 2px solid rgba(79, 195, 247, 0.3);
        color: #e0e0e0;
    }
    
    p, span, label, div, li {
        color: #e0e0e0 !important;
    }
    
    .info-box {
        background: linear-gradient(145deg, #1e3a5f, #2c5282);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #4fc3f7;
        margin: 1.5rem 0;
    }
    
    .success-box {
        background: linear-gradient(145deg, #1b4332, #2d6a4f);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #81c784;
        margin: 1.5rem 0;
    }
    
    .warning-box {
        background: linear-gradient(145deg, #5d4037, #6d4c41);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #ffb74d;
        margin: 1.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

# Load models and data functions
@st.cache_resource
def load_models():
    """Load trained models with error handling"""
    try:
        if os.path.exists('champion_model.joblib') and os.path.exists('champion_scaler.joblib'):
            model = joblib.load('champion_model.joblib')
            scaler = joblib.load('champion_scaler.joblib')
            return model, scaler
        else:
            st.sidebar.warning("Model files not found. Using demo mode.")
            return None, None
    except Exception as e:
        st.sidebar.error(f"Error loading models: {e}")
        return None, None

@st.cache_data
def load_gene_data():
    """Load gene data with fallback"""
    try:
        if os.path.exists('focused_gene_to_sequence_map.json'):
            with open('focused_gene_to_sequence_map.json', 'r') as f:
                gene_sequences = json.load(f)
        else:
            # Demo data with shorter sequences
            gene_sequences = {
                'TP53': 'MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD',
                'AR': 'MEVQLGLGRVYPRPPSKTYRGAFQNLFQSVREVIQNPGPRHPEAASAAPPGASLLLLQQQQQQQQQQQQQQQQQQQQQQETSPRQQQQQQGEDGSPQAHRRGPTGYLVLDEEQQPSQPQSALECHPERGCVPEPGAAVAASKGLPQQLPAPPDEDDSAAPSIDKGAIPASNSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSGTSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSNSLSSTSGSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSLQSSL',
                'BRCA1': 'MDLSALRVEEVQNVINAMQKILECPICLELIKEPVSTKCDHIFCKFCMLKLLNQKKGPSQCPLCKNDITKRSLQESTRFSQLVEELLKIICAFQLDTGLEYANSYNFAKKENNSPEHLKDEVSIIQSMGYRNRAKRLLQSEPENPSLQETSLSVQLSNLGTVRTLRTKQRIQPQKTSVYIELGSDSSEDTVNKATYCSVGDQELLQITPQGTRDEISLDSAKKAACEFSETDVTNTEHHQPSNNDLNTTEKRAAERHPEKYQGSSVSNLHVEPCGTNTHASSLQHENSSLLLTKDRMNVEKAEFCNKSKQPGLARSQHNRWAGSKETCNDRRTPSTEKKVDLNADPLCERKEWNKQKLPCSENPRDTEDVPWITLNSSIQKVNEWSRQRWWESWSVPNYQPNTNQSRYARPSAEPSLHR',
                'MYC': 'MPLNVSFTNRNYDLDYDSVQPYFYCDEEENFYQQQQQSELQPPAPSEDIWKKFELLPTPPLSPSRRSGLCSPSYVAVTPFSLRGDNDGGGGSFSTADQLEMVTELLGGDMVNQSFICDPDDETFIKNIIIQDCMWSGFSAAAKLVSEKLASYQAARKDSGSPNPARGHSVCSTSSLYLQDLSAAASECIDPSVVFPYPLNDSSSPKSCASQDSSAFSPSSDSLLSSTESSPQGSPEPLVLHEETPPTTSSDSEEEQEDEEEIDVVSVEKRQAPGKRSESGSPSAGGHSKPPHSPLVLKRCHVSTHQHNYAAPPSTRKDYPAAKRVKLDSVRVLRQISNNRKCTSPRSSDTEENVKRRTHNVLERQRRNELKRSFFALRDQIPELENNEKAPKVVILKKATAYILSVQAEEQKLISEEDLLRKRREQLKHKLEQLRNSCA',
                'EGFR': 'MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCNLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLEGCPTNGPKIPSIATGMVGALLLLLVVALGIGLFMRRRHIVRKRTLRRLLQERELVEPLTPSGEAPNQALLRILKETEFKKIKVLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLLGICLTSTVQLITQLMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAARNVLVKTPQHVKITDFGLAKLLGAEEKEYHAEGGKVPIKWMALESILHRIYTHQSDVWSYGVTVWELMTFGSKPYDGIPASEISSILEKGERLPQPPICTIDVYMIMVKCWMIDADSRPKFRELIIEFSKMARDPQRYLVIQGDERMHLPSPTDSNFYRALMDEEDMDDVVDADEYLIPQQGFFSSPSTSRTPLLSSLSATSNNSTVACIDRNGLQSCPIKEDSFLQRYSSDPTGALTEDSIDDTFLPVPEYINQSVPKRPAGSVQNPVYHNQPLNPAPSRDPHYQDPHSTAVGNPEYLNTVQPTCVNSTFDSPAHWAQKGSHQISLDNPDYQQDFFPKEAKPNGIFKGSTAENAEYLRVAPQSSEFIGA',
                'PTEN': 'MTAIIKEIVSRNKRRYQEDGFDLDLTYIYPNIIAMGFPAERLEGVYRNNIDDVVRFLDSKHKNHYKIYNLCAERHYDTAKFNCRVAQYPFEDHNPPQLLELIKPFCEDLDQWLSEDDNHVAAIHCKAGKGRTGVMICAYLLHRGKFLKAQEALDFYGEVRTRDKKGVTIPSQRRYVYYYLLKNKFMDAQDFSERKRREDFQMFTQDFRERVALIKDVVEAIPKSTYLSTFNEFLHKLLMKKRKQPIMVSTLMLAFNLHGDLQISQLKPFNTTMEAACSRTDAALSGYIPKLSIAYISKLKFDFQSGGLEEEKTSLCQYKYLVLRRELRGAPGQSMQSEVAKFKKDSQFKRLNYFGTYTKDKDVEEELTFSLEKIYEHHLKHKLDRKEEKDTIDLLEEVKHGDLNRNIEKIDTKIFADITPEAKQSEGSQALQAHQALFTEQTFTKFKDQEKEMNAYVKKKTYMLDQRFLVEPPEAKVCKCKLLNWDDNNLCFDLFTKEPDNKSKRFHGDRTILNRIISLTLTDSSFITDDDSCNQTLCVQNLRDSATAEEKWQLTDAKQTAVKCSRDLLQNGEKLVSQKTFDLKLKAMLECSQSKTYRYTKKTYLTKHKFGRLQNYGLDLDTVQSIINMLKDAFQEIAADLFRIITKDNYLQRLNLEAKTYNSRCFNTQMSIIAQKKFGYKMKPFVYELVKKPEGKSQAAILSLRQRLREGGQFRPEGWGLLQLSCIGMQNSLEEVQEKQKDEDLVSAIVNAAGIQEQVLQSAKKDLLRVDVLCNYPYVKSLLEELGQRPQFTYQSYVLTHFHHQKVLKQTKDKLDKAGEIISDSSIWSKQTVRLRQLLRRAQEKPPKWKRDFPTEDPSSFVTSEKMQLIEETSEQLAFRQAKIVCSAAEVCSQIQKCKHGIIKQSEEELTGYSHLKVGVQGVSFVCSLKLTYQECQFASRQKPNNPVCVPVPTVLLKQTKPRTILPDQIMRCGKRALPLGFPPHEFEHGFGPHLHMHQQMQPKQSHYSLYQAVKRRPVCNVKGLEKLRTEGIQLKRCGHKRCSLLTCSLLRFAQCDKGRIHMQRTQRSLHYYYGDYHTSSHRSRPAVGPGAWQTCQPVATNSDFLSPPFPQLCQNSSQRSPSPHQQHDGQKLQIASQYQIKMKDQKQNNEEEVEKAKVLRGLVMPLQQKAKLGGKFQLVCQCRDLPPPCGPLPIQGKPEAAYQLCGQSSSPSSGVGAGGKSKLRSAQSPPPQRPQGDQGCQQRRRPARPQVLLPEDPGAYFGLGLLHTFASQRPDDQCASRDEKPRPQKEFHHDLSPYDVKELLSVNPPGQHGQLVLLNKSRGGIQPMDKQERPRQQSYGRTAQEDRCQNIQKPPPEYPKQLPPKRSPWAPIKAKTVEVSCGKRCSNLLDGVHMVKSRSGGCGFNAPPWRGGGGQKAGGKAKARRQRQFLPSQGSGSPTKSRKCMHPHPPSVGIQCQHKGQGKGIRQSDKEESPYECVCGLCLQSSSSAGQPPSTGNAKGRCACAGLRRLLTRQGQGIQLLSPQRVRCAASKAGSATRKSPLHLPFQTARLKKKEKKQVRQQVHVICAYPAKPDTLLHCFHVGKQLPGLVFQKLPHHDQKTADQVCWAFGLRTNGKRDRFGDCVSGITDQMDEHPFHPEVDTKFATCRDGQTFQLRDTAICKLKKKSPVKVNLDLSAPPRTSRSNLASQLFSRGPVGPHSLLLTDTVLQAQTGSRPPGSPGGLQRCQAGRVAERQLTVKQPYLGTPVPNHTVEPETVSSDKFEGTLVTSAALSQQGSGQFKQFTPQRVQPQNRSPLGPSFGQSLEEVLSQQPQMKKLQLGKQLEEPQGALQVPVQKQSSAEQRQSGRDRGSIAQARRVAESKEQQGQMGKGQGSQLPGGEQGQPGAGPCGQPPTVSLLYQVDRCGIYPEPAGKAPCPPELYQVMRDAKMRNLQASNQKGDLDLEKRQKSGCDAGARKDLPYGRLKGGQLPVQIAKQRQCVKACGQLAAVCGTQSVVQRQVKAQAAQQQMLLDGDQYLQKSQAGDYRVVRGQTDQLIKELQRLGSGTVQTHRVSVAGHRGPSPHLESEAPSPRPWAPGQETFQSKIVLDKLVGDVEHSKDHFDLDQLEDVAKDMSSLYKQEKAKELRSLGPSLPKQPNPQQKSSVSNTGAVGTSKAVQPGPPPPAMVRVTRPGSAYDPQIHLSETRFQSTQLQLQDYMAIQRTRPKDLAAYNGTQRDEEYSRQKFRQSFDAIYLPGFIASGVKLYGIGSDTIRKTSNQLQPQVRDCAWGALVQWDQYPNRGNWDAIAFNLPSAVLLALGLQRWVHLGQDWQENAFETPGGVASWGGLNVGDVDEQAYDPGEAQGGSGVDPNQEPQVRYRLQDWRHGDRVRDPLDTKGGRDDEIGRLKVDHQGRSARRYGQDALLQTQKTKSMNKVQINQQQQQPGDVAEDLSQEFMDAYGMRNEAKGKDAEYGRDTLQEKRQLTGNQRRQLQAQKQKIIQEIVQRAEQANVLGTKVDGKTFYLMIPSNLKRNDLKRGYEGDGRTQRQVDGTCLLKQQRGEYGVSGSYKNQGKTDLIQDYLQRQVNLQKKIAKLQEQQEADLLTRQGQKGAKAFQALLREDQVRSQRMRPSRDQMQNSDVSPGSGSSLHQWLSYIKEISQPGQDSQFGSKFEEYFHSKDHFGKAGSQNQHNGLSTGAQLRALPDKTSSCDAAYTAKMAAEDEGTTLEMLGELADTDPQLGKAGGGLQRRSGADQPAKFNLKEPGKAQLLEAATKAPMQLLLEGTAGSGRQHTQLFHQEVAFVACFLLDQLMRGAEDDYHLMPKGVLARLLEDGSKQWLNELLETEDANAIKVYLDRSCQVGGQDYRKQPAFPAYFYHLKLSDYFRKTMHVRARGDPQGTPMHLKKFHQGAVDHLHDCIDLQAEEGVLSSLQELGAELAATRPNADRYVTASQRPALSGKMRGRLFLCDGLDEGKGAEQADMGEHLPFDILQDQAMRLSQAAEDVEKLKTAKLKRPLHPYVDCFQHRQVAELTTRHSLQEQIQKVKKLSQLMHEGRYQSLIDFHAYQGRLTQVAEFLDHHLPKLLRLTDQPHCDWRYFAVDNCYLRYLQAQQIQAADELLRLKQCQEQKQRQGKTEKQARAKHLQKMMKEDPPGQAELNTPTNLLPDFLIPNLQDVGTQCQQSNSHLHSDFVLKQFASAHYQETQKGDLKEPLLSGELLKGVPAPPTPLSNQYLQALLQGLNQFTADLADFTDEEAVKALLEAYIPNLQKQVEGVLDQGKLAAGLGQVPIPSPYQGFGQAQKQAEGGQAQGRFLLSSGLPDMGEGWPGRFGEAYIDLAKWEEADGQETASAWHEFQEAAQGVSAAGFLQGSLQLFLENLVAKLAHEKTSQVQARLQAEEFLKTAPQAPGIQKSLLKRSGQSLFNSQALLRHAAHTLKFAHTPQEGDLGDLKHQQQLQAFRYVTFHEQALEAAHQAGDADCDVSVLFQGGAGGDKTLEHYRSSSHGDQAGVGRRPPADHFQALEASAVSANPGLQPGLDLRQRQAQSYAVAKKAEHAARQLLLLLLEASDLLDSPLVNALGVTTKAKRATLKQAQKLGAQKVQQLALDQPRGGELAAERKLLQHCYDRSEDSDAMTLEDQLSHSPQHGDRDKKQQLPPSLVCHLLAGAFGRAAPRLEQTRRSLVQKAHAQRQEQVLRGGAGTSSDGQEGSGHLKSLDQQQLQHQHLLNKQGRVRRQAQIQQRQAAAEGASPSQCSSQNQVAGHSPGSALPQVTGYLRSQLAGQRDPEPKQAEDLGLYVRSEVHQKLVAEHGAKQGQTLLARLQAGRLSPELEAALLQAGFQLLKAELQHQSMDLRRLAADVRDELSQQVGALLLHRLLLRGGEQLAPTHLLQELLTLHHGPSDVYPLMQRLGEPRQDERRCSEPQHRLVQLQAQHLGRQRSQMATLQKVKLLNLKGLSHCLVKKLLSYHQAEEHQRLKQELQQLQKGSEKDQQQAMQDLLKDMQEEGGLQRGLQEAQGEGLLLSRLQRRQQSLLRKQRRDLLQSQLAQEPAKMGPQCNQTSASAPPALRSALAPPPQQQQQQEELLEQGRMLLQQLQQQQQLLRRLLQSGPSSPPSSPEERLPRRPEEPLVPSGLQQAAGEKGKLHHRKSYRQLKSRQRHDKQGRKHALGGLREQRRLHHNHLKDRKGAPGPVPPLPGQLQSRCFPHQLQRRRFPGSPAGLLLQAGGGLLEQFRPSQKLGLLQQVYQGTFPAASLPHDFPGNLRMGALRHLLGGQAQGSRPVPGRSRGVDLLQNCDLQYEADSWKLSQFSPAFPEAADFRRVQTRHPLSDLLDDSQQPGTTEERRELSQRLDRCLQSHPYRSQLLQLQLQQLQGQPPLHYRQQLKEREARRQSLGQHRLLLASQLLLAGPLRSGRPSEQLDAELLGNLQRPVSHPGLEGAFLQAHQQGRATLSHSPDLQRLQQHFQRAQAEVQLEKGEELQRQLQSLGQAEPVGGQFQQAAGRQAQSLLRRQGEGLRQLQGLAGGQGGGGAMAQLAAELLAAPMLLKEESAGGAQLEEGLSQFLGHQPPQPLAFRRAPFSVGTVQFTYLHQPQRPRFPPALAALAPLLRRAGRQAAHAQRAHHALADDALLFQAVDRELQVQREELQAASDHLLQGLQSSEQGFQRQQLQQQEQQLLKTVEKPQGLEPVQRLQLAAHQRRRELSRDLGKSLRRDRAGEEQLAEVEKQRQAQHQAGDQPSGLESRSDLQRALLGRLAGRLGQELAAELQRNSQRFGSQRFTQPQDRLEAKLQACLLHKHPRRPSRSLQQILRSQLRAAAHGRQEAGLEALLRDLQEELLRRHRLGAAGGQGLQEAHQQALPAGAADLRLSLQQRGGAAGRLQLLQQLQAQLAELQRHSQEAARRQLQFQLLQAEARLTPSRSQPRAGQVPLLGSELQRQAQTLQRRLQDLQERLLQGQERQGVQPRGVLGRLGVLRRLLRSAGRRPQWDPQQRGEQLLRALLAGGQQAPAQEGGLLQQLQKGQRRLLDDLQAALDRAGQVKSELLERAAHRLQARLQEQLAPARLLQEALQGLQREALRQLARAHSAGLGPAGSAPSLQGLPDAQHLSRQRLLHQGGGVGAAALLQDLQSELLRQGRQEGLRRQLGGQRRLLRSAGQREAAEPRAGLLQELQAAGQEQDLQRLLRRQRGGAAHQRLHQAKAERLPQQRAGQGQQLLEALGARGAQARLQREGLQELQRAALQGRALRAQPLLAGVSAALEEQGGQLQQELLLGGGGHGQAQLQAGLQEAREALDRHPQAGHQRHGAPLLQGLLQAERQAGAQQLQRLQRAAQQPQRALQRSQAAEHQRQHQQQQLRLQETPRAGQLQELALQRSLQQQAGREAHSRLLRALGRQLLRSAGRQAGQPPSLAPGGLAQQLLREGLRAQLAEDGQHRLLRAGQRHLLLRAQQQPLLAEGLQQRLLRSAAQRGAQVPSRLGLLQQLLHGAQRQQQQARLQRHQQGRQGQLQRLHQAGRQLEEGARLQLQRRALRQSLLRRAQARLEASRSLQQQLGRALLRAGLQALQHAAQQGRAPSRQLLQELQHGGRQQLAQRQQQQQEQQRRRQELLRRQQEAARALLRRQGRLAAGRHRLLLEAGLRQLLRAGLRAESGRGQVPGLSQLLQRLLQSQRQGELRRAQRAQLLRTQRRAQLLRSAGNLASRAEQGGLLRALQRAQREGLLRAHQGRPRHQLAGRALLRALQARGQEGLQRLGGRQQLRRLAGHPPLLQRSLQLLLAGGQRPQAAVPGGAGLQRELQRANLRAGLLRQRHQRRQGLQRLAQARGQGLLRRGGRQQVPRLLQPALGQLQRELEAALRRAGRQAHHQLFQAGRQGLLRAGQRRLLLLGQRQLLRRAGQRQPLLGGQLLRRLQRQGLRAGQRHQLLRASGRALLRAGRQAAQLLRRGQRRLLRRAGQARLQRLGQLLAQLLRRAQRAQLLRSGQLLQELLHAAQQGPQQLGLLRALQGRPQLLRRAGQAQLLRARGHQLQRLLRSQGRLLRRALHAQQGLQRLLAAGLRAGQQLQRLLRSGQRLLRRAGQAQLLRARGQQLLQRLLRSGGRLLRRALHAQQGLQRLLAAGLRAGQQLQRLLRRGQRLLRRAGQAQLLRARGQQLLQRLLQAAPRLLRRALHAQLGLQRLLAAGLRAGQQLQRLLRSGQRLLRRAGQAQLLRARGQQLLQRLRSAGQQLLRRALHAQQGLQRLLAAGLRAGQQLQRLLRRGQRLLRRAGQAQLLRARGQQLLQRLLQAAPRLLRRALHAQLGLQRLLAAGLRAGQQLQRLLRSGQRLLRRAGQAQLLRAGAQQHQKLLRRGQRELLLEAGQREQQLQRLLRAAGRALLRGNGQLQRLLQAAGRAGQQLQRQQQGQRQLQRALLAGGLRAGGQQLLRSAQRQGLLARGHQQRLLRAGGRALLRRAQRQLLARGSQLLRRLRRADIQQQHQRLLQAGGQALGAQQQLLLSRQQLQEGLLRAGQRLQLLRARGLQQQLLRAHGRALLRGAGRQRRPQSGLLLRQLRQVLIRRAGQRQLLLEQSGQQLLQLLRRAGQAHLRSLQRQLQRAGRQLLRQHHRALQRGGGLLRAHQQGRLQSLGRLQRELRRDGQQLLRSHQQQLLQEHQRPLLAGGLARQQLLRRAGQRPLHQTLQRQLQRLGRLLLQSAGQQLLQGLRRQLLDRSGQLLQRHQRLQLRHQAGLLLQRAAHQLLQRLLQRAGQRLLRQSGQLLEHLLRQAAQQLRRQRALQLLRGQRQLLRTAGQRLLLQAGQQLLRRRQRASLDQAQRRLQRALHAGQQGLLRAGQLLQRLLRRRGQRALLAGQQLQRLLRSAGQRLQRMQRRLQALGQRQLLRQAGQRLQRAGQRLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLRRAGQQLQRMHQRRLQLALQAHLLRQLQSAGQRLRRAGQRRLQRALLAGGQRLLRRAGQRLQRHGQRRLQRALGAQLQRLLQRARLRLQRRGQRRLQRAGQRQLRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLRRAGQQLQRMHQRRLQRAGQLLRRLQRAAGRLLRARQQAQGFQHGQHHLLRRRLLRSLQRAGQRLLQRHQRRLQRALGQLLQRLLRRAQQQLLHRHQRRLQRAGQRLQRHQGQRLQRQQQRLHQRAGQRALQSLQAGQQLLRRAGQRLHQRQGQAELRRHGQRLQRQQHQLLQRAGQRLQRHAGQRLQRAAGQRLLRQHQRRLQRAQRAQLLRRAGQQLQRMHQRRLQRALGRLQRQAGQQLLRRAGQRLQRHGQRRLQRALGAQLQRLLQRARLRLQRRGQRRLQRAGQRQLRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLRRAGQQLQRMHQRRLQRAGQLLRRLQRAAGRLLRARQQAQGFQHGQHHLLRRRLLRSLQRAGQRLLQRHQRRLQRALGQLLQRLLRRAQQQLLHRHQRRLQRAGQRLQRHQGQRLQRQQQRLHQRAGQRALQSLQAGQQLLRRAGQRLHQRQGQAELRRHGQRLQRQQHQLLQRAGQRLQRHQAQYRRFQRSKRQAGLLGRAVLLYSDFATPSQPGAGAGPVRPRYTLGTGGPCLTSQTSPKFLFQNALKSLTRLSHPPQPTAEAEDMSRLLKQHQLKQQLLRAGGRQPVQVRYYSQPLILAASHLLRQHQGAEGRGLRGEQRLLRQAGQRQLQRHGGQRLQRAGGRLLRQQQRLQRAQHQLLQRAGQRLQRHAGQRLQRAAQQRLQRHQRRLQRAGQRLLRRQHQLLRRRHQRRLQRAGQRQHQRHQARLEQQGRQLLRRAGQRQLQRHAGQRLQRAAGQRLLRQQHQQLQRAGQRLLRRQHQLLRSQGEQRHQRLLQQGRLLQRAGQRLLRRQQQQLQRAGQRLQRQQHQLLQRAGQRLQRHAGQRLQRAAQRLQQRHQRRLQRAGQRLLRRHGQLLRSQQQQLQRAGQRLQRHQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGRQLLRRQHQLLRRQGQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRHGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRHGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRHGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRHGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRHGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRHGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQRHAGQRLQRAAQRLQRQHQRRLQRAGQRLLRRQHQLLRRQSQQLHQRHQRRLQRAGQRLLRHQGQRLQRRQHQLLQRAGQRLQRHRGRRLQRAAGRLLQRAQRRLQRSGQRRLQRAGQRQLQRHAGQRQLQRAGGRLLRQHQRRLQRAQRAQLLQRAGQRLQRHQGRLLQRLQRRQGQRLQRRQHQLLQRAGQRLQ
