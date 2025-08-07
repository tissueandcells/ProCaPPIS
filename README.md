ProCaPPIS - Prostate Cancer Protein-Protein Interaction Prediction System

ProCaPPIS is a state-of-the-art machine learning platform designed specifically for predicting protein-protein interactions (PPIs) in prostate cancer context. By integrating multi-omic data from TCGA-PRAD cohort, STRING database, and leveraging advanced protein language models (ESM-2), our system achieves unprecedented accuracy in identifying novel therapeutic targets and understanding disease mechanisms.
Key Innovations

Cancer-specific approach: First PPI prediction system tailored specifically for prostate cancer
Multi-omic integration: Combines transcriptomics, proteomics, and functional genomics
Deep learning features: Utilizes ESM-2 protein language model embeddings
Clinical relevance: Identifies druggable targets and resistance mechanisms

✨ Features
Core Functionality

🔮 Real-time PPI Prediction: Instant prediction with confidence scores
🌐 3D Network Visualization: Interactive protein interaction networks using Plotly
🔍 Gene Explorer: Comprehensive analysis of 1,545 differentially expressed genes
📊 Statistical Dashboard: Performance metrics, prediction history, and analytics
🎯 Batch Processing: Process multiple protein pairs simultaneously
💊 Drug Target Analysis: Identify potential therapeutic intervention points

🚀 Quick Start
Online Demo
🌐 Live Application: https://procappis.streamlit.app
Local Installation
Prerequisites

Python 3.10 or higher
8GB RAM minimum (32GB recommended for full pipeline)
10GB free disk space

Step-by-step Installation

Clone the repository

bashgit clone https://github.com/tissueandcells/ProCaPPIS.git
cd ProCaPPIS

Create and activate virtual environment

bash# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

Install dependencies

bashpip install --upgrade pip
pip install -r requirements.txt
