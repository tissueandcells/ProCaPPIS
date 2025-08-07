# ProCaPPIS

ProCaPPIS is a machine learning-based web application for predicting protein-protein interactions (PPIs) specific to prostate cancer. The system integrates multi-omic data from TCGA, STRING database, and uses state-of-the-art protein language models to achieve high prediction accuracy.
Features

PPI Prediction: Real-time prediction of protein interactions
Network Visualization: Interactive 3D protein interaction networks
Gene Explorer: Search and analyze differentially expressed genes
Statistical Dashboard: Comprehensive performance metrics and analytics
Batch Processing: Process multiple protein pairs simultaneously


Quick Start
Online Demo
Visit our live demo at: ProCaPPIS Web App
Local Installation

Clone the repository

bashgit clone https://github.com/yourusername/ProCaPPIS.git
cd ProCaPPIS

Create a virtual environment

bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies

bashpip install -r requirements.txt

Download required data files

Due to file size limitations, some data files need to be downloaded separately:

focused_esm_embeddings.h5 (>100MB) - [Download Link]
champion_model.joblib - [Download Link]
champion_scaler.joblib - [Download Link]

Place these files in the project root directory.

Run the application

bashstreamlit run app.py
The application will open in your default browser at http://localhost:8501
📊 Dataset Information
Data Sources

TCGA-PRAD: 494 tumor and 52 normal tissue samples
STRING v12.0: Protein interaction database
UniProt: Protein sequences and annotations

VIP Gene Selection

Total VIP genes: 1,545
Upregulated: 743 genes
Downregulated: 802 genes
Selection criteria: |Log2FC| ≥ 1.0, FDR < 0.05

🤖 Model Performance
MetricValueAccuracy79.85%Precision79.9%Recall79.9%F1-Score0.798ROC-AUC0.872
📁 Project Structure
ProCaPPIS/
├── app.py                              # Main Streamlit application
├── requirements.txt                    # Python dependencies
├── README.md                          # Project documentation
├── LICENSE                            # MIT License
├── .gitignore                         # Git ignore file
├── data/
│   ├── vip_gen_listesi.csv          # VIP gene list
│   ├── focused_ppi_pairs.csv        # PPI training data
│   └── focused_gene_to_sequence_map.json  # Gene sequences
├── models/
│   ├── champion_model.joblib        # Trained SVM model
│   └── champion_scaler.joblib       # Feature scaler
├── notebooks/
│   ├── 01_differential_expression.ipynb
│   ├── 02_network_analysis.ipynb
│   └── 03_model_training.ipynb
└── docs/
    ├── methodology.md
    └── api_documentation.md
🔬 Methodology
Feature Engineering

ESM-2 Embeddings: 2,560 dimensions from protein language model
Sequence Features: Amino acid and dipeptide composition (840 dims)
Physicochemical Properties: 8 dimensional features
GO Similarity: Functional similarity scores

Machine Learning Pipeline

Differential expression analysis (TCGA-PRAD)
VIP gene selection
PPI network construction (STRING)
Feature extraction
Model training (SVM with RBF kernel)
5-fold cross-validation

📝 Citation
If you use ProCaPPIS in your research, please cite:
bibtex@article{procappis2024,
  title={Prostate Cancer-Specific Protein-Protein Interaction Prediction Using Multi-Omic Data Integration},
  author={Your Name et al.},
  journal={Journal Name},
  year={2024},
  doi={10.1234/example}
}
🤝 Contributing
We welcome contributions! Please see CONTRIBUTING.md for details.

Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request





🙏 Acknowledgments

TCGA Research Network for providing genomic data
STRING database team for protein interaction resources
Meta AI for ESM-2 protein language model
Streamlit team for the web framework

📊 Performance Benchmarks
ModelAccuracyF1-ScoreTraining TimeSVM79.85%0.7983.53 hoursRandom Forest79.03%0.79039.28 secXGBoost78.31%0.7835.14 minLogistic Regression74.73%0.7474.67 min


Initial release
Core PPI prediction functionality
Network visualization
Gene explorer module

Planned Features

 Batch prediction API
 Drug target analysis
 Pathway enrichment visualization
 Export to Cytoscape format
 Multi-cancer type support


Note: This project is for research purposes only. Not intended for clinical use.
