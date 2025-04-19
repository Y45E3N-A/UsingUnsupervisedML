# Unsupervised PRPD Pattern Clustering using CNN + Wavelet Scattering

This repository contains the code, data, and clustering results for my final year engineering project:  
**"Using Unsupervised Machine Learning and Wavelet Scattering Transform to Analyse PRPD Patterns for High Voltage Applications."**

---

## 📌 Project Overview

This project uses an **unsupervised AI model** to cluster Phase Resolved Partial Discharge (PRPD) graphs.  
It aims to identify structural patterns that are often overlooked during manual analysis — by leveraging:

- **Wavelet Scattering Transform (WST)** for robust multi-scale feature extraction.
- A **Custom Convolutional Neural Network (CNN)** for high-level feature encoding.
- **UMAP** for dimensionality reduction.
- **K-Means Clustering** to group similar PRPD patterns without needing labels.

The model clusters **388 PRPD images** into 5 distinct groups, each correlating with variations in temperature, insulation type, voltage, and electrode configuration.

---

## 🧠 Key Results

| Metric                   | Value     |
|--------------------------|-----------|
| Silhouette Score         | 0.7401    |
| Calinski-Harabasz Index  | 2149.32   |
| Davies-Bouldin Index     | 0.3572    |
| Optimal Clusters (k)     | 5         |
| Total PRPD Images        | 388       |

---

## 🗂 Repository Structure

```
├── Clustering Results/
│   ├── Cluster_1/                 # Cluster 1 PRPD images
│   ├── Cluster_2/                 # Cluster 2 PRPD images
│   ├── Cluster_3/                 # Cluster 3 PRPD images
│   ├── Cluster_4/                 # Cluster 4 PRPD images
│   ├── Cluster_5/                 # Cluster 5 PRPD images
│   ├── best_cluster_plot.png      # UMAP visualisation of 5-cluster solution
│   └── cluster_assignments.txt    # Mapping of image filenames to cluster labels
│
├── Code/
│   ├── Image generator from CSV.m # MATLAB script to convert CSV → PRPD image
│   ├── Final unsupervised code.py # Apply WST to PRPD images
│
├── Raw CSV PD data/              # Original CSVs: phase angle & charge magnitudes
│
├── Raw Images/                   # 512x512 preprocessed PRPD images (PNG)
│
├── final_report.pdf              # Full technical report (9,100+ words)
└── README.md                     # You're here!
```

---

## ⚙️ Requirements

- Python 3.8+
- MATLAB R2022a+
- TensorFlow
- scikit-learn
- Kymatio
- umap-learn
- matplotlib, numpy, pandas



## 🙏 Acknowledgements

- **Dr. Qiang Liu** — Project Supervisor  
- **Adam Nor** — For providing partial discharge experimental datasets

---

## 📬 Contact

For questions, suggestions, or collaboration, feel free to reach out:  
📧 [yaseenua1@gmail.com](mailto:yaseenua1@gmail.com)



