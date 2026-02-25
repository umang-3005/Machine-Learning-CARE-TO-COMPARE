# Predictive Maintenance for Wind Turbines

## Comparative Analysis of Supervised and Clustering Techniques

------------------------------------------------------------------------

## Project Overview

This project implements and compares two machine learning approaches for
early fault detection in wind turbines using SCADA data from Wind Farm C
of the CARE to Compare benchmark dataset.

### Objectives

-   Predict all failure events
-   Detect failures as early as possible
-   Minimize false alarm rate

Two approaches are implemented:

1.  Supervised Learning -- Random Forest classifier
2.  Unsupervised Learning -- PCA + K-Means clustering

------------------------------------------------------------------------

## Dataset

-   Source: CARE to Compare benchmark dataset (Wind Farm C)
-   Turbine analyzed: Turbine 44
-   Sampling rate: 10-minute SCADA data
-   Features: 957 anonymized sensor signals per timestamp
-   Events: 24 anomaly periods, 19 normal periods

### Data Challenges

-   High dimensionality
-   Strong class imbalance
-   Missing values encoded as zero
-   Multiple operational states via status_type_id

------------------------------------------------------------------------

# WP1 -- Data Analysis & Feature Selection

## Preprocessing Steps

1.  Missing value handling
    -   Zero blocks treated as missing
    -   Forward-fill for short gaps
    -   Median imputation for longer gaps
2.  Outlier filtering
    -   Removed physically impossible values
    -   Retained borderline extreme values
3.  Feature scaling
    -   Z-score normalization for PCA and clustering

## Feature Reduction Pipeline

Raw features: 957\
After variance & correlation filtering: \~307\
After domain knowledge refinement: \~165\
Final selected features (Random Forest importance): 30

------------------------------------------------------------------------

# WP2 -- Model Development & Evaluation

## Supervised Approach -- Random Forest

### Label Definition

Normal (0): status 0, 2\
Anomaly (1): early fault phases + abnormal states\
Status 3 & 4 excluded (post-failure states)

### Configuration

-   100 trees
-   Gini impurity
-   Class balancing

### Results

-   Recall (Anomaly): 1.00
-   Precision (Anomaly): 0.05
-   ROC-AUC: 0.9986
-   Early detection: 12--14 hours before failures

Strength: No missed failures\
Weakness: High false positives

------------------------------------------------------------------------

## Unsupervised Approach -- PCA + K-Means

### Dimensionality Reduction

-   PCA retaining \~90% variance
-   Reduced from 30 → 5 components

### Clustering

-   K-Means
-   Optimal k = 2
-   Silhouette score: 0.6898

### Results

-   Recall (Anomaly): 0.56
-   Precision (Anomaly): 0.025
-   Higher false positives
-   Does not require labeled data

Strength: Label-independent and adaptable\
Weakness: Lower precision and early detection performance

------------------------------------------------------------------------

# CARE Score Evaluation

True Positive Rate: 1.00\
False Alarm Rate: 0.91\
Average Normalized Lead Time: 0.56\
Overall CARE Score: 0.225

------------------------------------------------------------------------

# Comparison Summary

  --------------------------------------------------------------------------
  Aspect             Supervised                Unsupervised
  ------------------ ------------------------- -----------------------------
  Requires labels    Yes                       No

  Recall             Very High                 Moderate

  Precision          Low                       Very Low

  Early Detection    Strong                    Limited

  Generalizability   Limited to known faults   Better for unknown faults
  --------------------------------------------------------------------------

------------------------------------------------------------------------

# Installation

pip install numpy pandas scikit-learn matplotlib seaborn plotly

Python version: 3.10+

------------------------------------------------------------------------

# How to Run

Run Jupyter notebook:

jupyter notebook experiment.ipynb

Or execute scripts:

python src/supervised_model.py\
python src/unsupervised_model.py

------------------------------------------------------------------------

# Author

Umang Dholakiya\
M.Sc. Artificial Intelligence\
Ostbayerische Technische Hochschule Amberg-Weiden

Supervisor: Prof. Dr. Patrick Levi

------------------------------------------------------------------------

Full technical details are available in ML_REPORT_Umang.pdf.
