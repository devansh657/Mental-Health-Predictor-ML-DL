Mental Health Treatment Prediction: A Multi-Model Analysis
This project investigates the factors influencing mental health treatment-seeking behavior. It utilizes a combination of Unsupervised Learning for pattern discovery and Supervised Learning (ML/DL) for predictive modeling.

🚀 Key Features
Knowledge Discovery: Implemented K-Means Clustering and PCA to identify hidden respondent profiles.

Parallel Computing (HPC): Leveraged multi-core processing (n_jobs=-1) for hyperparameter tuning to optimize computational efficiency.

Comparative Modeling: Evaluated Decision Trees, XGBoost, and a Multi-Layer Perceptron (Neural Network).

Scientific Metrics: Advanced evaluation using AUC-ROC curves, Confusion Matrices, and Permutation Importance.

Model,Accuracy,F1-Score,AUC
Neural Network,78.5%,0.801,0.877
XGBoost,76.8%,0.780,0.856
Decision Tree,76.7%,0.785,0.845
