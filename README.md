# All-core-concepts-of-ML
A comprehensive, production-ready repository mapping end-to-end Machine Learning workflows. Features structured implementations of Data Acquisition, EDA, Preprocessing, Feature Engineering, Core Models (Regression, Classification, Clustering), Ensembles, and Optimization Pipelines.
Welcome to the ultimate reference repository for Machine Learning workflows and foundational algorithms. This project is meticulously structured to mirror an industry-standard production pipeline, moving from raw data extraction to model evaluation and pipeline deployment. 

Each folder serves as an isolated, clean module detailing the mathematical intuition, implementation code, and best practices for specific data science challenges.

---
🛠️ Core Concepts Covered
1. Data Engineering & Preprocessing
Multi-Source Extraction: Advanced data ingestion using SQL relational joins, raw JSON parsing, automated API requests, and BeautifulSoup/Scrapy for Web Scraping.

Advanced Imputation: Handling missing data under MCAR, MAR, and MNAR mechanisms using statistical means and distance-based estimators (KNN Imputer).

Anomalies & Imbalance: Outlier identification via IQR/Z-score thresholds, structural feature parsing for mixed string-numeric inputs, and class balancing using SMOTE oversampling.

2. Feature Transformations
Geometric stabilization using PowerTransformers (Box-Cox, Yeo-Johnson) and categorical processing avoiding the dummy variable trap.

Capturing cyclical chronological patterns using Trigonometric (Sine/Cosine) Transformations for date-time variables.

Dimensionality reduction using Principal Component Analysis (PCA) to isolate maximum variance orthogonal axes.

3. Algorithmic Deep Dive & Ensembles
Comprehensive coverage of parametric, non-parametric, and distance/density-based algorithms including K-Means, DBSCAN, and Hierarchical Agglomerative clustering.

Optimization: Core gradient descent mathematical variations (Batch, Stochastic, Mini-Batch).

Advanced Ensembles: Comparative setups of parallel Bagging (Random Forest), sequential Gradient Boosting (XGBoost), and combined Hard/Soft Voting classifiers.

Next-Gen Tuning: State-of-the-art Bayesian Optimization hyperparameter searches using Optuna paired with automated real-time trial pruning.

4. Production Grade Pipelines
Architectural decoupling of extraction steps from training logic using Scikit-Learn Pipelines and ColumnTransformer to completely isolate data workflows and prevent data leakage during model training.

🚀 How to Use This Repository
Prerequisites
Ensure you have Python 3.8+ installed along with the essential data science stack:

Bash
```pip install numpy pandas scikit-learn matplotlib seaborn xgboost optuna ydata-profiling```
Navigating Notebooks
Clone the repository to your local directory:

Bash
```git clone https://github.com/Mohammed-Imran-Meman/All-core-concepts-of-ML.git```
Navigate to any modular directory (e.g., 04_Feature_Engineering) to run specific Jupyter Notebook scripts and view isolated code implementations.

🤝 Contributions
Feel free to open an issue or submit a pull request if you want to expand these pipelines with Deep Learning frameworks, neural architectures, or additional production engineering patterns!
