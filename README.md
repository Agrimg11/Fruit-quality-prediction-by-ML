# Fruit Quality Prediction System

A machine learning project that predicts fruit quality (**Good** or **Bad**) using binary classification. Trained and evaluated on a dataset of **8,000 samples** across 6 ML models with full EDA, feature importance analysis, and hyperparameter tuning.

---

## 📊 Dataset

- **Source:** Banana Quality Dataset
- **Samples:** 8,000
- **Target:** `Quality` → `Good` (1) / `Bad` (0)
- **Features:**

| Feature | Description |
|---|---|
| `Size` | Physical size of the banana |
| `Weight` | Weight measurement |
| `Sweetness` | Sugar content level |
| `Softness` | Texture/firmness |
| `HarvestTime` | Days since harvest |
| `Ripeness` | Ripeness score |
| `Acidity` | pH / acidity level |

---

## 🧠 Models Trained

| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear model |
| Decision Tree | Interpretable tree-based |
| **Random Forest** | Best performer |
| Gradient Boosting | Ensemble boosting |
| SVM (RBF Kernel) | Support vector classifier |
| K-Nearest Neighbors | Distance-based |

> ✅ **Best Model: SVM** (after GridSearchCV hyperparameter tuning)

---

## 📈 Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Random Forest | ~0.96 | ~0.97 |
| Gradient Boosting | ~0.94 | ~0.98 |
| SVM | ~0.98 | ~0.99 |
| Logistic Regression | ~0.87 | ~0.94 |

> *Exact values depend on your run — check the printed output after execution.*

---

## 🗂️ Project Structure

Fruit-quality-prediction-by-ML/
│
├── data/
│   └── banana_quality.csv
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── plots/
│   ├── plot1_class_distribution.png
│   ├── plot2_feature_distributions.png
│   ├── plot3_correlation_heatmap.png
│   ├── plot4_boxplots.png
│   ├── plot5_model_comparison.png
│   ├── plot6_confusion_matrix.png
│   ├── plot7_roc_curve.png
│   └── plot8_feature_importance.png
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md


---

## ⚙️ Setup & Usage

### 1. Clone the repository
```bash
git clone [https://github.com/Agrimg11/Fruit-quality-prediction-by-ML.git](https://github.com/Agrimg11/Fruit-quality-prediction-by-ML.git)
cd Fruit-quality-prediction-by-ML
2. Install dependencies
Bash
pip install -r requirements.txt
3. Run the project
Bash
python main.py
This will:

Load and explore the dataset.

Generate 8 EDA and evaluation plots (saved to the plots/ folder).

Train 6 ML models and compare them.

Run hyperparameter tuning via GridSearchCV on the Random Forest architecture.

Save the optimized model, scaler, and label encoder to the models/ folder.

Print a sample production inference prediction.

🔍 Pipeline Overview
Data Loading → EDA → Preprocessing → Model Training → Evaluation → Tuning → Save Artifacts
EDA — Class distribution, feature histograms, correlation heatmap, boxplots.

Preprocessing — Drop NaN values, Label Encoding, 80/20 train-test split, and StandardScaler application.

Training — 6 models executed alongside 5-fold cross-validation routines.

Evaluation — Tracks Metrics via Accuracy, ROC-AUC, Confusion Matrix, and ROC Curve charts.

Feature Importance — Maps Gini importance scores leveraging the Random Forest algorithm.

Tuning — Executes GridSearchCV to prevent model overfitting.

Inference — Serializes production files to classify unseen entries.

🧪 Sample Prediction
Python
import joblib
import pandas as pd

# Load structural pipelines from the models directory
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')
le = joblib.load('models/label_encoder.pkl')

# Define features profile schema
features = ['Size', 'Weight', 'Sweetness', 'Softness', 'HarvestTime', 'Ripeness', 'Acidity']
sample = pd.DataFrame([[1.2, 0.5, 2.1, -1.0, 0.3, 1.8, 0.9]], columns=features)

# Run Inference Pipeline
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)
print(f"Predicted Quality Class: {le.inverse_transform(prediction)[0]}")  # → 'Good' or 'Bad'
📦 Requirements
See requirements.txt for the full list. Core dependencies include:

pandas

numpy

scikit-learn

matplotlib

seaborn

joblib

🛠️ Tech Stack
## 🛠️ Tech Stack
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data-green?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Viz-blue)

📄 License
This project is open-source and available under the MIT License.

👤 Author
Agrim Gupta Feel free to connect or raise issues for enhancements!
