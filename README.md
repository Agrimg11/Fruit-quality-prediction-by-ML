# Fruit Quality Prediction System

A machine learning project that predicts banana quality (**Good** or **Bad**) using binary classification. Trained and evaluated on a dataset of **8,000 samples** across 6 ML models with full EDA, feature importance analysis, and hyperparameter tuning.

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

> ✅ **Best Model: Random Forest** (after GridSearchCV hyperparameter tuning)

---

## 📈 Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Random Forest | ~0.92+ | ~0.97+ |
| Gradient Boosting | High | High |
| SVM | High | High |
| Logistic Regression | Baseline | Baseline |

> *Exact values depend on your run — check the printed output after execution.*

---

## 🗂️ Project Structure

```
banana-quality-prediction/
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
├── fruit_quality_prediction.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Usage

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/banana-quality-prediction.git
cd banana-quality-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the project
```bash
python fruit_quality_prediction.py
```

This will:
- Load and explore the dataset
- Generate 8 EDA and evaluation plots (saved to `plots/`)
- Train 6 ML models and compare them
- Run hyperparameter tuning on Random Forest
- Save the best model, scaler, and label encoder to `models/`
- Print a sample prediction

---

## 🔍 Pipeline Overview

```
Data Loading → EDA → Preprocessing → Model Training → Evaluation → Tuning → Save
```

1. **EDA** — Class distribution, feature histograms, correlation heatmap, boxplots
2. **Preprocessing** — Drop NaN, Label Encoding, 80/20 train-test split, StandardScaler
3. **Training** — 6 models with 5-fold cross-validation
4. **Evaluation** — Accuracy, ROC-AUC, Confusion Matrix, ROC Curve
5. **Feature Importance** — Random Forest importances
6. **Tuning** — GridSearchCV on Random Forest
7. **Inference** — Predict quality on new samples

---

## 🧪 Sample Prediction

```python
import joblib
import pandas as pd

model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')
le = joblib.load('models/label_encoder.pkl')

features = ['Size', 'Weight', 'Sweetness', 'Softness', 'HarvestTime', 'Ripeness', 'Acidity']
sample = pd.DataFrame([[1.2, 0.5, 2.1, -1.0, 0.3, 1.8, 0.9]], columns=features)

sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)
print(le.inverse_transform(prediction)[0])  # → 'Good' or 'Bad'
```

---

## 📦 Requirements

See `requirements.txt` for full list. Core dependencies:
- `pandas`, `numpy`
- `scikit-learn`
- `matplotlib`, `seaborn`
- `joblib`

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data-green?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Viz-blue)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Agrim Gupta**  
Feel free to connect or raise issues for suggestions!

