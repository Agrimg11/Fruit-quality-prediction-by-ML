# Fruit Quality Prediction

Binary classification system predicting fruit quality (**Good** / **Bad**) using 6 ML models, trained on 8,000 samples with full EDA, feature importance analysis, and hyperparameter tuning.

---

## 📊 Dataset

**Source:** Banana Quality Dataset &nbsp;|&nbsp; **Samples:** 8,000 &nbsp;|&nbsp; **Target:** `Quality` → `Good (1)` / `Bad (0)`

| Feature | Description |
|---|---|
| `Size` | Physical size |
| `Weight` | Weight measurement |
| `Sweetness` | Sugar content level |
| `Softness` | Texture / firmness |
| `HarvestTime` | Days since harvest |
| `Ripeness` | Ripeness score |
| `Acidity` | pH / acidity level |

---

## 🧠 Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | ~0.87 | ~0.94 |
| Decision Tree | ~0.92 | ~0.92 |
| Random Forest | ~0.96 | ~0.97 |
| Gradient Boosting | ~0.94 | ~0.98 |
| **SVM (Best ✅)** | **~0.98** | **~0.99** |
| K-Nearest Neighbors | ~0.95 | ~0.94 |

> Best model: **SVM** after GridSearchCV hyperparameter tuning.

---

## 🗂️ Project Structure

```
Fruit-quality-prediction-by-ML/
├── data/
│   └── banana_quality.csv
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
├── plots/              
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
git clone https://github.com/Agrimg11/Fruit-quality-prediction-by-ML.git
cd Fruit-quality-prediction-by-ML
pip install -r requirements.txt
python main.py
```

**`main.py` will:**
- Run EDA and generate 8 plots → `plots/`
- Train 6 models with 5-fold cross-validation
- Tune best model via GridSearchCV
- Save model artifacts → `models/`
- Print a sample inference prediction

---

## 🔍 Pipeline

```
Data Loading → EDA → Preprocessing → Training → Evaluation → Tuning → Inference
```

**Preprocessing:** Drop NaN → Label Encoding → 80/20 split → StandardScaler

**Evaluation:** Accuracy, ROC-AUC, Confusion Matrix, ROC Curve

---

## 🧪 Sample Inference

```python
import joblib, pandas as pd

model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')
le    = joblib.load('models/label_encoder.pkl')

features = ['Size', 'Weight', 'Sweetness', 'Softness', 'HarvestTime', 'Ripeness', 'Acidity']
sample = pd.DataFrame([[1.2, 0.5, 2.1, -1.0, 0.3, 1.8, 0.9]], columns=features)

prediction = model.predict(scaler.transform(sample))
print(le.inverse_transform(prediction)[0])  # → 'Good' or 'Bad'
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data-green?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Viz-blue)

---

## 📄 License

MIT License — open source and free to use.

## 👤 Author

**Agrim Gupta** — feel free to open issues or contribute!
