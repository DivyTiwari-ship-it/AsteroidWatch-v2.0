<div align="center">

# 🌌 AsteroidWatch v2.0

### Real-Time NASA Asteroid Hazard Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![NASA API](https://img.shields.io/badge/NASA-NeoWs%20API-0B3D91?style=for-the-badge&logo=nasa&logoColor=white)](https://api.nasa.gov/)
[![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Accuracy](https://img.shields.io/badge/Accuracy-92%25-brightgreen?style=for-the-badge)](https://github.com/DivyTiwari-ship-it/AsteroidWatch-v2.0)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/NASA_logo.svg/200px-NASA_logo.svg.png" width="100"/>

> **"Not all asteroids are harmless. This model knows the difference."**

---

</div>

## 🚀 Overview

**AsteroidWatch v2.0** is a machine learning system that connects to **NASA's real-time NeoWs (Near Earth Object Web Service) API** to classify whether an asteroid poses a **potential hazard to Earth** — with **92% accuracy**.

Unlike typical ML projects that rely on static CSV datasets, AsteroidWatch fetches **live asteroid data** directly from NASA's servers, preprocesses it on-the-fly, and runs it through a trained classification model — making every prediction as fresh as NASA's latest feed.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🛰️ **Live NASA API** | Fetches real-time Near Earth Object data from NASA's official NeoWs API |
| 🤖 **ML Classification** | Trained XGBoost + Random Forest ensemble for hazard detection |
| 📊 **92% Accuracy** | High-precision hazard classification on unseen asteroid data |
| 🌑 **Dark Visualizations** | All plots rendered with black-background, space-themed styling |
| 🔄 **Auto Preprocessing** | Raw API JSON → cleaned feature matrix in one pipeline |
| 📈 **Feature Importance** | Visual breakdown of which asteroid properties matter most |

---

## 🛸 How It Works

```
NASA NeoWs API  →  Raw JSON Data  →  Feature Engineering  →  ML Model  →  Hazard Prediction
      ↓                  ↓                   ↓                   ↓               ↓
  Live Feed         Diameter,           Normalization,       XGBoost /       ✅ Safe /
  (Date Range)     Velocity,            Encoding,           RandomForest    ☄️ Hazardous
                   Distance,            SMOTE Balance
                   Magnitude
```

---

## 📊 Model Performance

```
              precision    recall    f1-score
Not Hazardous    0.93       0.95       0.94
    Hazardous    0.90       0.87       0.88
     Accuracy                          0.92
```

| Metric | Score |
|--------|-------|
| Accuracy | **92%** |
| Precision | 90–93% |
| Recall | 87–95% |
| F1 Score | 88–94% |

---

## 🧠 Features Used

The model was trained on the following asteroid properties pulled from NASA API:

- `estimated_diameter_min` / `estimated_diameter_max` (km)
- `relative_velocity` (km/h)
- `miss_distance` (km from Earth)
- `absolute_magnitude_h`
- `is_sentry_object`
- `close_approach_date`
- `orbiting_body`

---

## 🗂️ Project Structure

```
AsteroidWatch-v2.0/
│
├── 📓 asteroid_watch.ipynb       # Main notebook — EDA, training, evaluation
├── 🔌 nasa_api_fetch.py          # Live NASA API integration module
├── 🤖 model/
│   ├── xgb_model.pkl             # Trained XGBoost model (joblib)
│   └── rf_model.pkl              # Trained Random Forest model
├── 📊 visualizations/
│   ├── feature_importance.png    # Feature importance (dark theme)
│   ├── confusion_matrix.png      # Confusion matrix heatmap
│   ├── roc_curve.png             # ROC-AUC curve
│   └── asteroid_distribution.png # Hazardous vs safe class distribution
├── 📄 requirements.txt
└── 📘 README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/DivyTiwari-ship-it/AsteroidWatch-v2.0.git
cd AsteroidWatch-v2.0
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Get Your FREE NASA API Key
> Register at → [https://api.nasa.gov/](https://api.nasa.gov/) (takes 30 seconds)

### 4. Add Your API Key
In `nasa_api_fetch.py` or the notebook, replace:
```python
API_KEY = "YOUR_NASA_API_KEY_HERE"
```

### 5. Run the Notebook
```bash
jupyter notebook asteroid_watch.ipynb
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
requests
joblib
imbalanced-learn
jupyter
```

---

## 🌠 Sample Visualizations

> All plots use black-background space-themed styling for maximum visual impact.

- **Confusion Matrix** — Model prediction accuracy breakdown
- **Feature Importance** — What drives the hazard prediction
- **ROC Curve** — Model discrimination ability (AUC score)
- **Class Distribution** — Hazardous vs. non-hazardous asteroid counts

---

## 🧪 Quick Prediction (After Setup)

```python
from nasa_api_fetch import get_asteroid_data
from joblib import load

# Load model
model = load('model/xgb_model.pkl')

# Fetch today's NASA asteroid data
df = get_asteroid_data(start_date="2024-01-01", end_date="2024-01-07")

# Predict
predictions = model.predict(df)
print("Hazardous Asteroids Today:", predictions.sum())
```

---

## 🤝 Connect With Me

<div align="center">

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Divyansh%20Tiwari-0077B5?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/divyanshtiwari)
[![GitHub](https://img.shields.io/badge/GitHub-DivyTiwari--ship--it-181717?style=for-the-badge&logo=github)](https://github.com/DivyTiwari-ship-it)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-FF6B6B?style=for-the-badge&logo=googlechrome)](https://divytiwari-ship-it.github.io/divyanshtiwari.github.io/)
[![Fiverr](https://img.shields.io/badge/Fiverr-Hire%20Me-1DBF73?style=for-the-badge&logo=fiverr)](https://www.fiverr.com/s/GzDkpxL)

</div>

---

<div align="center">

**Built with 🚀 by Divyansh Tiwari**

*Real data. Real NASA. Real predictions.*

⭐ **Star this repo if you found it useful!** ⭐

</div>
