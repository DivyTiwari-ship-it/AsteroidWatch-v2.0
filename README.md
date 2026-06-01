# 🛸 AstroShield AI — Space Habitation Risk Analyzer

> An AI-powered system that analyzes asteroid threats, solar activity, and planetary habitability to determine how safe it is for humans to live in space.

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io)
[![NASA API](https://img.shields.io/badge/Data-NASA%20API-green)](https://api.nasa.gov)
[![Accuracy](https://img.shields.io/badge/Accuracy-85.3%25-brightgreen)]()
[![Recall](https://img.shields.io/badge/Recall-91.2%25-brightgreen)]()

---

## 🌌 What is AstroShield AI?

AstroShield AI is a multi-source space safety analysis system that combines real NASA data to assess risks for human habitation in space. It detects potentially hazardous asteroids, monitors solar flare risk, and scores planetary habitability — all in one pipeline.

**Vision:** To build the foundation of an AI system that could one day help scientists determine safe zones for human space stations and long-term space living.

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 85.3% |
| Precision | 36.6% |
| Recall | **91.2%** |
| F1 Score | 52.3% |

> **Why Recall matters most here:** Missing a hazardous asteroid is far more dangerous than a false alarm. A 91.2% Recall means the model catches 9 out of 10 dangerous asteroids.

---

## 🔭 Data Sources

| Source | API | Data |
|--------|-----|------|
| NASA NeoWs | `api.nasa.gov` | 38,027 asteroid records |
| DONKI Solar Flare API | `api.nasa.gov` | Solar flare risk score |
| NASA Exoplanet Archive | `exoplanetarchive.ipac.caltech.edu` | Habitability score |

---

## 🧠 Features Used

| Feature | Description |
|---------|-------------|
| `absolute_magnitude_h` | Asteroid brightness (size indicator) |
| `min_diameter` / `max_diameter` | Size range in km |
| `miss_dis_km` | How close it passed Earth |
| `velocity_km_s` | Speed of asteroid |
| `avg_flare_risk` | Solar flare activity score |
| `habitability_score` | Exoplanet-based habitability index |
| `diameter_avg` | Engineered: average size |
| `threat_score` | Engineered: velocity / miss distance |
| `size_velocity` | Engineered: impact force estimate |

---

## ⚙️ Tech Stack

- **Language:** Python 3.13
- **Model:** XGBoost Classifier
- **Imbalance Handling:** SMOTE (34k safe vs 3.4k hazardous)
- **Libraries:** Pandas, Scikit-learn, XGBoost, imbalanced-learn, Requests
- **Platform:** Google Colab + Google Drive

---

## 🚀 Project Pipeline

NASA APIs → Data Collection → Feature Engineering → SMOTE → XGBoost → Evaluation → FastAPI (coming soon) → Azure Deploy (coming soon)



---

## 🔜 Upcoming Features

- [ ] FastAPI REST endpoint for real-time asteroid risk prediction
- [ ] Live ISS location tracking
- [ ] Interactive 3D Earth + asteroid visualization
- [ ] Azure cloud deployment
- [ ] Safe orbit zone recommender

---

## 👨‍💻 Author

**Divyansh Tiwari**
3rd Year BCA | ML Engineer

[![GitHub](https://img.shields.io/badge/GitHub-DivyTiwari--ship--it-black)](https://github.com/DivyTiwari-ship-it)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-blue)](https://divytiwari-ship-it.github.io/divyanshtiwari.github.io/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-20BEFF)](https://www.kaggle.com/code/divyanshtiwari01/spacex)
[![Fiverr](https://img.shields.io/badge/Fiverr-Hire%20Me-1DBF73)](https://www.fiverr.com/s/GzDkpxL)

---

## 📜 License

MIT License — feel free to use and build on this project.

