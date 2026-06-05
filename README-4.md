<div align="center">

# 🛸 AstroShield AI — AsteroidWatch v2.0

### Real-Time NASA Asteroid Hazard Detection & 3D Solar System Visualizer

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://asteroidwatch-v2o.streamlit.app)
[![NASA API](https://img.shields.io/badge/NASA-NeoWs%20API-0B3D91?style=for-the-badge&logo=nasa&logoColor=white)](https://api.nasa.gov/)
[![JPL](https://img.shields.io/badge/JPL-Horizons%20API-1a1a2e?style=for-the-badge)](https://ssd.jpl.nasa.gov/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-FF6600?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![SMOTE](https://img.shields.io/badge/SMOTE-Balanced-brightgreen?style=for-the-badge)](https://imbalanced-learn.org/)

<br>

> **"38,573 real asteroids. 3 live NASA APIs. 1 model deciding what's dangerous."**

<br>

🌐 **[Live App → asteroidwatch-v2o.streamlit.app](https://asteroidwatch-v2o.streamlit.app)**

---

</div>

## 🚀 Overview

**AstroShield AI (AsteroidWatch v2.0)** is a full-stack ML + space visualization system that:

- Fetches **38,573+ real asteroids** from NASA's NeoWs API (2015–2026)
- Pulls **live planet positions** from NASA JPL Horizons API using real Keplerian orbital mechanics
- Classifies asteroid hazard using an **XGBoost model trained with SMOTE balancing**
- Renders an **interactive 3D Solar System** with real asteroid positions, planets, Lagrange points, space stations, and a Dyson Ring
- Deploys as a **Streamlit web app** — anyone can check if an asteroid is dangerous

---

## ✨ What Makes This Different

| Feature | Detail |
|---|---|
| 🛰️ **3 Live APIs** | NASA NeoWs + JPL Horizons + JPL SBDB Orbital Elements |
| 🪐 **Real Orbital Mechanics** | Kepler's equation (Newton-Raphson) to compute true 3D asteroid positions |
| 🌌 **3D Solar System** | Interactive Plotly viz with real planet positions, asteroid orbits, Lagrange points |
| ⚖️ **SMOTE Balancing** | Class imbalance fixed — balanced hazardous vs. safe training data |
| 🛸 **Streamlit App** | Live deployment — sliders, real-time prediction, confidence score |
| 🌍 **Exoplanet Data** | Habitability scoring via NASA Exoplanet Archive API |
| ☀️ **Solar Flare Risk** | NASA DONKI API integrated as a risk feature |

---

## 🧠 ML Pipeline

```
NASA NeoWs API (2015–2026)
        ↓
  38,573 Asteroids Fetched
        ↓
  Feature Engineering
  • min/max diameter (km)
  • velocity (km/s)
  • miss distance (km)
  • absolute magnitude H
  • threat_score = velocity / miss_distance
  • size_velocity = diameter × velocity
  • flare_risk (from NASA DONKI)
  • habitability_score (from Exoplanet Archive)
        ↓
  Train/Test Split (80/20, stratified)
        ↓
  SMOTE Oversampling (balanced classes)
        ↓
  XGBoost Classifier
  (n_estimators=500, max_depth=5, lr=0.05)
        ↓
  Hazardous / Safe Prediction + Confidence %
```

---

## 🌌 3D Visualization Features

The interactive 3D Solar System includes:

- ☀️ **Sun** with Dyson Ring energy collectors
- 🌍 **Earth** with protection shield ring
- 🪐 **Mercury, Venus, Mars, Jupiter, Saturn** — real JPL positions
- 🛸 **Space Stations** at Lagrange points (L4 Human Colony, L5 Industrial Station, Mars Mining Base, Titan Frontier)
- ☄️ **Hazardous asteroids** — red markers, sized by diameter
- ✅ **Safe asteroids** — green markers, 300+ plotted
- 🔆 **Dyson Ring** around the Sun
- Dotted **orbital paths** for all planets
- **Star field** background

---

## 🌐 Live Streamlit App

The deployed app lets you predict asteroid threat in real time:

```
Inputs:
  • Absolute Magnitude (H)     → slider 10–35
  • Estimated Diameter (km)    → slider 0.001–10
  • Velocity (km/s)            → slider 1–40
  • Miss Distance (km)         → number input

Output:
  ⚠️  HAZARDOUS — XX.X% confidence
  ✅  SAFE      — XX.X% confidence
```

🔗 **[Try it live →](https://asteroidwatch-v2o.streamlit.app)**

---

## 📡 APIs Used

| API | Purpose |
|-----|---------|
| [NASA NeoWs](https://api.nasa.gov/) | Real asteroid close-approach data (2015–2026) |
| [NASA JPL Horizons](https://ssd.jpl.nasa.gov/horizons/) | Real planet XYZ positions (vector ephemeris) |
| [JPL SBDB](https://ssd-api.jpl.nasa.gov/sbdb_query.api) | Orbital elements (a, e, i, Ω, ω, M) for real asteroid orbits |
| [NASA DONKI](https://api.nasa.gov/) | Solar flare class data → flare risk feature |
| [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) | Habitability scoring feature |

---

## 🗂️ Project Structure

```
AsteroidWatch-v2.0/
│
├── 📓 AsteroidWatch_v2_0.ipynb    # Full pipeline: fetch → EDA → train → visualize
├── 🌐 app.py                      # Streamlit web app
├── 🤖 asteroidmodel.pkl           # Trained XGBoost model
├── 📊 asteroid_positions.csv      # Real orbital positions (from JPL SBDB)
├── 📄 requirements.txt
└── 📘 README.md
```

---

## ⚙️ Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/DivyTiwari-ship-it/AsteroidWatch-v2.0.git
cd AsteroidWatch-v2.0
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get a FREE NASA API Key
Register at → **[https://api.nasa.gov/](https://api.nasa.gov/)**

### 4. Add your key in the notebook
```python
API_KEY = 'YOUR_NASA_API_KEY_HERE'
```

### 5. Run the Streamlit app
```bash
streamlit run app.py
```

---

## 📦 Requirements

```
pandas
numpy
requests
scikit-learn
xgboost
imbalanced-learn
plotly
matplotlib
seaborn
streamlit
joblib
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

*Real NASA data. Real orbital mechanics. Real predictions.*

⭐ **Star this repo if you found it useful!** ⭐

</div>
