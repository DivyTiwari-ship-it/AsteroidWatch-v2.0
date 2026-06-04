import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('asteroid_model.pkl', 'rb'))

st.set_page_config(page_title="AstroShield AI", page_icon="🛸", layout="wide")

st.title("🛸 AstroShield AI")
st.subheader("Real-time Asteroid Hazard Predictor — Powered by NASA Data")

col1, col2 = st.columns(2)

with col1:
    magnitude = st.slider("Absolute Magnitude (H)", 10.0, 35.0, 20.0)
    diameter  = st.slider("Estimated Diameter (km)", 0.001, 10.0, 0.5)
    velocity  = st.slider("Velocity (km/s)", 1.0, 40.0, 15.0)

with col2:
    miss_dist = st.number_input("Miss Distance (km)", value=500000.0)

if st.button("🔍 Predict Threat Level", use_container_width=True):
    threat   = velocity / miss_dist
    size_v   = diameter * velocity
    features = np.array([[magnitude,
                          diameter * 0.8,
                          diameter * 1.2,
                          miss_dist,
                          velocity,
                          diameter,
                          threat,
                          size_v]])
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    st.divider()
    if pred == 1:
        st.error(f"⚠️ HAZARDOUS — {prob*100:.1f}% confidence")
    else:
        st.success(f"✅ SAFE — {(1-prob)*100:.1f}% confidence")

st.divider()
st.caption("Data: NASA NeoWs | JPL Horizons | SBDB Orbital Elements")
