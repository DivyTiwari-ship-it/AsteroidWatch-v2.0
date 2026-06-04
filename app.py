import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math
from datetime import datetime

# --- CONFIGURATION ---
model = pickle.load(open('asteroidmodel.pkl', 'rb'))
st.set_page_config(page_title="AstroShield AI", page_icon="🛸", layout="wide")

# --- CSS STYLES ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600&display=swap');
html, body, [data-testid="stAppViewContainer"] { background-color: #03030c !important; font-family: 'Inter', sans-serif; }
.main-title { font-family: 'Orbitron', sans-serif; font-size: 3.2rem !important; font-weight: 900 !important; background: linear-gradient(45deg, #00f2fe, #4facfe, #9b51e0); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 0 30px rgba(0, 242, 254, 0.25); margin-bottom: 0px; }
.sub-title { color: #7f8c8d !important; font-size: 1.05rem !important; letter-spacing: 2px; margin-bottom: 0.8rem; text-transform: uppercase; }
.dev-badge { font-family: 'Orbitron', sans-serif; font-size: 0.85rem; color: #00f2fe; letter-spacing: 1px; margin-bottom: 2.5rem; background: rgba(0, 242, 254, 0.05); border: 1px solid rgba(0, 242, 254, 0.2); display: inline-block; padding: 6px 16px; border-radius: 6px; box-shadow: 0 0 10px rgba(0, 242, 254, 0.1); }
div.stButton > button:first-child { font-family: 'Orbitron', sans-serif !important; background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%) !important; color: #03030c !important; font-weight: 700 !important; border: none !important; border-radius: 8px !important; padding: 15px 24px !important; box-shadow: 0 0 15px rgba(0, 242, 254, 0.4) !important; }
.hud-card { background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(0, 242, 254, 0.2); border-radius: 8px; padding: 12px; text-align: center; }
.hud-label { font-size: 0.75rem; color: #7f8c8d; text-transform: uppercase; }
.hud-value { font-family: 'Orbitron', sans-serif; font-size: 1.2rem; color: #00f2fe; font-weight: 700; }
.hazard-card { background: linear-gradient(135deg, rgba(255, 68, 68, 0.15) 0%, rgba(15, 3, 3, 0.95) 100%); border: 2px solid #ff4444; border-radius: 12px; padding: 20px; text-align: center; }
.safe-card { background: linear-gradient(135deg, rgba(0, 255, 136, 0.15) 0%, rgba(3, 15, 8, 0.95) 100%); border: 2px solid #00ff88; border-radius: 12px; padding: 20px; text-align: center; }
.metrics-table { width: 100%; border-collapse: collapse; color: #e2e8f0; }
.metrics-table th { font-family: 'Orbitron'; color: #00f2fe; text-align: left; padding: 8px; border-bottom: 1px solid rgba(0, 242, 254, 0.2); }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<h1 class="main-title">🛸 ASTROSHIELD AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">🤖 Deep Space Threat Monitor & Tactical Forecast Center</p>', unsafe_allow_html=True)
st.markdown('<div class="dev-badge">⚡ DEVELOPER: DIVYANSH TIWARI | AI & PREDICTIVE ANALYTICS ENGINEER</div>', unsafe_allow_html=True)

# --- APP LOGIC ---
tab1, tab2 = st.tabs(["🔍 Predict Threat Matrix", "🪐 Solar System Grid"])

with tab1:
    col_l, col_r = st.columns([1.1, 1.9], gap="large")
    with col_l:
        st.markdown("<p style='font-family:\"Orbitron\"; color: #00f2fe;'>🎛️ CONTROL INTERFACE</p>", unsafe_allow_html=True)
        with st.container(border=True):
            magnitude = st.slider("Absolute Magnitude (H)", 10.0, 35.0, 20.0)
            diameter  = st.slider("Diameter (km)", 0.001, 10.0, 0.5)
            velocity  = st.slider("Velocity (km/s)", 1.0, 40.0, 15.0)
            miss_dist = st.number_input("Miss Distance (km)", value=500000.0, step=50000.0)
            predict_btn = st.button("⚡ RUN ASSESSMENT", use_container_width=True)

    with col_r:
        st.markdown("<p style='font-family:\"Orbitron\"; color: #00f2fe;'>📊 LIVE TELEMETRY HUD</p>", unsafe_allow_html=True)
        
        # HUD CARDS
        h1, h2, h3, h4 = st.columns(4)
        h1.markdown(f'<div class="hud-card"><div class="hud-label">Mag</div><div class="hud-value">{magnitude}</div></div>', unsafe_allow_html=True)
        h2.markdown(f'<div class="hud-card"><div class="hud-label">Size</div><div class="hud-value">{diameter}km</div></div>', unsafe_allow_html=True)
        h3.markdown(f'<div class="hud-card"><div class="hud-label">Vel</div><div class="hud-value">{velocity}km/s</div></div>', unsafe_allow_html=True)
        h4.markdown(f'<div class="hud-card"><div class="hud-label">Dist</div><div class="hud-value">{miss_dist/1000:.0f}k km</div></div>', unsafe_allow_html=True)
        
        # ML PREDICTION
        threat = velocity / miss_dist
        features = np.array([[magnitude, diameter*0.8, diameter*1.2, miss_dist, velocity, diameter, threat, diameter*velocity]])
        pred = model.predict(features)[0]
        
        st.markdown("<br>", unsafe_allow_html=True)
        out1, out2 = st.columns([1.2, 1.8])
        with out1:
            if pred == 1:
                st.markdown('<div class="hazard-card"><div style="font-size:1.2rem; font-weight:bold; color:#ff4444;">⚠️ CRITICAL</div><div style="font-size:2rem; font-family:\'Orbitron\';">HAZARD</div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="safe-card"><div style="font-size:1.2rem; font-weight:bold; color:#00ff88;">✅ SECURE</div><div style="font-size:2rem; font-family:\'Orbitron\';">ORBIT</div></div>', unsafe_allow_html=True)
        
        with out2:
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=12, color='#2e86c1'), name='Earth'))
            fig.add_trace(go.Scatter3d(x=[1], y=[1], z=[0], mode='markers', marker=dict(size=8, color='#ff4444'), name='Asteroid'))
            fig.update_layout(paper_bgcolor='#03030c', scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)), margin=dict(l=0,r=0,t=0,b=0), height=250)
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("<p style='font-family:\"Orbitron\"; color: #00f2fe;'>🛰️ SYSTEM COORDINATES</p>", unsafe_allow_html=True)
    fig = go.Figure()
    # Mocking planetary grid for visualization
    planets = {'Sun': [0,0,0], 'Mercury': [0.4, 0, 0], 'Venus': [0.7, 0, 0], 'Earth': [1, 0, 0]}
    for name, pos in planets.items():
        fig.add_trace(go.Scatter3d(x=[pos[0]], y=[pos[1]], z=[pos[2]], mode='markers+text', text=[name], marker=dict(size=10)))
    fig.update_layout(paper_bgcolor='#00000f', scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)), height=600)
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("Data Architecture: NASA NeoWs | JPL Horizons API | Developed by Divyansh Tiwari")
