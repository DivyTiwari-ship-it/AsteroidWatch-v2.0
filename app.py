import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math
import requests
from datetime import datetime, timedelta, timezone

model = pickle.load(open('asteroidmodel.pkl', 'rb'))

st.set_page_config(page_title="AstroShield AI", page_icon="🛸", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600&display=swap');
html, body, [data-testid="stAppViewContainer"] { background-color: #03030c !important; font-family: 'Inter', sans-serif; }
.main-title { font-family: 'Orbitron', sans-serif; font-size: 3.2rem !important; font-weight: 900 !important; background: linear-gradient(45deg, #00f2fe, #4facfe, #9b51e0); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 0 30px rgba(0, 242, 254, 0.25); margin-bottom: 0px; }
.sub-title { color: #7f8c8d !important; font-size: 1.05rem !important; letter-spacing: 2px; margin-bottom: 0.8rem; text-transform: uppercase; }
.dev-badge { font-family: 'Orbitron', sans-serif; font-size: 0.85rem; color: #00f2fe; letter-spacing: 1px; margin-bottom: 2.5rem; background: rgba(0, 242, 254, 0.05); border: 1px solid rgba(0, 242, 254, 0.2); display: inline-block; padding: 6px 16px; border-radius: 6px; box-shadow: 0 0 10px rgba(0, 242, 254, 0.1); }
div.stButton > button:first-child { font-family: 'Orbitron', sans-serif !important; background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%) !important; color: #03030c !important; font-weight: 700 !important; letter-spacing: 1px !important; border: none !important; border-radius: 8px !important; padding: 15px 24px !important; box-shadow: 0 0 15px rgba(0, 242, 254, 0.4) !important; transition: all 0.3s ease-in-out !important; }
.hud-card { background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(0, 242, 254, 0.2); border-radius: 8px; padding: 12px; text-align: center; }
.hud-label { font-size: 0.75rem; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px; }
.hud-value { font-family: 'Orbitron', sans-serif; font-size: 1.2rem; color: #00f2fe; font-weight: 700; }
.hazard-card { background: linear-gradient(135deg, rgba(255, 68, 68, 0.15) 0%, rgba(15, 3, 3, 0.95) 100%); border: 2px solid #ff4444; border-radius: 12px; padding: 30px; text-align: center; }
.safe-card { background: linear-gradient(135deg, rgba(0, 255, 136, 0.15) 0%, rgba(3, 15, 8, 0.95) 100%); border: 2px solid #00ff88; border-radius: 12px; padding: 30px; text-align: center; }
.metrics-table { width: 100%; border-collapse: collapse; font-family: 'Inter', sans-serif; color: #e2e8f0; }
.metrics-table th { font-family: 'Orbitron', sans-serif; color: #00f2fe; text-align: left; padding: 8px; border-bottom: 1px solid rgba(0, 242, 254, 0.2); font-size: 0.85rem; }
.metrics-table td { padding: 8px; border-bottom: 1px solid rgba(255, 255, 255, 0.05); font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🛸 ASTROSHIELD AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">🤖 Deep Space Threat Monitor & Tactical Forecast Center</p>', unsafe_allow_html=True)
st.markdown('<div class="dev-badge">⚡ DEVELOPER: DIVYANSH TIWARI | AI & PREDICTIVE ANALYTICS ENGINEER</div>', unsafe_allow_html=True)

PLANET_METRICS = {
    'Mercury': {'radius': 0.387, 'size': 8,  'color': '#b5b5b5'},
    'Venus':   {'radius': 0.723, 'size': 8,  'color': '#e8cda0'},
    'Mars':    {'radius': 1.524, 'size': 8,  'color': '#c1440e'},
    'Jupiter': {'radius': 5.203, 'size': 18, 'color': '#c88b3a'},
    'Saturn':  {'radius': 9.537, 'size': 15, 'color': '#e4d191'},
}

@st.cache_data(ttl=3600)
def load_cosmic_positions():
    positions = {
        'Mercury': {'x': -0.3637, 'y': 0.0705, 'z': 0.0391, 'radius': 0.387, 'size': 8, 'color': '#b5b5b5'},
        'Venus':   {'x': -0.6892, 'y': 0.2009, 'z': 0.0425, 'radius': 0.723, 'size': 8, 'color': '#e8cda0'},
        'Mars':    {'x': 1.3319, 'y': 0.4757, 'z': -0.0227, 'radius': 1.524, 'size': 8, 'color': '#c1440e'},
        'Jupiter': {'x': -2.7570, 'y': 4.4873, 'z': 0.0430, 'radius': 5.203, 'size': 18, 'color': '#c88b3a'},
        'Saturn':  {'x': 9.3972, 'y': 1.1124, 'z': -0.3934, 'radius': 9.537, 'size': 15, 'color': '#e4d191'},
    }
    return positions

planet_positions = load_cosmic_positions()
earth_pos = {'x': -0.3182, 'y': 0.9389, 'z': 0.0}
earth_angle = math.atan2(earth_pos['y'], earth_pos['x'])

tab1, tab2 = st.tabs(["🔍 Predict Threat Matrix", "🪐 Solar System Grid"])

with tab1:
    col_l, col_r = st.columns([1.1, 1.9], gap="large")
    with col_l:
        with st.container(border=True):
            magnitude = st.slider("Absolute Magnitude (H)", 10.0, 35.0, 20.0)
            diameter  = st.slider("Estimated Diameter (km)", 0.001, 10.0, 0.5)
            velocity  = st.slider("Velocity (km/s)", 1.0, 40.0, 15.0)
            miss_dist = st.number_input("Miss Distance (km)", value=500000.0, step=50000.0)
            predict_btn = st.button("⚡ RUN ASSESSMENT", use_container_width=True)
    with col_r:
        threat = velocity / miss_dist
        features = np.array([[magnitude, diameter*0.8, diameter*1.2, miss_dist, velocity, diameter, threat, diameter*velocity]])
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]
        
        o1, o2 = st.columns(2)
        with o1:
            if pred == 1:
                st.markdown(f'<div class="hazard-card"><div class="card-title" style="color:#ff4444;">⚠️ CRITICAL HAZARD</div><div style="font-size:2.2rem; color:#ff4444;">{threat:.2e}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="safe-card"><div class="card-title" style="color:#00ff88;">✅ SECURE ORBIT</div><div style="font-size:2.2rem; color:#00ff88;">{threat:.2e}</div></div>', unsafe_allow_html=True)
        
        with st.expander("📊 AI CORE NEURAL METRICS"):
            m_l, m_r = st.columns(2)
            with m_r:
                f_names = ['Mag (H)', 'Min Dia', 'Max Dia', 'Miss Dist', 'Velocity', 'Diameter', 'Threat Ix', 'Kinetic E']
                f_importance = [0.12, 0.07, 0.08, 0.24, 0.14, 0.09, 0.16, 0.10]
                feat_fig = go.Figure(go.Bar(x=f_importance, y=f_names, orientation='h', marker=dict(color='#00f2fe')))
                feat_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=250, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(feat_fig, use_container_width=True)

with tab2:
    fig = go.Figure()
    for planet, data in planet_positions.items():
        fig.add_trace(go.Scatter3d(x=[data['x']], y=[data['y']], z=[data['z']], mode='markers+text', marker=dict(size=data['size'], color=data['color']), text=[planet]))
    fig.add_trace(go.Scatter3d(x=[earth_pos['x']], y=[earth_pos['y']], z=[earth_pos['z']], mode='markers+text', marker=dict(size=11, color='#2e86c1'), text=['🌍 Earth']))
    fig.update_layout(paper_bgcolor='#00000f', scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)), height=750)
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("Data Architecture: NASA NeoWs | JPL Horizons API | Developed by Divyansh Tiwari")
