import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math

model = pickle.load(open('asteroidmodel.pkl', 'rb'))

st.set_page_config(page_title="AstroShield AI", page_icon="🛸", layout="wide")
st.title("🛸 AstroShield AI")
st.subheader("Real NASA + JPL Data | Asteroid Hazard Predictor + Solar System")

tab1, tab2 = st.tabs(["🔍 Predict Threat", "🪐 Solar System"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        magnitude = st.slider("Absolute Magnitude (H)", 10.0, 35.0, 20.0)
        diameter  = st.slider("Estimated Diameter (km)", 0.001, 10.0, 0.5)
        velocity  = st.slider("Velocity (km/s)", 1.0, 40.0, 15.0)
    with col2:
        miss_dist = st.number_input("Miss Distance (km)", value=500000.0)

    # Indentation Fixed Here 👇
    if st.button("🔍 Predict Threat Level", use_container_width=True):
        threat   = velocity / miss_dist
        size_v   = diameter * velocity
        features = np.array([[magnitude, diameter*0.8, diameter*1.2,
                              miss_dist, velocity, diameter, threat, size_v]])
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]
      
        st.divider()
        if pred == 1:
            st.error(f"⚠️ HAZARDOUS — {prob*100:.1f}% confidence")
            st.metric("Threat Score", f"{threat:.2e}", delta="HIGH RISK")
        else:
            st.success(f"✅ SAFE — {(1-prob)*100:.1f}% confidence")
            st.metric("Threat Score", f"{threat:.2e}", delta="LOW RISK")

        st.divider()
        st.info("👇Go to 'Solar System' tab — see asteroid real position !")

with tab2:
    st.caption("Real JPL Horizons planet positions + SBDB Keplerian asteroid orbits")

    ast_df = pd.read_csv('asteroid_positions.csv')

    planet_positions = {
        'Mercury': {'x': -0.3637, 'y':  0.0705, 'z':  0.0391, 'color': '#b5b5b5', 'size': 6,  'radius': 0.387},
        'Venus':   {'x': -0.6892, 'y':  0.2009, 'z':  0.0425, 'color': '#e8cda0', 'size': 8,  'radius': 0.723},
        'Mars':    {'x':  1.3319, 'y':  0.4757, 'z': -0.0227, 'color': '#c1440e', 'size': 7,  'radius': 1.524},
        'Jupiter': {'x': -2.7570, 'y':  4.4873, 'z':  0.0430, 'color': '#c88b3a', 'size': 18, 'radius': 5.203},
        'Saturn':  {'x':  9.3972, 'y':  1.1124, 'z': -0.3934, 'color': '#e4d191', 'size': 15, 'radius': 9.537},
    }
    earth_pos = {'x': -0.3182, 'y': 0.9389, 'z': 0.0}
    earth_angle = math.atan2(earth_pos['y'], earth_pos['x'])
    mars_angle  = math.atan2(planet_positions['Mars']['y'], planet_positions['Mars']['x'])

    station_config = {
        'Mercury L1 — Solar Harvester': {'x': planet_positions['Mercury']['x'] * 0.85, 'y': planet_positions['Mercury']['y'] * 0.85, 'z': 0, 'color': '#FFD700'},
        'Venus L2 — Research Post':     {'x': planet_positions['Venus']['x'] * 1.15,   'y': planet_positions['Venus']['y'] * 1.15,   'z': 0, 'color': '#FFA07A'},
        'Earth L4 — Human Colony':      {'x': 1.0 * math.cos(earth_angle + math.radians(60)), 'y': 1.0 * math.sin(earth_angle + math.radians(60)), 'z': 0, 'color': '#00FFFF'},
        'Earth L5 — Industrial':        {'x': 1.0 * math.cos(earth_angle - math.radians(60)), 'y': 1.0 * math.sin(earth_angle - math.radians(60)), 'z': 0, 'color': '#FF00FF'},
        'Mars Phobos — Mining Base':    {'x': planet_positions['Mars']['x'] * 1.02,    'y': planet_positions['Mars']['y'] * 1.02,    'z': 0.01, 'color': '#FF6600'},
        'Jupiter L4 — Trojan Outpost':  {'x': 5.203 * math.cos(math.atan2(planet_positions['Jupiter']['y'], planet_positions['Jupiter']['x']) + math.radians(60)), 'y': 5.203 * math.sin(math.atan2(planet_positions['Jupiter']['y'], planet_positions['Jupiter']['x']) + math.radians(60)), 'z': 0, 'color': '#98FB98'},
        'Titan Orbit — Outer Frontier': {'x': planet_positions['Saturn']['x'] * 1.008, 'y': planet_positions['Saturn']['y'] * 1.008, 'z': 0.05, 'color': '#DDA0DD'},
    }

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=np.random.uniform(-12, 12, 400), y=np.random.uniform(-12, 12, 400), z=np.random.uniform(-12, 12, 400),
        mode='markers', marker=dict(size=0.8, color='white', opacity=0.35),
        showlegend=False, hoverinfo='none'
    ))

    theta = np.linspace(0, 2*np.pi, 300)
    for planet, data in planet_positions.items():
        r = data['radius']
        fig.add_trace(go.Scatter3d(
            x=r*np.cos(theta), y=r*np.sin(theta), z=np.zeros(300),
            mode='lines', line=dict(color='rgba(255,255,255,0.18)', width=1),
            showlegend=False, hoverinfo='none'
        ))
    fig.add_trace(go.Scatter3d(
        x=1.0*np.cos(theta), y=1.0*np.sin(theta), z=np.zeros(300),
        mode='lines', line=dict(color='rgba(255,255,255,0.18)', width=1),
        showlegend=False, hoverinfo='none'
    ))

    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers+text',
        marker=dict(size=22, color='#FDB813', line=dict(color='#FF6600', width=3)),
        text=['☀️ Sun'], textfont=dict(color='#FDB813', size=13),
        textposition='top center', name='☀️ Sun'
    ))

    for dr in [0.18, 0.22, 0.26]:
        fig.add_trace(go.Scatter3d(
            x=dr*np.cos(theta), y=dr*np.sin(theta), z=np.zeros(300),
            mode='lines', line=dict(color='rgba(255,215,0,0.6)', width=2),
            showlegend=dr==0.18, name='🔆 Dyson Ring', hoverinfo='none'
        ))

    fig.add_trace(go.Scatter3d(
        x=[earth_pos['x']], y=[earth_pos['y']], z=[earth_pos['z']],
        mode='markers+text',
        marker=dict(size=11, color='#2e86c1', line=dict(color='#85c1e9', width=2)),
        text=['🌍 Earth'], textfont=dict(color='#85c1e9', size=12),
        textposition='top center', name='🌍 Earth'
    ))

    shield_r = 0.05
    fig.add_trace(go.Scatter3d(
        x=earth_pos['x'] + shield_r*np.cos(theta),
        y=earth_pos['y'] + shield_r*np.sin(theta),
        z=np.zeros(300),
        mode='lines', line=dict(color='rgba(0,255,136,0.5)', width=3),
        name='🛡️ Earth Shield', hoverinfo='none'
    ))

    for name, pos in planet_positions.items():
        fig.add_trace(go.Scatter3d(
            x=[pos['x']], y=[pos['y']], z=[pos['z']],
            mode='markers+text',
            marker=dict(size=pos['size'], color=pos['color'], line=dict(color='white', width=1)),
            text=[name], textfont=dict(color='white', size=10),
            textposition='top center', name=name
        ))

    for sname, sp in station_config.items():
        fig.add_trace(go.Scatter3d(
            x=[sp['x']], y=[sp['y']], z=[sp['z']],
            mode='markers+text',
            marker=dict(size=8, color=sp['color'], symbol='diamond', line=dict(color='white', width=1)),
            text=[f"🛸 {sname.split('—')[0]}"],
            textfont=dict(color=sp['color'], size=8),
            textposition='top center', name=sname
        ))

    haz  = ast_df[ast_df['class'] != 'AMO'].head(500) if 'class' in ast_df.columns else ast_df.head(500)
    safe = ast_df.tail(500)

    fig.add_trace(go.Scatter3d(
        x=haz['ast_x'], y=haz['ast_y'], z=haz['ast_z'],
        mode='markers', marker=dict(size=2, color='#ff4444', opacity=0.7),
        name='☄️ Hazardous Asteroids',
        hovertemplate='<b>%{customdata}</b><br>Dist from Sun: %{marker.size:.2f} AU<extra></extra>',
        customdata=haz['full_name'] if 'full_name' in haz.columns else haz.index
    ))

    fig.add_trace(go.Scatter3d(
        x=safe['ast_x'], y=safe['ast_y'], z=safe['ast_z'],
        mode='markers', marker=dict(size=2, color='#00ff88', opacity=0.5),
        name='✅ Safe Asteroids',
        hovertemplate='<b>%{customdata}</b><extra></extra>',
        customdata=safe['full_name'] if 'full_name' in safe.columns else safe.index
    ))

    fig.update_layout(
        title=dict(text='AstroShield AI — Real JPL + NASA Data', font=dict(color='white', size=13)),
        paper_bgcolor='#00000f',
        scene=dict(
            bgcolor='#00000f',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showbackground=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showbackground=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showbackground=False),
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.8)),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=45, b=0),
        height=750,
        legend=dict(font=dict(color='white', size=9), bgcolor='rgba(0,0,20,0.85)', x=0.01, y=0.99)
    )

    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("Data: NASA NeoWs | JPL Horizons | SBDB Orbital Elements | NASA DONKI")
