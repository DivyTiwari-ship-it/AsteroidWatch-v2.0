import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math
import requests
from datetime import datetime, timedelta, timezone

# Load trained ML Model
model = pickle.load(open('asteroidmodel.pkl', 'rb'))

st.set_page_config(page_title="AstroShield AI", page_icon="🛸", layout="wide")
st.title("🛸 AstroShield AI")
st.subheader("Real NASA + JPL Data | Science-Backed Hazard Predictor & Space Grid")

# Master planet properties dictionary to guarantee keys are NEVER missing
PLANET_METRICS = {
    'Mercury': {'radius': 0.387, 'size': 8,  'color': '#b5b5b5'},
    'Venus':   {'radius': 0.723, 'size': 8,  'color': '#e8cda0'},
    'Mars':    {'radius': 1.524, 'size': 8,  'color': '#c1440e'},
    'Jupiter': {'radius': 5.203, 'size': 18, 'color': '#c88b3a'},
    'Saturn':  {'radius': 9.537, 'size': 15, 'color': '#e4d191'},
}

# ════════════ LIVE JPL HORIZONS DATA FETCHING ════════════
def get_planet_position(planet_id):
    url = "https://ssd.jpl.nasa.gov/api/horizons.api"
    today = datetime.now(timezone.utc)
    start = today.strftime('%Y-%m-%d')
    stop = (today + timedelta(days=1)).strftime('%Y-%m-%d')
    
    params = {
        'format': 'json',
        'COMMAND': planet_id,
        'OBJ_DATA': 'NO',
        'MAKE_EPHEM': 'YES',
        'EPHEM_TYPE': 'VECTORS',
        'CENTER': '500@10',  # Sun-centered
        'START_TIME': start,
        'STOP_TIME': stop,
        'STEP_SIZE': '1d',
        'VEC_TABLE': '2',
        'CSV_FORMAT': 'YES'
    }
    try:
        response = requests.get(url, params=params, timeout=5).json()
        return response.get('result', '')
    except:
        return ''

def parse_xyz(result_text):
    import re
    match = re.search(r'\$\$SOE(.*?)\$\$EOE', result_text, re.DOTALL)
    if not match:
        return None
    lines = match.group(1).strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 5:
            try:
                x = float(parts[2]) / 1.496e8
                y = float(parts[3]) / 1.496e8
                z = float(parts[4]) / 1.496e8
                return {'x': round(x, 4), 'y': round(y, 4), 'z': round(z, 4)}
            except:
                continue
    return None

@st.cache_data(ttl=3600)
def load_cosmic_positions():
    planets = {'Mercury': '199', 'Venus': '299', 'Mars': '499', 'Jupiter': '599', 'Saturn': '699'}
    positions = {}
    
    # Static strict fallbacks to protect against KeyError
    fallbacks = {
        'Mercury': {'x': -0.3637, 'y': 0.0705, 'z': 0.0391},
        'Venus':   {'x': -0.6892, 'y': 0.2009, 'z': 0.0425},
        'Mars':    {'x': 1.3319, 'y': 0.4757, 'z': -0.0227},
        'Jupiter': {'x': -2.7570, 'y': 4.4873, 'z': 0.0430},
        'Saturn':  {'x': 9.3972, 'y': 1.1124, 'z': -0.3934},
    }
    
    for name, pid in planets.items():
        raw_data = get_planet_position(pid)
        pos = parse_xyz(raw_data)
        if not pos:
            pos = fallbacks[name]
        
        # Inject core style fields to guarantee safety across references
        pos['radius'] = PLANET_METRICS[name]['radius']
        pos['size'] = PLANET_METRICS[name]['size']
        pos['color'] = PLANET_METRICS[name]['color']
        positions[name] = pos
        
    return positions

planet_positions = load_cosmic_positions()

# Static Coordinates Baseline Matrix
earth_pos = {'x': -0.3182, 'y': 0.9389, 'z': 0.0}
earth_angle = math.atan2(earth_pos['y'], earth_pos['x'])

# Navigation System Setup
tab1, tab2 = st.tabs(["🔍 Predict Threat", "🪐 Solar System"])

with tab1:
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

        # ════════════ LOCAL VISUAL ENGINE CLOSE-APPROACH ════════════
        st.divider()
        st.subheader("🛡️ AstroShield Close-Approach Interface")
        
        plot_dist_scaled = max(0.2, min(1.2, (miss_dist / 1000000.0) * 0.6))
        ast_size_scaled = max(6, min(24, diameter * 4))
        theme_color = '#ff4444' if pred == 1 else '#00ff88'
        shield_line_color = 'rgba(255, 68, 68, 0.8)' if pred == 1 else 'rgba(0, 255, 136, 0.6)'
        
        if pred == 1:
            traj_x = np.linspace(plot_dist_scaled * 1.5, 0.08, 50)
            traj_y = np.linspace(plot_dist_scaled * 1.5, 0.05, 50)
            traj_z = np.linspace(0.2, 0.0, 50)
            ast_x, ast_y, ast_z = plot_dist_scaled * 0.8, plot_dist_scaled * 0.8, 0.1
        else:
            traj_x = np.linspace(plot_dist_scaled * 1.5, -plot_dist_scaled, 50)
            traj_y = np.linspace(plot_dist_scaled * 1.2, plot_dist_scaled * 0.9, 50)
            traj_z = np.linspace(0.1, 0.1, 50)
            ast_x, ast_y, ast_z = plot_dist_scaled, plot_dist_scaled, 0.1

        btn_fig = go.Figure()
        
        btn_fig.add_trace(go.Scatter3d(
            x=np.random.uniform(-2, 2, 100), y=np.random.uniform(-2, 2, 100), z=np.random.uniform(-2, 2, 100),
            mode='markers', marker=dict(size=1, color='white', opacity=0.3), showlegend=False, hoverinfo='none'
        ))
        btn_fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0], mode='markers+text',
            marker=dict(size=15, color='#2e86c1', line=dict(color='#85c1e9', width=2)),
            text=['🌍 Earth'], textfont=dict(color='#85c1e9', size=11), textposition='top center', name='🌍 Earth'
        ))
        
        theta = np.linspace(0, 2*np.pi, 100)
        btn_fig.add_trace(go.Scatter3d(
            x=0.15*np.cos(theta), y=0.15*np.sin(theta), z=np.zeros(100),
            mode='lines', line=dict(color=shield_line_color, width=3 if pred==1 else 2), name='🛡️ Shield Vector'
        ))
        btn_fig.add_trace(go.Scatter3d(
            x=[ast_x], y=[ast_y], z=[ast_z], mode='markers+text',
            marker=dict(size=ast_size_scaled, color=theme_color, line=dict(color='white', width=1)),
            text=[f"☄️ Asteroid Track (V: {velocity} km/s)"], textfont=dict(color='white', size=10), textposition='top center', name='Target Asteroid'
        ))
        btn_fig.add_trace(go.Scatter3d(
            x=traj_x, y=traj_y, z=traj_z, mode='lines',
            line=dict(color='rgba(255,255,255,0.25)', width=2, dash='longdash'), name='Predicted Path Matrix', hoverinfo='none'
        ))

        btn_fig.update_layout(
            paper_bgcolor='#00000f', scene=dict(bgcolor='#00000f', xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data'),
            margin=dict(l=0, r=0, t=10, b=0), height=500, showlegend=True, legend=dict(font=dict(color='white', size=9), bgcolor='rgba(0,0,20,0.85)')
        )
        st.plotly_chart(btn_fig, use_container_width=True)
        st.info("🪐 Pure solar system space dynamics aur baki planets ko explore karne ke liye upar 'Solar System' tab par jaao!")

with tab2:
    st.caption("Live Feed Status: Connected directly to JPL Horizons Barycentric Ephemeris Service via REST vectors.")

    ast_df = pd.read_csv('asteroid_positions.csv')

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
        mode='markers', marker=dict(size=0.8, color='white', opacity=0.35), showlegend=False, hoverinfo='none'
    ))

    theta_sys = np.linspace(0, 2*np.pi, 300)
    for planet, data in planet_positions.items():
        r = data['radius']
        fig.add_trace(go.Scatter3d(
            x=r*np.cos(theta_sys), y=r*np.sin(theta_sys), z=np.zeros(300),
            mode='lines', line=dict(color='rgba(255,255,255,0.18)', width=1), showlegend=False, hoverinfo='none'
        ))
    fig.add_trace(go.Scatter3d(
        x=1.0*np.cos(theta_sys), y=1.0*np.sin(theta_sys), z=np.zeros(300),
        mode='lines', line=dict(color='rgba(255,255,255,0.18)', width=1), showlegend=False, hoverinfo='none'
    ))

    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode='markers+text',
        marker=dict(size=22, color='#FDB813', line=dict(color='#FF6600', width=3)),
        text=['☀️ Sun'], textfont=dict(color='#FDB813', size=13), textposition='top center', name='☀️ Sun'
    ))

    for dr in [0.18, 0.22, 0.26]:
        fig.add_trace(go.Scatter3d(
            x=dr*np.cos(theta_sys), y=dr*np.sin(theta_sys), z=np.zeros(300),
            mode='lines', line=dict(color='rgba(255,215,0,0.6)', width=2), showlegend=dr==0.18, name='🔆 Dyson Ring', hoverinfo='none'
        ))

    fig.add_trace(go.Scatter3d(
        x=[earth_pos['x']], y=[earth_pos['y']], z=[earth_pos['z']], mode='markers+text',
        marker=dict(size=11, color='#2e86c1', line=dict(color='#85c1e9', width=2)),
        text=['🌍 Earth'], textfont=dict(color='#85c1e9', size=12), textposition='top center', name='🌍 Earth'
    ))

    shield_r = 0.05
    fig.add_trace(go.Scatter3d(
        x=earth_pos['x'] + shield_r*np.cos(theta_sys), y=earth_pos['y'] + shield_r*np.sin(theta_sys), z=np.zeros(300),
        mode='lines', line=dict(color='rgba(0,255,136,0.5)', width=3), name='🛡️ Earth Shield', hoverinfo='none'
    ))

    # Safely rendering active planets
    for name, pos in planet_positions.items():
        fig.add_trace(go.Scatter3d(
            x=[pos['x']], y=[pos['y']], z=[pos['z']], mode='markers+text',
            marker=dict(size=pos.get('size', 8), color=pos.get('color', 'white'), line=dict(color='white', width=1)),
            text=[name], textfont=dict(color='white', size=10), textposition='top center', name=name
        ))

    for sname, sp in station_config.items():
        fig.add_trace(go.Scatter3d(
            x=[sp['x']], y=[sp['y']], z=[sp['z']], mode='markers+text',
            marker=dict(size=8, color=sp['color'], symbol='diamond', line=dict(color='white', width=1)),
            text=[f"🛸 {sname.split('—')[0]}"], textfont=dict(color=sp['color'], size=8), textposition='top center', name=sname
        ))

    haz  = ast_df[ast_df['class'] != 'AMO'].head(500) if 'class' in ast_df.columns else ast_df.head(500)
    safe = ast_df.tail(500)

    fig.add_trace(go.Scatter3d(
        x=haz['ast_x'], y=haz['ast_y'], z=haz['ast_z'], mode='markers',
        marker=dict(size=2, color='#ff4444', opacity=0.7), name='☄️ Hazardous Asteroids',
        hovertemplate='<b>%{customdata}</b><br>Dist: %{x:.2f} AU<extra></extra>', customdata=haz['full_name'] if 'full_name' in haz.columns else haz.index
    ))

    # Fixed safe coordinate columns safety check mapping
    s_x = safe['ast_x']
    s_y = safe['safe_y'] if 'safe_y' in safe.columns else (safe['ast_y'] if 'ast_y' in safe.columns else safe.iloc[:, 1])
    s_z = safe['safe_z'] if 'safe_z' in safe.columns else (safe['ast_z'] if 'ast_z' in safe.columns else safe.iloc[:, 2])

    fig.add_trace(go.Scatter3d(
        x=s_x, y=s_y, z=s_z, mode='markers',
        marker=dict(size=2, color='#00ff88', opacity=0.5), name='✅ Safe Asteroids',
        hovertemplate='<b>%{customdata}</b><extra></extra>', customdata=safe['full_name'] if 'full_name' in safe.columns else safe.index
    ))

    fig.update_layout(
        title=dict(text='AstroShield AI — Real JPL Realtime Coordinates Matrix', font=dict(color='white', size=13)),
        paper_bgcolor='#00000f',
        scene=dict(
            bgcolor='#00000f', xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.8)), aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=45, b=0), height=750, legend=dict(font=dict(color='white', size=9), bgcolor='rgba(0,0,20,0.85)', x=0.01, y=0.99)
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("Data Architecture: NASA NeoWs | JPL Horizons API REST System | NASA POWER | NOAA SWPC")        'MAKE_EPHEM': 'YES',
        'EPHEM_TYPE': 'VECTORS',
        'CENTER': '500@10',  # Sun-centered
        'START_TIME': start,
        'STOP_TIME': stop,
        'STEP_SIZE': '1d',
        'VEC_TABLE': '2',
        'CSV_FORMAT': 'YES'
    }
    try:
        response = requests.get(url, params=params, timeout=5).json()
        return response.get('result', '')
    except:
        return ''

def parse_xyz(result_text):
    import re
    match = re.search(r'\$\$SOE(.*?)\$\$EOE', result_text, re.DOTALL)
    if not match:
        return None
    lines = match.group(1).strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 5:
            try:
                x = float(parts[2]) / 1.496e8
                y = float(parts[3]) / 1.496e8
                z = float(parts[4]) / 1.496e8
                return {'x': round(x, 4), 'y': round(y, 4), 'z': round(z, 4)}
            except:
                continue
    return None

@st.cache_data(ttl=3600)
def load_cosmic_positions():
    planets = {'Mercury': '199', 'Venus': '299', 'Mars': '499', 'Jupiter': '599', 'Saturn': '699'}
    positions = {}
    
    # Static strict fallbacks to protect against KeyError
    fallbacks = {
        'Mercury': {'x': -0.3637, 'y': 0.0705, 'z': 0.0391},
        'Venus':   {'x': -0.6892, 'y': 0.2009, 'z': 0.0425},
        'Mars':    {'x': 1.3319, 'y': 0.4757, 'z': -0.0227},
        'Jupiter': {'x': -2.7570, 'y': 4.4873, 'z': 0.0430},
        'Saturn':  {'x': 9.3972, 'y': 1.1124, 'z': -0.3934},
    }
    
    for name, pid in planets.items():
        raw_data = get_planet_position(pid)
        pos = parse_xyz(raw_data)
        if not pos:
            pos = fallbacks[name]
        
        # Inject core style fields to guarantee safety across references
        pos['radius'] = PLANET_METRICS[name]['radius']
        pos['size'] = PLANET_METRICS[name]['size']
        pos['color'] = PLANET_METRICS[name]['color']
        positions[name] = pos
        
    return positions

planet_positions = load_cosmic_positions()

# Static Coordinates Baseline Matrix
earth_pos = {'x': -0.3182, 'y': 0.9389, 'z': 0.0}
earth_angle = math.atan2(earth_pos['y'], earth_pos['x'])

# Navigation System Setup
tab1, tab2 = st.tabs(["🔍 Predict Threat", "🪐 Solar System"])

with tab1:
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

        # ════════════ LOCAL VISUAL ENGINE CLOSE-APPROACH ════════════
        st.divider()
        st.subheader("🛡️ AstroShield Close-Approach Interface")
        
        plot_dist_scaled = max(0.2, min(1.2, (miss_dist / 1000000.0) * 0.6))
        ast_size_scaled = max(6, min(24, diameter * 4))
        theme_color = '#ff4444' if pred == 1 else '#00ff88'
        shield_line_color = 'rgba(255, 68, 68, 0.8)' if pred == 1 else 'rgba(0, 255, 136, 0.6)'
        
        if pred == 1:
            traj_x = np.linspace(plot_dist_scaled * 1.5, 0.08, 50)
            traj_y = np.linspace(plot_dist_scaled * 1.5, 0.05, 50)
            traj_z = np.linspace(0.2, 0.0, 50)
            ast_x, ast_y, ast_z = plot_dist_scaled * 0.8, plot_dist_scaled * 0.8, 0.1
        else:
            traj_x = np.linspace(plot_dist_scaled * 1.5, -plot_dist_scaled, 50)
            traj_y = np.linspace(plot_dist_scaled * 1.2, plot_dist_scaled * 0.9, 50)
            traj_z = np.linspace(0.1, 0.1, 50)
            ast_x, ast_y, ast_z = plot_dist_scaled, plot_dist_scaled, 0.1

        btn_fig = go.Figure()
        
        btn_fig.add_trace(go.Scatter3d(
            x=np.random.uniform(-2, 2, 100), y=np.random.uniform(-2, 2, 100), z=np.random.uniform(-2, 2, 100),
            mode='markers', marker=dict(size=1, color='white', opacity=0.3), showlegend=False, hoverinfo='none'
        ))
        btn_fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0], mode='markers+text',
            marker=dict(size=15, color='#2e86c1', line=dict(color='#85c1e9', width=2)),
            text=['🌍 Earth'], textfont=dict(color='#85c1e9', size=11), textposition='top center', name='🌍 Earth'
        ))
        
        theta = np.linspace(0, 2*np.pi, 100)
        btn_fig.add_trace(go.Scatter3d(
            x=0.15*np.cos(theta), y=0.15*np.sin(theta), z=np.zeros(100),
            mode='lines', line=dict(color=shield_line_color, width=3 if pred==1 else 2), name='🛡️ Shield Vector'
        ))
        btn_fig.add_trace(go.Scatter3d(
            x=[ast_x], y=[ast_y], z=[ast_z], mode='markers+text',
            marker=dict(size=ast_size_scaled, color=theme_color, line=dict(color='white', width=1)),
            text=[f"☄️ Asteroid Track (V: {velocity} km/s)"], textfont=dict(color='white', size=10), textposition='top center', name='Target Asteroid'
        ))
        btn_fig.add_trace(go.Scatter3d(
            x=traj_x, y=traj_y, z=traj_z, mode='lines',
            line=dict(color='rgba(255,255,255,0.25)', width=2, dash='longdash'), name='Predicted Path Matrix', hoverinfo='none'
        ))

        btn_fig.update_layout(
            paper_bgcolor='#00000f', scene=dict(bgcolor='#00000f', xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data'),
            margin=dict(l=0, r=0, t=10, b=0), height=500, showlegend=True, legend=dict(font=dict(color='white', size=9), bgcolor='rgba(0,0,20,0.85)')
        )
        st.plotly_chart(btn_fig, use_container_width=True)
        st.info("🪐 Pure solar system space dynamics aur baki planets ko explore karne ke liye upar 'Solar System' tab par jaao!")

with tab2:
    st.caption("Live Feed Status: Connected directly to JPL Horizons Barycentric Ephemeris Service via REST vectors.")

    ast_df = pd.read_csv('asteroid_positions.csv')

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
        mode='markers', marker=dict(size=0.8, color='white', opacity=0.35), showlegend=False, hoverinfo='none'
    ))

    theta_sys = np.linspace(0, 2*np.pi, 300)
    for planet, data in planet_positions.items():
        r = data['radius']
        fig.add_trace(go.Scatter3d(
            x=r*np.cos(theta_sys), y=r*np.sin(theta_sys), z=np.zeros(300),
            mode='lines', line=dict(color='rgba(255,255,255,0.18)', width=1), showlegend=False, hoverinfo='none'
        ))
    fig.add_trace(go.Scatter3d(
        x=1.0*np.cos(theta_sys), y=1.0*np.sin(theta_sys), z=np.zeros(300),
        mode='lines', line=dict(color='rgba(255,255,255,0.18)', width=1), showlegend=False, hoverinfo='none'
    ))

    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode='markers+text',
        marker=dict(size=22, color='#FDB813', line=dict(color='#FF6600', width=3)),
        text=['☀️ Sun'], textfont=dict(color='#FDB813', size=13), textposition='top center', name='☀️ Sun'
    ))

    for dr in [0.18, 0.22, 0.26]:
        fig.add_trace(go.Scatter3d(
            x=dr*np.cos(theta_sys), y=dr*np.sin(theta_sys), z=np.zeros(300),
            mode='lines', line=dict(color='rgba(255,215,0,0.6)', width=2), showlegend=dr==0.18, name='🔆 Dyson Ring', hoverinfo='none'
        ))

    fig.add_trace(go.Scatter3d(
        x=[earth_pos['x']], y=[earth_pos['y']], z=[earth_pos['z']], mode='markers+text',
        marker=dict(size=11, color='#2e86c1', line=dict(color='#85c1e9', width=2)),
        text=['🌍 Earth'], textfont=dict(color='#85c1e9', size=12), textposition='top center', name='🌍 Earth'
    ))

    shield_r = 0.05
    fig.add_trace(go.Scatter3d(
        x=earth_pos['x'] + shield_r*np.cos(theta_sys), y=earth_pos['y'] + shield_r*np.sin(theta_sys), z=np.zeros(300),
        mode='lines', line=dict(color='rgba(0,255,136,0.5)', width=3), name='🛡️ Earth Shield', hoverinfo='none'
    ))

    # Safely rendering active planets
    for name, pos in planet_positions.items():
        fig.add_trace(go.Scatter3d(
            x=[pos['x']], y=[pos['y']], z=[pos['z']], mode='markers+text',
            marker=dict(size=pos.get('size', 8), color=pos.get('color', 'white'), line=dict(color='white', width=1)),
            text=[name], textfont=dict(color='white', size=10), textposition='top center', name=name
        ))

    for sname, sp in station_config.items():
        fig.add_trace(go.Scatter3d(
            x=[sp['x']], y=[sp['y']], z=[sp['z']], mode='markers+text',
            marker=dict(size=8, color=sp['color'], symbol='diamond', line=dict(color='white', width=1)),
            text=[f"🛸 {sname.split('—')[0]}"], textfont=dict(color=sp['color'], size=8), textposition='top center', name=sname
        ))

    haz  = ast_df[ast_df['class'] != 'AMO'].head(500) if 'class' in ast_df.columns else ast_df.head(500)
    safe = ast_df.tail(500)

    fig.add_trace(go.Scatter3d(
        x=haz['ast_x'], y=haz['ast_y'], z=haz['ast_z'], mode='markers',
        marker=dict(size=2, color='#ff4444', opacity=0.7), name='☄️ Hazardous Asteroids',
        hovertemplate='<b>%{customdata}</b><br>Dist: %{x:.2f} AU<extra></extra>', customdata=haz['full_name'] if 'full_name' in haz.columns else haz.index
    ))

    # Fixed safe coordinate columns safety check mapping
    s_x = safe['ast_x']
    s_y = safe['safe_y'] if 'safe_y' in safe.columns else (safe['ast_y'] if 'ast_y' in safe.columns else safe.iloc[:, 1])
    s_z = safe['safe_z'] if 'safe_z' in safe.columns else (safe['ast_z'] if 'ast_z' in safe.columns else safe.iloc[:, 2])

    fig.add_trace(go.Scatter3d(
        x=s_x, y=s_y, z=s_z, mode='markers',
        marker=dict(size=2, color='#00ff88', opacity=0.5), name='✅ Safe Asteroids',
        hovertemplate='<b>%{customdata}</b><extra></extra>', customdata=safe['full_name'] if 'full_name' in safe.columns else safe.index
    ))

    fig.update_layout(
        title=dict(text='AstroShield AI — Real JPL Realtime Coordinates Matrix', font=dict(color='white', size=13)),
        paper_bgcolor='#00000f',
        scene=dict(
            bgcolor='#00000f', xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.8)), aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=45, b=0), height=750, legend=dict(font=dict(color='white', size=9), bgcolor='rgba(0,0,20,0.85)', x=0.01, y=0.99)
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("Data Architecture: NASA NeoWs | JPL Horizons API REST System | NASA POWER | NOAA SWPC")    try:
        response = requests.get(url, params=params).json()
        return response.get('result', '')
    except:
        return ''

def parse_xyz(result_text):
    import re
    match = re.search(r'\$\$SOE(.*?)\$\$EOE', result_text, re.DOTALL)
    if not match:
        return None
    lines = match.group(1).strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 5:
            try:
                # JPL returns data in KM. Converting to Astronomical Units (AU)
                x = float(parts[2]) / 1.496e8
                y = float(parts[3]) / 1.496e8
                z = float(parts[4]) / 1.496e8
                return {'x': round(x, 4), 'y': round(y, 4), 'z': round(z, 4)}
            except:
                continue
    return None

# Fetch Real-Time Coordinates via JPL API
@st.cache_data(ttl=3600)  # Cache data for 1 hour to prevent API spamming
def load_cosmic_positions():
    planets = {'Mercury': '199', 'Venus': '299', 'Mars': '499', 'Jupiter': '599', 'Saturn': '699'}
    positions = {}
    for name, pid in planets.items():
        raw_data = get_planet_position(pid)
        pos = parse_xyz(raw_data)
        if pos:
            positions[name] = pos
        else:
            # Fallback coordinates if API limits out
            fallbacks = {
                'Mercury': {'x': -0.3637, 'y': 0.0705, 'z': 0.0391},
                'Venus':   {'x': -0.6892, 'y': 0.2009, 'z': 0.0425},
                'Mars':    {'x': 1.3319, 'y': 0.4757, 'z': -0.0227},
                'Jupiter': {'x': -2.7570, 'y': 4.4873, 'z': 0.0430},
                'Saturn':  {'x': 9.3972, 'y': 1.1124, 'z': -0.3934},
            }
            positions[name] = fallbacks[name]
    return positions

planet_positions = load_cosmic_positions()
# Standard baseline values for radii metrics
radii_mapping = {'Mercury': 0.387, 'Venus': 0.723, 'Mars': 1.524, 'Jupiter': 5.203, 'Saturn': 9.537}
for p in planet_positions:
    planet_positions[p]['radius'] = radii_mapping[p]
    planet_positions[p]['size'] = 18 if p == 'Jupiter' else (15 if p == 'Saturn' else 8)

# Fixed Earth Real Coordinate Array (Barycentric Baseline)
earth_pos = {'x': -0.3182, 'y': 0.9389, 'z': 0.0}
earth_angle = math.atan2(earth_pos['y'], earth_pos['x'])

# ════════════ APP NAVIGATION TABS ════════════
tab1, tab2 = st.tabs(["🔍 Predict Threat", "🪐 Solar System"])

with tab1:
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

        # ════════════ LOCAL CLOSE-APPROACH SYSTEM VIEW ════════════
        st.divider()
        st.subheader("🛡️ AstroShield Close-Approach Interface")
        
        plot_dist_scaled = max(0.2, min(1.2, (miss_dist / 1000000.0) * 0.6))
        ast_size_scaled = max(6, min(24, diameter * 4))
        theme_color = '#ff4444' if pred == 1 else '#00ff88'
        shield_line_color = 'rgba(255, 68, 68, 0.8)' if pred == 1 else 'rgba(0, 255, 136, 0.6)'
        
        if pred == 1:
            traj_x = np.linspace(plot_dist_scaled * 1.5, 0.08, 50)
            traj_y = np.linspace(plot_dist_scaled * 1.5, 0.05, 50)
            traj_z = np.linspace(0.2, 0.0, 50)
            ast_x, ast_y, ast_z = plot_dist_scaled * 0.8, plot_dist_scaled * 0.8, 0.1
        else:
            traj_x = np.linspace(plot_dist_scaled * 1.5, -plot_dist_scaled, 50)
            traj_y = np.linspace(plot_dist_scaled * 1.2, plot_dist_scaled * 0.9, 50)
            traj_z = np.linspace(0.1, 0.1, 50)
            ast_x, ast_y, ast_z = plot_dist_scaled, plot_dist_scaled, 0.1

        btn_fig = go.Figure()
        
        # Space Stars Base
        btn_fig.add_trace(go.Scatter3d(
            x=np.random.uniform(-2, 2, 100), y=np.random.uniform(-2, 2, 100), z=np.random.uniform(-2, 2, 100),
            mode='markers', marker=dict(size=1, color='white', opacity=0.3), showlegend=False, hoverinfo='none'
        ))
        # Central Target Earth
        btn_fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0], mode='markers+text',
            marker=dict(size=15, color='#2e86c1', line=dict(color='#85c1e9', width=2)),
            text=['🌍 Earth'], textfont=dict(color='#85c1e9', size=11), textposition='top center', name='🌍 Earth'
        ))
        # Atmospheric Defense Ring Line
        theta = np.linspace(0, 2*np.pi, 100)
        btn_fig.add_trace(go.Scatter3d(
            x=0.15*np.cos(theta), y=0.15*np.sin(theta), z=np.zeros(100),
            mode='lines', line=dict(color=shield_line_color, width=3 if pred==1 else 2), name='🛡️ Shield Vector'
        ))
        # Active Asteroid Render Instance
        btn_fig.add_trace(go.Scatter3d(
            x=[ast_x], y=[ast_y], z=[ast_z], mode='markers+text',
            marker=dict(size=ast_size_scaled, color=theme_color, line=dict(color='white', width=1)),
            text=[f"☄️ Asteroid Track (V: {velocity} km/s)"], textfont=dict(color='white', size=10), textposition='top center', name='Target Asteroid'
        ))
        # Projected Trajectory Intercept Matrice
        btn_fig.add_trace(go.Scatter3d(
            x=traj_x, y=traj_y, z=traj_z, mode='lines',
            line=dict(color='rgba(255,255,255,0.25)', width=2, dash='longdash'), name='Predicted Path Matrix', hoverinfo='none'
        ))

        btn_fig.update_layout(
            paper_bgcolor='#00000f', scene=dict(bgcolor='#00000f', xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data'),
            margin=dict(l=0, r=0, t=10, b=0), height=500, showlegend=True, legend=dict(font=dict(color='white', size=9), bgcolor='rgba(0,0,20,0.85)')
        )
        st.plotly_chart(btn_fig, use_container_width=True)

        st.info("🪐 Pure solar system space dynamics aur baki planets ko explore karne ke liye upar 'Solar System' tab par jaao!")

with tab2:
    st.caption("Live Feed Status: Connected directly to JPL Horizons Barycentric Ephemeris Service via REST vectors.")

    ast_df = pd.read_csv('asteroid_positions.csv')

    # Advanced Orbital Structure System Setup
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

    # Ambient Background Stars
    fig.add_trace(go.Scatter3d(
        x=np.random.uniform(-12, 12, 400), y=np.random.uniform(-12, 12, 400), z=np.random.uniform(-12, 12, 400),
        mode='markers', marker=dict(size=0.8, color='white', opacity=0.35), showlegend=False, hoverinfo='none'
    ))

    theta_sys = np.linspace(0, 2*np.pi, 300)
    for planet, data in planet_positions.items():
        r = data['radius']
        fig.add_trace(go.Scatter3d(
            x=r*np.cos(theta_sys), y=r*np.sin(theta_sys), z=np.zeros(300),
            mode='lines', line=dict(color='rgba(255,255,255,0.18)', width=1), showlegend=False, hoverinfo='none'
        ))
    fig.add_trace(go.Scatter3d(
        x=1.0*np.cos(theta_sys), y=1.0*np.sin(theta_sys), z=np.zeros(300),
        mode='lines', line=dict(color='rgba(255,255,255,0.18)', width=1), showlegend=False, hoverinfo='none'
    ))

    # Sun Node
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode='markers+text',
        marker=dict(size=22, color='#FDB813', line=dict(color='#FF6600', width=3)),
        text=['☀️ Sun'], textfont=dict(color='#FDB813', size=13), textposition='top center', name='☀️ Sun'
    ))

    # Dyson Rings Wireframes
    for dr in [0.18, 0.22, 0.26]:
        fig.add_trace(go.Scatter3d(
            x=dr*np.cos(theta_sys), y=dr*np.sin(theta_sys), z=np.zeros(300),
            mode='lines', line=dict(color='rgba(255,215,0,0.6)', width=2), showlegend=dr==0.18, name='🔆 Dyson Ring', hoverinfo='none'
        ))

    # Earth Node
    fig.add_trace(go.Scatter3d(
        x=[earth_pos['x']], y=[earth_pos['y']], z=[earth_pos['z']], mode='markers+text',
        marker=dict(size=11, color='#2e86c1', line=dict(color='#85c1e9', width=2)),
        text=['🌍 Earth'], textfont=dict(color='#85c1e9', size=12), textposition='top center', name='🌍 Earth'
    ))

    # Planetary Defense Grid Ring
    shield_r = 0.05
    fig.add_trace(go.Scatter3d(
        x=earth_pos['x'] + shield_r*np.cos(theta_sys), y=earth_pos['y'] + shield_r*np.sin(theta_sys), z=np.zeros(300),
        mode='lines', line=dict(color='rgba(0,255,136,0.5)', width=3), name='🛡️ Earth Shield', hoverinfo='none'
    ))

    # Render Active Live Planets
    for name, pos in planet_positions.items():
        fig.add_trace(go.Scatter3d(
            x=[pos['x']], y=[pos['y']], z=[pos['z']], mode='markers+text',
            marker=dict(size=pos['size'], color=pos['color'], line=dict(color='white', width=1)),
            text=[name], textfont=dict(color='white', size=10), textposition='top center', name=name
        ))

    # Render Infrastructure Space Stations
    for sname, sp in station_config.items():
        fig.add_trace(go.Scatter3d(
            x=[sp['x']], y=[sp['y']], z=[sp['z']], mode='markers+text',
            marker=dict(size=8, color=sp['color'], symbol='diamond', line=dict(color='white', width=1)),
            text=[f"🛸 {sname.split('—')[0]}"], textfont=dict(color=sp['color'], size=8), textposition='top center', name=sname
        ))

    # Asteroid Database Arrays
    haz  = ast_df[ast_df['class'] != 'AMO'].head(500) if 'class' in ast_df.columns else ast_df.head(500)
    safe = ast_df.tail(500)

    fig.add_trace(go.Scatter3d(
        x=haz['ast_x'], y=haz['ast_y'], z=haz['ast_z'], mode='markers',
        marker=dict(size=2, color='#ff4444', opacity=0.7), name='☄️ Hazardous Asteroids',
        hovertemplate='<b>%{customdata}</b><br>Dist: %{x:.2f} AU<extra></extra>', customdata=haz['full_name'] if 'full_name' in haz.columns else haz.index
    ))

    fig.add_trace(go.Scatter3d(
        x=safe['ast_x'], y=safe['safe_y'] if 'safe_y' in ast_df.columns else safe['ast_y'], z=safe['safe_z'] if 'safe_z' in ast_df.columns else safe['ast_z'], mode='markers',
        marker=dict(size=2, color='#00ff88', opacity=0.5), name='✅ Safe Asteroids',
        hovertemplate='<b>%{customdata}</b><extra></extra>', customdata=safe['full_name'] if 'full_name' in safe.columns else safe.index
    ))

    fig.update_layout(
        title=dict(text='AstroShield AI — Real JPL Realtime Coordinates Matrix', font=dict(color='white', size=13)),
        paper_bgcolor='#00000f',
        scene=dict(
            bgcolor='#00000f', xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.8)), aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=45, b=0), height=750, legend=dict(font=dict(color='white', size=9), bgcolor='rgba(0,0,20,0.85)', x=0.01, y=0.99)
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("Data Architecture: NASA NeoWs | JPL Horizons API REST System | NASA POWER | NOAA SWPC")
