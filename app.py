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

html, body, [data-testid="stAppViewContainer"] {
    background-color: #03030c !important;
    font-family: 'Inter', sans-serif;
}

.main-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 3.2rem !important;
    font-weight: 900 !important;
    background: linear-gradient(45deg, #00f2fe, #4facfe, #9b51e0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 30px rgba(0, 242, 254, 0.25);
    margin-bottom: 0px;
}

.sub-title {
    color: #7f8c8d !important;
    font-size: 1.05rem !important;
    letter-spacing: 2px;
    margin-bottom: 0.8rem;
    text-transform: uppercase;
}

.dev-badge {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.85rem;
    color: #00f2fe;
    letter-spacing: 1px;
    margin-bottom: 2.5rem;
    background: rgba(0, 242, 254, 0.05);
    border: 1px solid rgba(0, 242, 254, 0.2);
    display: inline-block;
    padding: 6px 16px;
    border-radius: 6px;
    box-shadow: 0 0 10px rgba(0, 242, 254, 0.1);
}

div.stButton > button:first-child {
    font-family: 'Orbitron', sans-serif !important;
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%) !important;
    color: #03030c !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 15px 24px !important;
    box-shadow: 0 0 15px rgba(0, 242, 254, 0.4) !important;
    transition: all 0.3s ease-in-out !important;
}

div.stButton > button:first-child:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 0 25px rgba(0, 242, 254, 0.8) !important;
    color: #000000 !important;
}

.hud-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(0, 242, 254, 0.2);
    border-radius: 8px;
    padding: 12px;
    text-align: center;
    box-shadow: inset 0 0 10px rgba(0, 242, 254, 0.05);
}

.hud-label {
    font-size: 0.75rem;
    color: #7f8c8d;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.hud-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.2rem;
    color: #00f2fe;
    font-weight: 700;
}

.hazard-card {
    background: linear-gradient(135deg, rgba(255, 68, 68, 0.15) 0%, rgba(15, 3, 3, 0.95) 100%);
    border: 2px solid #ff4444;
    border-radius: 12px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 0 30px rgba(255, 68, 68, 0.25);
}

.safe-card {
    background: linear-gradient(135deg, rgba(0, 255, 136, 0.15) 0%, rgba(3, 15, 8, 0.95) 100%);
    border: 2px solid #00ff88;
    border-radius: 12px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 0 30px rgba(0, 255, 136, 0.25);
}

.card-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 8px;
}

.metrics-val {
    font-family: 'Orbitron', sans-serif;
    font-size: 3.5rem;
    font-weight: 900;
    margin: 15px 0px;
    text-shadow: 0 0 20px rgba(255,255,255,0.2);
}

.metrics-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'Inter', sans-serif;
    color: #e2e8f0;
}

.metrics-table th {
    font-family: 'Orbitron', sans-serif;
    color: #00f2fe;
    text-align: left;
    padding: 8px;
    border-bottom: 1px solid rgba(0, 242, 254, 0.2);
    font-size: 0.85rem;
}

.metrics-table td {
    padding: 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    font-size: 0.9rem;
}

button[data-baseweb="tab"] {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 1.05rem !important;
    letter-spacing: 1px;
}
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

def get_planet_position(planet_id):
    url = "https://ssd.jpl.nasa.gov/api/horizons.api"
    today = datetime.now(timezone.utc)
    start = today.strftime('%Y-%m-%d')
    stop = (today + timedelta(days=1)).strftime('%Y-%m-%d')
    
    params = {
        'format': 'json', 'COMMAND': planet_id, 'OBJ_DATA': 'NO',
        'MAKE_EPHEM': 'YES', 'EPHEM_TYPE': 'VECTORS', 'CENTER': '500@10',
        'START_TIME': start, 'STOP_TIME': stop, 'STEP_SIZE': '1d',
        'VEC_TABLE': '2', 'CSV_FORMAT': 'YES'
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
            pos = fallbacks[name].copy()
        
        pos['radius'] = PLANET_METRICS[name]['radius']
        pos['size'] = PLANET_METRICS[name]['size']
        pos['color'] = PLANET_METRICS[name]['color']
        positions[name] = pos
        
    return positions

planet_positions = load_cosmic_positions()
earth_pos = {'x': -0.3182, 'y': 0.9389, 'z': 0.0}
earth_angle = math.atan2(earth_pos['y'], earth_pos['x'])

tab1, tab2 = st.tabs(["🔍 Predict Threat Matrix", "🪐 Solar System Grid"])

with tab1:
    st.markdown("<p style='font-family:\"Orbitron\"; color: #00f2fe; font-size: 1.2rem; letter-spacing: 2px; margin-bottom: 5px;'>🛰️ DEEP SPACE STATION TELEMETRY</p>", unsafe_allow_html=True)
    
    s1, s2, s3, s4 = st.columns(4)
    s1.markdown('<div class="hud-card"><div class="hud-label">ISS (Earth Orbit)</div><div class="hud-value" style="color:#00ff88;">ONLINE (Stable)</div><div style="font-size:0.75rem; color:#7f8c8d;">Alt: 420km</div></div>', unsafe_allow_html=True)
    s2.markdown('<div class="hud-card"><div class="hud-label">Ares Alpha (Mars)</div><div class="hud-value" style="color:#00ff88;">ONLINE (Active)</div><div style="font-size:0.75rem; color:#7f8c8d;">Dist: 1.52 AU</div></div>', unsafe_allow_html=True)
    s3.markdown('<div class="hud-card"><div class="hud-label">Hermes Hub (Venus)</div><div class="hud-value" style="color:#f1c40f;">MAINTENANCE</div><div style="font-size:0.75rem; color:#7f8c8d;">Dist: 0.72 AU</div></div>', unsafe_allow_html=True)
    s4.markdown('<div class="hud-card"><div class="hud-label">Chronos V (Saturn)</div><div class="hud-value" style="color:#ff4444;">ALERT (Solar Flare)</div><div style="font-size:0.75rem; color:#7f8c8d;">Dist: 9.54 AU</div></div>', unsafe_allow_html=True)

    st.markdown("<br><hr style='border-color: rgba(0, 242, 254, 0.1);'>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1.1, 1.9], gap="large")
    
    with col_l:
        st.markdown("<p style='font-family:\"Orbitron\"; color: #00f2fe; font-size: 1rem;'>🎛️ ORBITAL PARAMETERS</p>", unsafe_allow_html=True)
        with st.container(border=True):
            magnitude = st.slider("Absolute Magnitude (H)", 10.0, 35.0, 20.0)
            diameter  = st.slider("Diameter (km)", 0.001, 10.0, 0.5)
            velocity  = st.slider("Velocity (km/s)", 1.0, 40.0, 15.0)
            miss_dist = st.number_input("Miss Distance (km)", value=500000.0, step=50000.0)
            predict_btn = st.button("⚡ EXECUTE TRACKING", use_container_width=True)

    with col_r:
        st.markdown("<p style='font-family:\"Orbitron\"; color: #00f2fe; font-size: 1rem;'>📊 THREAT ASSESSMENT INTERCEPT</p>", unsafe_allow_html=True)
        
        threat = velocity / miss_dist
        features = np.array([[magnitude, diameter*0.8, diameter*1.2, miss_dist, velocity, diameter, threat, diameter*velocity]])
        pred = model.predict(features)[0]
        
        out1, out2 = st.columns([1.2, 1.8])
        with out1:
            if pred == 1:
                st.markdown('<div class="hazard-card"><div style="font-size:1.1rem; font-weight:bold; color:#ff4444; letter-spacing:1px;">⚠️ DANGER ZONE</div><div style="font-size:1.8rem; font-family:\'Orbitron\'; color:#ff4444;">HAZARD DETECTED</div><p style="font-size:0.8rem; color:#e2e8f0; margin-top:10px;">Asteroid trajectory poses high risk to Earth and orbital Space Stations.</p></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="safe-card"><div style="font-size:1.1rem; font-weight:bold; color:#00ff88; letter-spacing:1px;">✅ SAFE ZONE</div><div style="font-size:1.8rem; font-family:\'Orbitron\'; color:#00ff88;">STABLE ORBIT</div><p style="font-size:0.8rem; color:#e2e8f0; margin-top:10px;">Trajectory is safe. Planetary defense shields require no intercept protocols.</p></div>', unsafe_allow_html=True)
        
        with out2:
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers+text', marker=dict(size=14, color='#2e86c1'), text=['Earth'], name='Earth'))
            fig.add_trace(go.Scatter3d(x=[0.1], y=[0.1], z=[0.05], mode='markers+text', marker=dict(size=6, color='#00ff88', symbol='diamond'), text=['ISS Station'], name='Space Station'))
            
            ast_color = '#ff4444' if pred == 1 else '#00ff88'
            fig.add_trace(go.Scatter3d(x=[0.4], y=[0.5], z=[0.2], mode='markers+text', marker=dict(size=9, color=ast_color), text=['Target Asteroid'], name='Asteroid'))
            fig.add_trace(go.Scatter3d(x=[0.8, 0.4, 0], y=[0.9, 0.5, 0], z=[0.4, 0.2, 0], mode='lines', line=dict(color='rgba(255,255,255,0.2)', width=2, dash='dash'), name='Trajectory Path'))
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)), 
                margin=dict(l=0,r=0,t
with tab2:
    st.caption("Live Feed Status: Connected to JPL Horizons.")

    ast_df = pd.read_csv('asteroid_positions.csv')

    station_config = {
        'Mercury L1': {'x': planet_positions['Mercury']['x'] * 0.85, 'y': planet_positions['Mercury']['y'] * 0.85, 'z': 0, 'color': '#FFD700'},
        'Venus L2':   {'x': planet_positions['Venus']['x'] * 1.15,   'y': planet_positions['Venus']['y'] * 1.15,   'z': 0, 'color': '#FFA07A'},
        'Earth L4':   {'x': 1.0 * math.cos(earth_angle + math.radians(60)), 'y': 1.0 * math.sin(earth_angle + math.radians(60)), 'z': 0, 'color': '#00FFFF'},
        'Earth L5':   {'x': 1.0 * math.cos(earth_angle - math.radians(60)), 'y': 1.0 * math.sin(earth_angle - math.radians(60)), 'z': 0, 'color': '#FF00FF'},
        'Mars Phobos': {'x': planet_positions['Mars']['x'] * 1.02,    'y': planet_positions['Mars']['y'] * 1.02,    'z': 0.01, 'color': '#FF6600'},
        'Jupiter L4': {'x': 5.203 * math.cos(math.atan2(planet_positions['Jupiter']['y'], planet_positions['Jupiter']['x']) + math.radians(60)), 'y': 5.203 * math.sin(math.atan2(planet_positions['Jupiter']['y'], planet_positions['Jupiter']['x']) + math.radians(60)), 'z': 0, 'color': '#98FB98'},
        'Titan Orbit': {'x': planet_positions['Saturn']['x'] * 1.008, 'y': planet_positions['Saturn']['y'] * 1.008, 'z': 0.05, 'color': '#DDA0DD'},
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
            text=[f"🛸 {sname}"], textfont=dict(color=sp['color'], size=8), textposition='top center', name=sname
        ))

    haz = ast_df[ast_df['class'] != 'AMO'].head(500) if 'class' in ast_df.columns else ast_df.head(500)
    safe = ast_df.tail(500)

    fig.add_trace(go.Scatter3d(
        x=haz['ast_x'], y=haz['ast_y'], z=haz['ast_z'], mode='markers',
        marker=dict(size=2, color='#ff4444', opacity=0.7), name='☄️ Hazardous Asteroids',
        hovertemplate='<b>%{customdata}</b><br>Dist: %{x:.2f} AU<extra></extra>', customdata=haz['full_name'] if 'full_name' in haz.columns else haz.index
    ))

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
st.caption("Data Architecture: NASA NeoWs | JPL Horizons API REST System")
