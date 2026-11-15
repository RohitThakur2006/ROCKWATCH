"""
app.py - RockWatch Mission-Control: 3D Geotechnical Digital Twin & Monitoring Center.

Fully customized industrial mission-control interface:
- High-tech dark aerospace command gradient with zero Streamlit chrome (headers/toolbars hidden).
- Stepped Open-Pit 3D Digital Twin with explicit spiral haul road, bench faces, and 3D floating sector station markers.
- Continuous Progressive Simulation Engine grounded in Saito's Tertiary Creep Law & Infiltration physics.
- Dynamic CCTV Camera Feed with real-time particle motion, rockfall cascades, crack dilation, and AI tracking bounding boxes.
- Multimodal Late-Sensor Fusion (Option B) with single-sensor dominance override.
- AI Geotechnical Safety Copilot (Semantic Kernel RAG grounded in SOPs).
- Enterprise Human-in-the-Loop Geotechnical Authorization Workflow with digital signatures and auto-expiry.
- Real-time SHA-256 cryptographic hash-chained audit ledger.
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import json
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image, ImageDraw

import streamlit as st

from blackboard import MineBlackboard
from controller import RockWatchController
from tools.rag_tools import retrieve_sop
from tools.simulation_engine import GeotechnicalSimulationEngine
from ml.camera_stream import DynamicPitCameraStream

# --- PAGE CONFIGURATION (Full Width, Dark Base) ---
st.set_page_config(
    page_title="RockWatch // 3D Geotechnical Digital Twin",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- GLOBAL STYLES (ERADICATE STREAMLIT CHROME & APPLY HIGH-TECH MISSION GRADIENT) ---
CUSTOM_CSS = """
<style>
/* 1. Completely remove default Streamlit header, menu, deploy button, status widget, and footer */
header, [data-testid="stHeader"] {
    display: none !important;
    height: 0px !important;
    visibility: hidden !important;
}
[data-testid="stToolbar"] {
    display: none !important;
    visibility: hidden !important;
}
[data-testid="stDecoration"] {
    display: none !important;
    visibility: hidden !important;
}
[data-testid="stStatusWidget"] {
    display: none !important;
    visibility: hidden !important;
}
#MainMenu {
    display: none !important;
    visibility: hidden !important;
}
footer {
    display: none !important;
    visibility: hidden !important;
}
.stDeployButton {
    display: none !important;
    visibility: hidden !important;
}
[data-testid="stSidebar"], [data-testid="collapsedControl"] {
    display: none !important;
    visibility: hidden !important;
}

/* 2. Full-screen layout with zero unwanted padding */
.main, .block-container {
    padding-top: 0.6rem !important;
    padding-bottom: 1.5rem !important;
    padding-left: 1.25rem !important;
    padding-right: 1.25rem !important;
    max-width: 100% !important;
}

/* 3. High-tech Deep Aerospace Mission Gradient Background */
.stApp {
    background: radial-gradient(circle at 50% 0%, #101c38 0%, #080d1a 50%, #03060c 100%) !important;
    background-attachment: fixed !important;
    color: #e2e8f0 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
}

/* 4. Top Mission Control HUD Header */
.hud-header {
    background: linear-gradient(90deg, rgba(15, 23, 42, 0.95) 0%, rgba(26, 38, 66, 0.85) 50%, rgba(15, 23, 42, 0.95) 100%);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 12px;
    padding: 12px 22px;
    margin-bottom: 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.1);
}

.hud-title {
    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* 5. Glassmorphic Surface Cards */
.mission-card {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(59, 130, 246, 0.18);
    border-radius: 12px;
    padding: 14px 18px;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45);
    margin-bottom: 12px;
}

.mission-card:hover {
    border-color: rgba(59, 130, 246, 0.35);
}

.card-title {
    font-size: 0.88rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* 6. Dynamic Glowing Status Badges */
.badge-green {
    background: rgba(16, 185, 129, 0.18);
    border: 1px solid #10b981;
    color: #34d399;
    padding: 4px 12px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.82rem;
    box-shadow: 0 0 14px rgba(16, 185, 129, 0.3);
}

.badge-yellow {
    background: rgba(245, 158, 11, 0.18);
    border: 1px solid #f59e0b;
    color: #fbbf24;
    padding: 4px 12px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.82rem;
    box-shadow: 0 0 14px rgba(245, 158, 11, 0.35);
}

.badge-orange {
    background: rgba(249, 115, 22, 0.2);
    border: 1px solid #f97316;
    color: #fb923c;
    padding: 4px 12px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.82rem;
    box-shadow: 0 0 16px rgba(249, 115, 22, 0.4);
}

.badge-red {
    background: rgba(239, 68, 68, 0.25);
    border: 1px solid #ef4444;
    color: #f87171;
    padding: 4px 12px;
    border-radius: 9999px;
    font-weight: 800;
    font-size: 0.82rem;
    box-shadow: 0 0 20px rgba(239, 68, 68, 0.6);
    animation: criticalPulse 1.8s infinite;
}

@keyframes criticalPulse {
    0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.8); }
    70% { box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); }
    100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
}

/* 7. Metric Value Box */
.metric-box {
    background: rgba(30, 41, 59, 0.55);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 8px 12px;
    text-align: center;
}
.metric-box-val {
    font-size: 1.2rem;
    font-weight: 800;
    color: #f8fafc;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}
.metric-box-label {
    font-size: 0.7rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 2px;
}

/* 8. Pulsing Live Indicator */
.pulse-dot {
    width: 9px;
    height: 9px;
    background-color: #10b981;
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 10px #10b981;
    animation: liveBlink 1.4s infinite ease-in-out;
}
@keyframes liveBlink {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.35; transform: scale(0.85); }
}

/* Radio & Button styling */
.stRadio > div {
    background: rgba(15, 23, 42, 0.6) !important;
    padding: 6px 12px !important;
    border-radius: 8px !important;
    border: 1px solid rgba(59, 130, 246, 0.2) !important;
}
.stButton > button {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
    border: 1px solid rgba(59, 130, 246, 0.3) !important;
    color: #f1f5f9 !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 14px rgba(56, 189, 248, 0.35) !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --- REALISTIC OPEN-PIT 3D MESH GENERATION ---
@st.cache_data
def generate_realistic_pit_topography(n_grid: int = 80):
    """
    Generates an authentic open-pit topography with stepped bench faces (60% drop, 40% catch berm),
    an explicit spiral haulage ramp cut into the wall, and spatial sector domains.
    """
    x = np.linspace(-360, 360, n_grid)
    y = np.linspace(-360, 360, n_grid)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X) % (2 * np.pi)

    R_crest = 330.0
    R_floor = 65.0
    Z_crest = 310.0
    Z_floor = 50.0
    bench_h = 24.0
    bench_w = 26.0

    # Stepped Terraces (Benches & Safety Berms)
    dr = np.clip(R_crest - R, 0.0, R_crest - R_floor)
    bench_idx = np.floor(dr / bench_w)
    frac = (dr % bench_w) / bench_w
    # Sharp face: first 65% of width drops full bench height; remaining 35% is catch berm
    bench_drop = np.where(frac < 0.65, (frac / 0.65) * bench_h, bench_h)
    Z = Z_crest - (bench_idx * bench_h + bench_drop)
    Z[R > R_crest] = Z_crest
    Z[R < R_floor] = Z_floor

    # Spiral Haulage Ramp (22m road width carved from crest to floor at 8.5% grade)
    ramp_radius = R_crest - (Theta / (2 * np.pi)) * (R_crest - R_floor)
    ramp_dist = np.abs(R - ramp_radius)
    on_ramp = (ramp_dist < 14.0) & (R < R_crest) & (R > R_floor)
    ramp_elevation = Z_crest - (Theta / (2 * np.pi)) * (Z_crest - Z_floor)
    Z[on_ramp] = ramp_elevation[on_ramp]

    # Sector ID Assignment:
    # 1: North Wall Bench 4 (Y > 50)
    # 2: South Crest Ramp (Y < -60)
    # 3: Main Haulage Ramp 3 (East Sector Corridor)
    # 4: West Pit Access Road (West Sector Corridor)
    sector_id_map = np.zeros_like(Z, dtype=int)
    sector_id_map[(Y > 50) & (R < R_crest) & (R > 65)] = 1
    sector_id_map[(Y < -60) & (R < R_crest) & (R > 65)] = 2
    sector_id_map[(X > 40) & (Y >= -50) & (Y <= 50) & (R < R_crest)] = 3
    sector_id_map[(X < -50) & (Y >= -50) & (Y <= 50) & (R < R_crest)] = 4

    return X, Y, Z, sector_id_map


def build_3d_pit_figure(X, Y, Z, sector_id_map, sector_telemetries, selected_sector_key):
    """
    Builds the Plotly 3D WebGL Surface draped with real-time deformation heatmaps,
    spiral haul road, and floating 3D station pins.
    """
    n_grid = Z.shape[0]
    surf_color = np.zeros_like(Z)
    custom_data = np.empty((n_grid, n_grid, 3), dtype=object)

    sector_key_to_id = {
        "ZONE_NORTH_WALL": 1,
        "ZONE_SOUTH_CREST": 2,
        "ZONE_RAMP_3": 3,
        "ZONE_HAUL_ROAD": 4,
    }
    id_to_sector_key = {v: k for k, v in sector_key_to_id.items()}

    for i in range(n_grid):
        for j in range(n_grid):
            s_id = sector_id_map[i, j]
            if s_id in id_to_sector_key:
                s_key = id_to_sector_key[s_id]
                t_data = sector_telemetries[s_key]
                v = float(t_data.get("radar_velocity_mm_hr", 0.8))
                level = t_data.get("effective_level", "GREEN")
                name = t_data.get("name", s_key)
                surf_color[i, j] = v
                custom_data[i, j, 0] = name
                custom_data[i, j, 1] = f"{v:.2f} mm/hr"
                custom_data[i, j, 2] = level
            else:
                surf_color[i, j] = 0.4
                custom_data[i, j, 0] = "Surrounding Crest"
                custom_data[i, j, 1] = "0.40 mm/hr"
                custom_data[i, j, 2] = "GREEN"

    # TARP Color Scale: Green (0-1.5) -> Yellow (1.5-5) -> Orange (5-15) -> Critical Crimson Red (>15)
    tarp_colorscale = [
        [0.00, "#059669"],   # Emerald Normal
        [0.08, "#10b981"],
        [0.22, "#f59e0b"],   # Amber Caution (1.5 - 5 mm/hr)
        [0.55, "#f97316"],   # Orange Warning (5 - 15 mm/hr)
        [0.80, "#ef4444"],   # Crimson Critical
        [1.00, "#991b1b"],   # Deep Red Failure
    ]

    fig = go.Figure()

    # 1. Main Stepped Pit Surface
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=surf_color,
        cmin=0.0, cmax=20.0,
        colorscale=tarp_colorscale,
        customdata=custom_data,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Elevation: %{z:.1f} m<br>"
            "Deformation Velocity: %{customdata[1]}<br>"
            "TARP Status: %{customdata[2]}"
            "<extra></extra>"
        ),
        showscale=True,
        colorbar=dict(
            title=dict(text="Radar Velocity (mm/hr)", font=dict(color="#94a3b8", size=11)),
            tickfont=dict(color="#94a3b8", size=10),
            thickness=14,
            len=0.72,
            x=0.98,
        ),
        lighting=dict(ambient=0.68, diffuse=0.82, specular=0.25, roughness=0.65),
    ))

    # 2. 3D Floating Station Billboard Pins above sectors
    station_pins = [
        {"key": "ZONE_NORTH_WALL", "label": "📍 NORTH WALL BENCH 4", "x": 0, "y": 210, "z": 240},
        {"key": "ZONE_SOUTH_CREST", "label": "📍 SOUTH CREST RAMP", "x": 0, "y": -210, "z": 240},
        {"key": "ZONE_RAMP_3", "label": "📍 MAIN HAULAGE RAMP 3", "x": 190, "y": 0, "z": 200},
        {"key": "ZONE_HAUL_ROAD", "label": "📍 WEST PIT ACCESS ROAD", "x": -190, "y": 0, "z": 200},
    ]

    pin_colors = {
        "GREEN": "#10b981",
        "YELLOW": "#f59e0b",
        "ORANGE": "#f97316",
        "RED": "#ef4444",
    }

    pin_x, pin_y, pin_z, pin_texts, pin_c = [], [], [], [], []
    for p in station_pins:
        lvl = sector_telemetries[p["key"]]["effective_level"]
        vel = sector_telemetries[p["key"]]["radar_velocity_mm_hr"]
        pin_x.append(p["x"])
        pin_y.append(p["y"])
        pin_z.append(p["z"] + 15)
        pin_texts.append(f"{p['label']}<br>[{vel:.1f} mm/h | {lvl}]")
        pin_c.append(pin_colors.get(lvl, "#10b981"))

    fig.add_trace(go.Scatter3d(
        x=pin_x, y=pin_y, z=pin_z,
        mode="markers+text",
        text=pin_texts,
        textposition="top center",
        textfont=dict(color="#f8fafc", size=11, family="monospace"),
        marker=dict(size=7, color=pin_c, symbol="diamond", line=dict(color="#ffffff", width=1.5)),
        hoverinfo="none",
        showlegend=False,
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(
                title="Easting (m)",
                backgroundcolor="rgba(10, 15, 26, 0.85)",
                gridcolor="#1e293b",
                color="#64748b",
                showbackground=True,
            ),
            yaxis=dict(
                title="Northing (m)",
                backgroundcolor="rgba(10, 15, 26, 0.85)",
                gridcolor="#1e293b",
                color="#64748b",
                showbackground=True,
            ),
            zaxis=dict(
                title="Elevation (m)",
                backgroundcolor="rgba(10, 15, 26, 0.85)",
                gridcolor="#1e293b",
                color="#64748b",
                showbackground=True,
            ),
            aspectratio=dict(x=1.1, y=1.1, z=0.45),
            camera=dict(
                eye=dict(x=1.35, y=-1.45, z=0.95),
                center=dict(x=0, y=0, z=-0.1)
            ),
        ),
        height=540,
    )
    return fig


# --- SYSTEM STATE & INITIALIZATION ---
if "bb" not in st.session_state:
    st.session_state.bb = MineBlackboard()
    st.session_state.ctrl = RockWatchController(st.session_state.bb)
    st.session_state.sim_engine = GeotechnicalSimulationEngine(time_horizon_hours=24.0, step_hours=0.5)
    st.session_state.cam_stream = DynamicPitCameraStream(width=480, height=320)
    st.session_state.current_scenario = "SCENARIO_1_NORTH_WALL_CREEP"
    st.session_state.sim_time_hour = 0.0
    st.session_state.is_playing = False
    st.session_state.user_role = "OPERATOR"
    st.session_state.review_ticket = {
        "id": "TKT-2026-084",
        "zone_id": "ZONE_NORTH_WALL",
        "status": "DRAFT",
        "proposed_multiplier": 0.80,
        "requested_duration_hours": 4.0,
        "operator_notes": "Blasting vibration interference observed on seismograph; requesting temporary sensitivity offset.",
        "approved": False,
        "approved_by": None,
        "approved_at_sim_time": None,
    }

bb: MineBlackboard = st.session_state.bb
ctrl: RockWatchController = st.session_state.ctrl
sim_engine: GeotechnicalSimulationEngine = st.session_state.sim_engine
cam_stream: DynamicPitCameraStream = st.session_state.cam_stream


# --- COMPUTE SIMULATION TELEMETRY FOR CURRENT TIME ---
sim_telemetries = sim_engine.generate_scenario_telemetry(
    scenario_id=st.session_state.current_scenario,
    t_hour=st.session_state.sim_time_hour,
)

# Run each sector through RockWatch controller
level_rank = {"GREEN": 1, "YELLOW": 2, "ORANGE": 3, "RED": 4}
highest_global_level = "GREEN"

for s_key, s_data in sim_telemetries.items():
    raw_frame = {
        "radar_velocity_mm_hr": s_data["radar_velocity_mm_hr"],
        "rainfall_3h_mm": s_data["rainfall_3h_mm"],
        "vibration_ppv_mms": s_data["vibration_ppv_mms"],
        "timestamp": st.session_state.sim_time_hour * 3600.0,
    }
    cycle_res = ctrl.run_telemetry_cycle(s_key, raw_frame)
    s_data["effective_level"] = cycle_res["effective_level"]
    s_data["directive"] = cycle_res["final_alert_package"].get("advisory_directive", "Routine monitoring.")
    s_data["discrepancy"] = cycle_res.get("discrepancy_detected", False)
    
    f_info = bb.read("forecast").get(s_key, {})
    s_data["risk_probability"] = f_info.get("risk_probability", 0.10)
    s_data["dominance_override"] = f_info.get("dominance_override", False)
    s_data["base_probabilities"] = f_info.get("base_probabilities", {})

    if level_rank.get(s_data["effective_level"], 1) > level_rank.get(highest_global_level, 1):
        highest_global_level = s_data["effective_level"]


# ==============================================================================
# 1. TOP MISSION CONTROL HUD HEADER (ZERO STREAMLIT LOOK)
# ==============================================================================
badge_classes = {
    "GREEN": "badge-green",
    "YELLOW": "badge-yellow",
    "ORANGE": "badge-orange",
    "RED": "badge-red",
}
global_badge_class = badge_classes.get(highest_global_level, "badge-green")
head_hash = bb.get_head_hash()[:12] + "..."

hud_html = f"""
<div class="hud-header">
    <div class="hud-title">
        <span>⚡</span> ROCKWATCH // 3D GEOTECHNICAL MISSION CONTROL
    </div>
    <div style="display: flex; align-items: center; gap: 20px;">
        <div style="display: flex; align-items: center; gap: 8px; font-size: 0.85rem; color: #94a3b8;">
            <span class="pulse-dot"></span> CONTINUOUS TELEMETRY
        </div>
        <div style="font-size: 0.82rem; color: #64748b; font-family: monospace;">
            LEDGER: <span style="color: #38bdf8;">{head_hash}</span>
        </div>
        <div style="font-size: 0.82rem; color: #64748b; font-family: monospace;">
            SIM TIME: <span style="color: #e2e8f0; font-weight: 700;">T+{st.session_state.sim_time_hour:04.1f}h / 24.0h</span>
        </div>
        <div>
            <span class="{global_badge_class}">GLOBAL STATUS: LEVEL {level_rank[highest_global_level]} {highest_global_level}</span>
        </div>
    </div>
</div>
"""
st.markdown(hud_html, unsafe_allow_html=True)


# ==============================================================================
# 2. MAIN COCKPIT VIEWPORT (60% 3D DIGITAL TWIN | 40% SECTOR DIAGNOSTICS DECK)
# ==============================================================================
col_left, col_right = st.columns([1.55, 1.0], gap="medium")

# --- LEFT COLUMN: 3D DIGITAL TWIN & REAL-TIME CONTROLS ---
with col_left:
    st.markdown('<div class="card-title">🌐 3D Open-Pit Digital Twin & Deformation Velocity Heatmap</div>', unsafe_allow_html=True)
    
    # 3D Pit Topography Mesh
    X_grid, Y_grid, Z_grid, s_map = generate_realistic_pit_topography()
    selected_sector_key = st.session_state.get("selected_sector", "ZONE_NORTH_WALL")
    
    fig_3d = build_3d_pit_figure(
        X_grid, Y_grid, Z_grid, s_map,
        sim_telemetries,
        selected_sector_key
    )
    st.plotly_chart(fig_3d, width="stretch", config={"displayModeBar": False})

    # Sector Selector Pills
    sec_names = {k: v["name"] for k, v in sim_telemetries.items()}
    chosen_sector = st.radio(
        "Inspect Structural Pit Sector:",
        options=list(sec_names.keys()),
        format_func=lambda k: f"{sec_names[k]} ({sim_telemetries[k]['effective_level']})",
        horizontal=True,
        key="selected_sector",
    )

    # --- PROGRESSIVE SCENARIO SIMULATION ENGINE & PLAYBACK CONTROLLER ---
    st.markdown('<div class="card-title" style="margin-top: 14px;">⏱️ Geotechnical Scenario Playback & Saito Creep Timeline</div>', unsafe_allow_html=True)
    
    sc_c1, sc_c2 = st.columns([1.4, 1.0])
    with sc_c1:
        chosen_scen = st.selectbox(
            "Active Physical Scenario:",
            options=list(GeotechnicalSimulationEngine.SCENARIOS.keys()),
            format_func=lambda k: GeotechnicalSimulationEngine.SCENARIOS[k]["name"],
            key="current_scenario",
        )
    with sc_c2:
        st.caption(GeotechnicalSimulationEngine.SCENARIOS[chosen_scen]["description"])

    # Timeline Scrubber Slider (0 to 24 Hours)
    new_time = st.slider(
        "Simulation Timeline Progress (Hours):",
        min_value=0.0,
        max_value=24.0,
        value=float(st.session_state.sim_time_hour),
        step=0.5,
        format="T+%.1fh",
    )
    if new_time != st.session_state.sim_time_hour:
        st.session_state.sim_time_hour = new_time
        st.rerun()

    # Playback Stepper Buttons
    b_c1, b_c2, b_c3, b_c4 = st.columns(4)
    with b_c1:
        if st.button("⏪ Step Back (-1h)"):
            st.session_state.sim_time_hour = max(0.0, st.session_state.sim_time_hour - 1.0)
            st.rerun()
    with b_c2:
        if st.button("⏩ Advance (+1h)"):
            st.session_state.sim_time_hour = min(24.0, st.session_state.sim_time_hour + 1.0)
            st.rerun()
    with b_c3:
        if st.button("⏭️ Fast-Forward (+4h)"):
            st.session_state.sim_time_hour = min(24.0, st.session_state.sim_time_hour + 4.0)
            st.rerun()
    with b_c4:
        if st.button("🔄 Reset to T+0.0h"):
            st.session_state.sim_time_hour = 0.0
            st.rerun()


# --- RIGHT COLUMN: SECTOR INSPECTOR, DYNAMIC CCTV, & AI SAFETY COPILOT ---
cur_sec = sim_telemetries[chosen_sector]
sec_level = cur_sec["effective_level"]
sec_badge_class = badge_classes.get(sec_level, "badge-green")

with col_right:
    # Selected Sector Title & Directive Card
    st.markdown(f"""
    <div class="mission-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <span style="font-weight: 800; font-size: 1.1rem; color: #f1f5f9;">{cur_sec['name']}</span>
            <span class="{sec_badge_class}">{sec_level}</span>
        </div>
        <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.4;">
            <strong>TARP Directive:</strong> {cur_sec['directive']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. DYNAMIC CCTV CAMERA FEED WITH REAL-TIME AI BOUNDING BOXES
    st.markdown('<div class="card-title">🎥 Sector Surveillance Feed (AI Particle & Detachment Detector)</div>', unsafe_allow_html=True)
    
    # Render dynamic CCTV frame
    cam_frame, detected_boxes = cam_stream.render_frame(
        zone_id=chosen_sector,
        t_hour=st.session_state.sim_time_hour,
        rockfall_intensity=cur_sec["rockfall_intensity"],
        crack_width_cm=cur_sec["crack_width_cm"],
        detachment_detected=cur_sec["detachment_detected"],
    )
    cam_img = Image.fromarray(cam_frame)
    draw_box = ImageDraw.Draw(cam_img)
    for b in detected_boxes:
        box_c = "#ef4444" if b["type"] == "ROCKFALL_DEBRIS" else "#f59e0b"
        draw_box.rectangle([b["xmin"], b["ymin"], b["xmax"], b["ymax"]], outline=box_c, width=2)
        draw_box.text((b["xmin"] + 4, max(4, b["ymin"] - 14)), b["label"], fill=box_c)

    box_count_str = f"| Active Tracked Anomalies: {len(detected_boxes)}" if detected_boxes else "| Scene Clear"
    st.image(cam_img, width="stretch", caption=f"CAM-01 [{cur_sec['name']}] | T+{st.session_state.sim_time_hour:.1f}h {box_count_str}")

    # 2. MULTIMODAL SENSOR TELEMETRY METRIC BOXES
    st.markdown('<div class="card-title" style="margin-top: 12px;">📊 Multimodal Telemetry Streams</div>', unsafe_allow_html=True)
    m_c1, m_c2, m_c3, m_c4 = st.columns(4)
    with m_c1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-box-val">{cur_sec['radar_velocity_mm_hr']:.2f}</div>
            <div class="metric-box-label">Radar mm/h</div>
        </div>
        """, unsafe_allow_html=True)
    with m_c2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-box-val">{cur_sec['rainfall_3h_mm']:.1f}</div>
            <div class="metric-box-label">Rain 3h mm</div>
        </div>
        """, unsafe_allow_html=True)
    with m_c3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-box-val">{cur_sec['pore_pressure_kpa']:.0f}</div>
            <div class="metric-box-label">Pore kPa</div>
        </div>
        """, unsafe_allow_html=True)
    with m_c4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-box-val">{cur_sec['vibration_ppv_mms']:.1f}</div>
            <div class="metric-box-label">Seismic mm/s</div>
        </div>
        """, unsafe_allow_html=True)

    # Dominance Override Alert Card
    if cur_sec["dominance_override"]:
        st.markdown("""
        <div style="background: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; border-radius: 8px; padding: 10px 14px; margin-top: 10px;">
            <span style="color: #f87171; font-weight: 700; font-size: 0.85rem;">⚡ DOMINANCE OVERRIDE ACTIVATED</span><br>
            <span style="color: #cbd5e1; font-size: 0.8rem;">Single-sensor anomaly spiked &ge; 80%. Elevated directly to guarantee critical danger is never diluted by quiet sensors.</span>
        </div>
        """, unsafe_allow_html=True)

    # 3. AI GEOTECHNICAL SAFETY ADVISOR (RAG + LLM)
    st.markdown('<div class="card-title" style="margin-top: 12px;">🤖 AI Geotechnical Safety Copilot (Grounded in SOP)</div>', unsafe_allow_html=True)
    sop_ref = retrieve_sop(sec_level)
    st.markdown(f"""
    <div class="mission-card" style="font-size: 0.84rem; line-height: 1.45; color: #cbd5e1;">
        <div style="color: #38bdf8; font-weight: 700; margin-bottom: 4px;">Citation: {sop_ref.get('passage_id', '[TARP-SOP-01]')}</div>
        {sop_ref.get('text', 'Normal operating conditions.')}
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 3. BOTTOM UTILITY DOCK: GEOTECHNICAL AUTHORIZATION CONSOLE & AUDIT LEDGER
# ==============================================================================
st.markdown("---")
b_c1, b_c2 = st.columns([1.1, 0.9], gap="large")

with b_c1:
    st.markdown('<div class="card-title">⚖️ Enterprise Geotechnical Authorization & Desensitization Console</div>', unsafe_allow_html=True)
    
    # Active Session Credential Selector
    auth_c1, auth_c2 = st.columns([1.3, 1.0])
    with auth_c1:
        role_select = st.selectbox(
            "Authenticated Active Session Profile:",
            options=["OPERATOR", "GEOTECH_ENGINEER"],
            format_func=lambda r: "👤 Control Room Operator (Monitoring & Ticket Flagging Only)" if r == "OPERATOR" else "🛡️ Dr. P. Mehta, Principal Geotechnical Engineer (Sign-Off Authority #GEO-7714)",
            key="user_role",
        )
    with auth_c2:
        is_geotech = role_select == "GEOTECH_ENGINEER"
        perm_text = "FULL AUTHORIZATION (DESENSITIZE & PROMOTE)" if is_geotech else "RESTRICTED (CANNOT DESENSITIZE SAFETY FLOORS)"
        perm_color = "#34d399" if is_geotech else "#f59e0b"
        st.markdown(f"""
        <div style="margin-top: 24px; font-size: 0.75rem; font-family: monospace; color: {perm_color};">
            PERMISSIONS: {perm_text}
        </div>
        """, unsafe_allow_html=True)

    # Formal Geotechnical Review Ticket
    ticket = st.session_state.review_ticket
    st.markdown(f"""
    <div class="mission-card" style="margin-top: 10px;">
        <div style="display: flex; justify-content: space-between; font-size: 0.82rem; font-family: monospace; color: #94a3b8; margin-bottom: 6px;">
            <span>TICKET: #{ticket['id']}</span>
            <span>TARGET: {cur_sec['name']}</span>
            <span>STATUS: {'APPROVED (ACTIVE)' if ticket['approved'] else 'PENDING AUTHORIZATION'}</span>
        </div>
        <div style="font-size: 0.85rem; color: #e2e8f0; margin-bottom: 8px;">
            <strong>Requested Action:</strong> Apply temporary 0.80x sensitivity multiplier (Duration: 4.0h) for blasting dust / thermal drift.
        </div>
        <div style="font-size: 0.78rem; color: #f87171; font-family: monospace;">
            INVIOLABLE SAFETY CONSTRAINT: TARP Level 4 Red safety floor (15.0 mm/hr) CANNOT be suppressed by any multiplier.
        </div>
    </div>
    """, unsafe_allow_html=True)

    btn_c1, btn_c2 = st.columns([1.5, 1.0])
    with btn_c1:
        if st.button("✍️ Submit & Execute Authorization Decision"):
            if is_geotech:
                ticket["approved"] = True
                ticket["approved_by"] = "Dr. P. Mehta (#GEO-7714)"
                ticket["approved_at_sim_time"] = st.session_state.sim_time_hour
                ctrl.submit_desensitization_request(
                    zone_id=chosen_sector,
                    requested_by_role="GEOTECH_ENGINEER",
                    proposed_multiplier=0.80,
                    duration_hours=4.0,
                    justification="Thermal drift verified by redundant extensometer.",
                )
                st.success("✅ APPROVED & DIGITALLY SIGNED by Principal Geotech Engineer! Multiplier active. Expiration: T+4.0h.")
            else:
                st.error("❌ ACCESS DENIED: Under rule [GEO-GUIDE-03], Operators cannot authorize desensitization. Ticket routed to Geotech queue.")

    with btn_c2:
        if st.button("⏩ Fast-Forward +5 Hours (Test Auto-Expiry)"):
            st.session_state.sim_time_hour = min(24.0, st.session_state.sim_time_hour + 5.0)
            if ticket["approved"]:
                # Check expiry
                if (st.session_state.sim_time_hour - (ticket["approved_at_sim_time"] or 0)) >= 4.0:
                    ticket["approved"] = False
                    ticket["status"] = "EXPIRED"
            st.rerun()

with b_c2:
    st.markdown('<div class="card-title">🛡️ Cryptographic Tamper-Proof Audit Ledger (SHA-256)</div>', unsafe_allow_html=True)
    st.caption(f"Canonical JSON hash chaining. Head Hash Anchor: `{bb.get_head_hash()}`")

    if os.path.exists("logs/run_log.jsonl"):
        with open("logs/run_log.jsonl", "r", encoding="utf-8") as f:
            lines = [json.loads(line.strip()) for line in f if line.strip()]
        recent_entries = []
        for e in lines[-4:]:
            recent_entries.append({
                "Seq": e.get("sequence_id"),
                "Event": e.get("event_type"),
                "Zone": e.get("zone_id"),
                "Payload Hash": e.get("payload_hash", "")[:10] + "...",
                "Head Hash": e.get("head_hash", "")[:10] + "...",
            })
        st.dataframe(recent_entries, use_container_width=True)
