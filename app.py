"""
app.py - RockWatch: Human-Friendly 3D Open-Pit Mine Safety Copilot.

Simplified, intuitive, human-centered design:
- 3D Open-Pit Digital Twin as the visual anchor: clean floating terrain, natural lookout camera angle,
  no cluttering CAD gridlines or axis boxes, authentic rock strata shading, and glowing hazard halos.
- Prominent AI Safety Assistant on the right: friendly, conversational copilot with one-click quick prompts
  answering whether conditions are normal, explaining risks in plain English, and guiding operators on SOPs.
- Progressive Disclosure: Click any pit sector to get a simple, clear status ("Is everything normal or not?")
  with an optional dropdown to inspect detailed sensor numbers (Radar, Pore Pressure, Rainfall, Vibration).
- Administration & Cryptographic Audit Ledger tucked away in a clean bottom expander so the operator interface
  stays clean, focused, and human-friendly.
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import json
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import streamlit as st

from blackboard import (
    MineBlackboard,
    FORECASTER_BASELINE_SENSITIVITY,
    TARP_SAFETY_FLOOR_VELOCITY_MM_HR,
)
from controller import RockWatchController
from tools.rag_tools import retrieve_sop
from tools.simulation_engine import GeotechnicalSimulationEngine

# --- PAGE CONFIGURATION (Full Width, Dark Base) ---
st.set_page_config(
    page_title="RockWatch // 3D Mine Safety Copilot",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- CLEAN MODERN INDUSTRIAL STYLES (HUMAN-FRIENDLY & ZERO STREAMLIT CHROME) ---
CUSTOM_CSS = """
<style>
/* 1. Eradicate default Streamlit chrome */
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

/* 2. Page layout */
.main, .block-container {
    padding-top: 0.6rem !important;
    padding-bottom: 1.2rem !important;
    padding-left: 1.2rem !important;
    padding-right: 1.2rem !important;
    max-width: 100% !important;
}

/* 3. Deep space background */
body, .stApp {
    background: radial-gradient(circle at 50% 10%, #0d1527 0%, #060913 70%, #020408 100%) !important;
    color: #e2e8f0 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
}

/* 4. Top Mission Header */
.hud-header {
    background: linear-gradient(90deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.85) 50%, rgba(15, 23, 42, 0.95) 100%);
    border: 1px solid rgba(56, 189, 248, 0.28);
    border-radius: 12px;
    padding: 10px 18px;
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
}
.hud-title {
    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: 0.8px;
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: flex;
    align-items: center;
    gap: 10px;
}
.pulse-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background-color: #10b981;
    box-shadow: 0 0 10px #10b981;
    display: inline-block;
    animation: pulse 1.8s infinite;
}
@keyframes pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1.15); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

/* 5. Clean cards */
.clean-card {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 12px;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
}

/* 6. Human-friendly Status Badges */
.status-pill-green {
    background: rgba(16, 185, 129, 0.18);
    border: 1px solid #10b981;
    color: #34d399;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.status-pill-yellow {
    background: rgba(245, 158, 11, 0.18);
    border: 1px solid #f59e0b;
    color: #fbbf24;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.status-pill-orange {
    background: rgba(249, 115, 22, 0.2);
    border: 1px solid #f97316;
    color: #fb923c;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.status-pill-red {
    background: rgba(239, 68, 68, 0.22);
    border: 1px solid #ef4444;
    color: #f87171;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    animation: pulse-red 1.5s infinite;
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 12px rgba(239, 68, 68, 0.6); }
    50% { box-shadow: 0 0 4px rgba(239, 68, 68, 0.2); }
}

/* 7. Chat Styling */
.chat-bubble-ai {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 10px;
    font-size: 0.88rem;
    line-height: 1.5;
    color: #f1f5f9;
}
.chat-bubble-user {
    background: rgba(14, 116, 144, 0.25);
    border: 1px solid rgba(14, 165, 233, 0.35);
    border-radius: 10px;
    padding: 9px 13px;
    margin-bottom: 10px;
    font-size: 0.88rem;
    line-height: 1.4;
    color: #38bdf8;
    text-align: right;
}

/* 8. Radio Buttons */
.stRadio > div {
    background: rgba(15, 23, 42, 0.65) !important;
    padding: 5px 10px !important;
    border-radius: 10px !important;
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
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.35) !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# REALISTIC 3D OPEN-PIT MINE TOPOGRAPHY
# ==============================================================================
@st.cache_data
def generate_clean_pit_topography(n_grid: int = 85):
    """
    Generates authentic 3D open-pit mine topography with natural aesthetics:
    - Multi-lobed open-pit amphitheater (North Pit + South-East Expansion)
    - Bench terraces with realistic catch berms
    - Spiral switchback haul road carved into rock walls
    - Natural surrounding mountain ridges
    - Pit floor sump retention lagoon
    """
    x = np.linspace(-360, 360, n_grid)
    y = np.linspace(-360, 360, n_grid)
    X, Y = np.meshgrid(x, y)

    # 1. Multi-Lobed Pit Geometry
    r1 = np.sqrt(((X - 20) / 1.30)**2 + ((Y - 40) / 0.95)**2)
    r2 = np.sqrt(((X + 40) / 1.15)**2 + ((Y + 70) / 1.20)**2)
    k = 0.035
    R_pit = -np.log(np.exp(-k * r1) + np.exp(-k * r2)) / k

    R_crest = 265.0
    R_floor = 48.0
    Z_crest = 310.0
    Z_floor = 70.0
    bench_h = 24.0
    bench_w = 26.0

    # 2. Bench Terraces & Berms
    dr = np.clip(R_crest - R_pit, 0.0, R_crest - R_floor)
    bench_idx = np.floor(dr / bench_w)
    frac = (dr % bench_w) / bench_w
    bench_drop = np.where(frac < 0.70, (frac / 0.70) * bench_h, bench_h)
    Z = Z_crest - (bench_idx * bench_h + bench_drop)
    Z[R_pit < R_floor] = Z_floor

    # 3. Surrounding Natural Mountain Ridges
    ridge = 30.0 * np.sin(X / 100.0) * np.cos(Y / 120.0) + 15.0 * np.sin((X + Y) / 80.0)
    crest_mask = R_pit >= R_crest
    blend = np.clip((R_pit - R_crest) / 45.0, 0.0, 1.0)
    Z[crest_mask] = Z_crest + (ridge * blend)[crest_mask]

    # 4. Spiral Switchback Haul Road
    Theta = np.arctan2(Y, X) % (2 * np.pi)
    ramp_radius = R_crest - (Theta / (2 * np.pi)) * (R_crest - R_floor)
    ramp_dist = np.abs(R_pit - ramp_radius)
    on_ramp = (ramp_dist < 15.0) & (R_pit < R_crest) & (R_pit > R_floor)
    ramp_elevation = Z_crest - (Theta / (2 * np.pi)) * (Z_crest - Z_floor)
    Z[on_ramp] = ramp_elevation[on_ramp]

    # 5. Sector Zone ID Mapping
    sector_id_map = np.zeros_like(Z, dtype=int)
    sector_id_map[(Y > 35) & (R_pit < R_crest) & (R_pit > 50)] = 1   # North Wall
    sector_id_map[(Y < -45) & (R_pit < R_crest) & (R_pit > 50)] = 2  # South Crest
    sector_id_map[(X > 35) & (Y >= -40) & (Y <= 40) & (R_pit < R_crest)] = 3  # Ramp 3
    sector_id_map[(X < -40) & (Y >= -40) & (Y <= 40) & (R_pit < R_crest)] = 4  # Haul Road

    return X, Y, Z, sector_id_map, R_pit


# ==============================================================================
# BUILD GORGEOUS, CLEAN 3D MINE FIGURE (NO UGLY CAD BOXES OR NUMBER TICKS)
# ==============================================================================
def build_clean_3d_mine_figure(
    X, Y, Z, sector_id_map, R_pit,
    sim_telemetries: dict,
    chosen_sector: str,
    hazard_mode: str,
    view_angle: str = "🔭 Mine Lookout Deck (Default)",
):
    """
    Renders a stunning 3D mine visualization:
    - Clean floating digital twin (NO axis boxes, NO distracting grid lines, NO graph ticks)
    - Realistic rock lithology & warm sandstone bench shading
    - Dynamic, smooth hazard alert glow on active risk zones
    - Clean floating 3D status pins for the 4 sectors
    - Authentic Lookout Deck vantage point
    """
    n_grid = Z.shape[0]
    fig = go.Figure()

    # Calculate alert aura for active sector
    active_vel = float(sim_telemetries[chosen_sector]["radar_velocity_mm_hr"])
    sector_centers = {
        "ZONE_NORTH_WALL": (15.0, 175.0),
        "ZONE_SOUTH_CREST": (-20.0, -185.0),
        "ZONE_RAMP_3": (160.0, 0.0),
        "ZONE_HAUL_ROAD": (-160.0, 0.0),
    }
    cx, cy = sector_centers.get(chosen_sector, (15.0, 175.0))
    dist_to_target = np.sqrt((X - cx)**2 + (Y - cy)**2)
    alert_intensity = np.exp(-((dist_to_target / 75.0)**2))

    # Prepare custom data for hover
    custom_data = np.empty((n_grid, n_grid, 3), dtype=object)
    for i in range(n_grid):
        for j in range(n_grid):
            z_val = Z[i, j]
            if z_val >= 290:
                rock_type = "Overburden Regolith / Mountain Crest"
            elif z_val >= 240:
                rock_type = "Quartzite & Sandstone Benches"
            elif z_val >= 180:
                rock_type = "Carbonaceous Shale Highwall"
            elif z_val >= 120:
                rock_type = "Mineralized Porphyry Ore Vein"
            elif z_val >= 75:
                rock_type = "Lower Bench Footwall"
            else:
                rock_type = "Pit Floor Water Sump"

            is_target = dist_to_target[i, j] < 80.0
            sec_name = sim_telemetries[chosen_sector]["name"] if is_target else "Mine Perimeter"
            status_text = sim_telemetries[chosen_sector]["effective_level"] if is_target else "NORMAL"
            
            custom_data[i, j, 0] = rock_type
            custom_data[i, j, 1] = f"{sec_name} ({status_text})"
            custom_data[i, j, 2] = f"{z_val:.0f}m Elevation"

    # Color mapping: natural earth tones with glowing alert overlay
    earth_colorscale = [
        [0.00, "#0891b2"],  # 70m Sump lagoon (deep teal water)
        [0.04, "#0e7490"],
        [0.05, "#52525b"],  # Footwall slate
        [0.20, "#71717a"],
        [0.21, "#c2410c"],  # Ore seam (terracotta copper)
        [0.42, "#9a3412"],
        [0.43, "#3f3f46"],  # Shale
        [0.60, "#27272a"],
        [0.61, "#a8a29e"],  # Sandstone highwall
        [0.78, "#78716c"],
        [0.79, "#d97706"],  # Overburden warm golden earth
        [1.00, "#92400e"],  # 350m Mountain ridge
    ]

    # Render main terrain surface
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=Z,
        cmin=70.0, cmax=350.0,
        colorscale=earth_colorscale,
        customdata=custom_data,
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "Layer: %{customdata[0]}<br>"
            "Height: %{customdata[2]}"
            "<extra></extra>"
        ),
        showscale=False,  # No ugly colorbar distracting the operator
        lighting=dict(ambient=0.62, diffuse=0.88, roughness=0.85, specular=0.15),
    ))

    # Hazard Halo (Smooth glowing alert overlay on active hazard zone)
    if hazard_mode != "NORMAL":
        halo_colors = {
            "MILD": "rgba(245, 158, 11, 0.45)",       # Soft Amber
            "MODERATE": "rgba(249, 115, 22, 0.65)",   # Vivid Orange
            "SEVERE": "rgba(239, 68, 68, 0.85)",      # Glowing Crimson
        }
        halo_z = Z + (alert_intensity * 3.5)
        # Add smooth glow scatter markers on the active sector
        mask = alert_intensity > 0.45
        fig.add_trace(go.Scatter3d(
            x=X[mask], y=Y[mask], z=halo_z[mask] + 1.5,
            mode="markers",
            marker=dict(
                size=4,
                color=halo_colors.get(hazard_mode, "rgba(239, 68, 68, 0.8)"),
                opacity=0.75,
            ),
            hoverinfo="none",
            showlegend=False,
            name="Hazard Alert Glow",
        ))

    # Sector Pins (Floating human-friendly badges)
    pin_data = [
        {"key": "ZONE_NORTH_WALL", "name": "North Wall Bench 4", "x": 15, "y": 175, "z": 260},
        {"key": "ZONE_SOUTH_CREST", "name": "South Crest", "x": -20, "y": -185, "z": 260},
        {"key": "ZONE_RAMP_3", "name": "Ramp 3 Switchback", "x": 160, "y": 0, "z": 220},
        {"key": "ZONE_HAUL_ROAD", "name": "West Access Road", "x": -160, "y": 0, "z": 220},
    ]

    pin_x, pin_y, pin_z, pin_labels, pin_colors = [], [], [], [], []
    for p in pin_data:
        lvl = sim_telemetries[p["key"]]["effective_level"]
        color = "#10b981" if lvl == "GREEN" else ("#f59e0b" if lvl == "YELLOW" else ("#f97316" if lvl == "ORANGE" else "#ef4444"))
        pin_x.append(p["x"])
        pin_y.append(p["y"])
        pin_z.append(p["z"] + 12)
        pin_labels.append(f"📍 {p['name']} [{lvl}]")
        pin_colors.append(color)

    fig.add_trace(go.Scatter3d(
        x=pin_x, y=pin_y, z=pin_z,
        mode="markers+text",
        text=pin_labels,
        textposition="top center",
        textfont=dict(color="#f8fafc", size=10, family="-apple-system, sans-serif"),
        marker=dict(size=7, color=pin_colors, symbol="diamond", line=dict(color="#ffffff", width=1.5)),
        hoverinfo="none",
        showlegend=False,
    ))

    # Camera Perspectives (Natural Human Observation Deck Angles)
    camera_presets = {
        "🔭 Mine Lookout Deck (Default)": dict(eye=dict(x=0.05, y=-1.75, z=1.12), center=dict(x=0, y=0.1, z=-0.1)),
        "🌐 Aerial 45° Perspective": dict(eye=dict(x=1.35, y=-1.35, z=0.95), center=dict(x=0, y=0, z=-0.1)),
        "🎯 Target Zone Close-Up": dict(eye=dict(x=cx/250.0 + 0.1, y=cy/250.0 - 0.7, z=0.65), center=dict(x=cx/400.0, y=cy/400.0, z=-0.1)),
    }
    cur_camera = camera_presets.get(view_angle, camera_presets["🔭 Mine Lookout Deck (Default)"])

    # Completely hide axis boxes, lines, tick marks, and grids
    clean_axis = dict(
        showgrid=False,
        showline=False,
        zeroline=False,
        showticklabels=False,
        title="",
        showbackground=False,
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=clean_axis,
            yaxis=clean_axis,
            zaxis=clean_axis,
            aspectratio=dict(x=1.15, y=1.15, z=0.40),
            camera=cur_camera,
        ),
        height=490,
    )
    return fig


# ==============================================================================
# SYSTEM STATE & INITIALIZATION
# ==============================================================================
if "bb" not in st.session_state:
    st.session_state.bb = MineBlackboard()
    st.session_state.ctrl = RockWatchController(st.session_state.bb)
    st.session_state.sim_engine = GeotechnicalSimulationEngine(time_horizon_hours=24.0, step_hours=0.5)
    st.session_state.hazard_mode = "NORMAL"  # Default clean start
    st.session_state.sim_time_hour = 12.0
    st.session_state.user_role = "OPERATOR"
    st.session_state.selected_sector = "ZONE_NORTH_WALL"
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": "👋 **Hello Operator!** I'm your RockWatch AI Safety Copilot. All 4 pit sectors are currently **STABLE (Level 1 Green)**. Ask me anything about slope conditions, radar trends, or safety protocols!"
        }
    ]
    st.session_state.review_ticket = {
        "id": "TKT-2026-084",
        "zone_id": "ZONE_NORTH_WALL",
        "status": "DRAFT",
        "proposed_multiplier": 0.80,
        "requested_duration_hours": 4.0,
        "approved": False,
        "approved_by": None,
        "approved_at_sim_time": None,
    }

bb: MineBlackboard = st.session_state.bb
ctrl: RockWatchController = st.session_state.ctrl

# --- SIMPLE USER-FRIENDLY SCENARIO PRESETS ---
SCENARIO_PRESETS = {
    "NORMAL": {
        "label": "🟢 Normal (All Safe)",
        "summary": "Everything is normal across the entire mine. Benches are stable.",
        "vel": 0.75,
        "rain": 1.2,
        "pore": 24.0,
        "ppv": 0.35,
        "crack_cm": 0.0,
        "status": "GREEN",
        "directive": "All operations authorized. Routine monitoring active.",
    },
    "MILD": {
        "label": "🟡 Mild Caution",
        "summary": "Minor surface gravel movement & small tension crack opening.",
        "vel": 2.85,
        "rain": 4.5,
        "pore": 42.0,
        "ppv": 0.85,
        "crack_cm": 2.4,
        "status": "YELLOW",
        "directive": "Restrict non-essential foot traffic. Alert pit supervisor.",
    },
    "MODERATE": {
        "label": "🟠 Moderate Warning",
        "summary": "Fissure expanding to 8 cm. Rock slab fracturing detected.",
        "vel": 7.85,
        "rain": 18.0,
        "pore": 72.0,
        "ppv": 2.4,
        "crack_cm": 8.2,
        "status": "ORANGE",
        "directive": "Stand down haul trucks under highwall. 150m exclusion zone.",
    },
    "SEVERE": {
        "label": "🔴 Severe Emergency",
        "summary": "IMMINENT DETACHMENT: Major slope collapse underway!",
        "vel": 22.40,
        "rain": 62.0,
        "pore": 135.0,
        "ppv": 6.8,
        "crack_cm": 18.5,
        "status": "RED",
        "directive": "MANDATORY EVACUATION: Clear all personnel within 500m immediately!",
    },
}

active_mode = st.session_state.hazard_mode
preset = SCENARIO_PRESETS[active_mode]
selected_sector_key = st.session_state.get("selected_sector", "ZONE_NORTH_WALL")

# Compute sector states
sim_telemetries = {
    "ZONE_NORTH_WALL": {
        "name": "North Wall Bench 4",
        "radar_velocity_mm_hr": preset["vel"] if selected_sector_key == "ZONE_NORTH_WALL" else 0.75,
        "rainfall_3h_mm": preset["rain"] if selected_sector_key == "ZONE_NORTH_WALL" else 1.2,
        "pore_pressure_kpa": preset["pore"] if selected_sector_key == "ZONE_NORTH_WALL" else 24.0,
        "vibration_ppv_mms": preset["ppv"] if selected_sector_key == "ZONE_NORTH_WALL" else 0.35,
        "crack_cm": preset["crack_cm"] if selected_sector_key == "ZONE_NORTH_WALL" else 0.0,
        "effective_level": preset["status"] if selected_sector_key == "ZONE_NORTH_WALL" else "GREEN",
        "directive": preset["directive"] if selected_sector_key == "ZONE_NORTH_WALL" else "Routine monitoring.",
        "dominance_override": preset["status"] == "RED" if selected_sector_key == "ZONE_NORTH_WALL" else False,
    },
    "ZONE_SOUTH_CREST": {
        "name": "South Crest",
        "radar_velocity_mm_hr": preset["vel"] if selected_sector_key == "ZONE_SOUTH_CREST" else 0.55,
        "rainfall_3h_mm": preset["rain"] if selected_sector_key == "ZONE_SOUTH_CREST" else 1.0,
        "pore_pressure_kpa": preset["pore"] if selected_sector_key == "ZONE_SOUTH_CREST" else 22.0,
        "vibration_ppv_mms": preset["ppv"] if selected_sector_key == "ZONE_SOUTH_CREST" else 0.30,
        "crack_cm": preset["crack_cm"] if selected_sector_key == "ZONE_SOUTH_CREST" else 0.0,
        "effective_level": preset["status"] if selected_sector_key == "ZONE_SOUTH_CREST" else "GREEN",
        "directive": preset["directive"] if selected_sector_key == "ZONE_SOUTH_CREST" else "Routine monitoring.",
        "dominance_override": preset["status"] == "RED" if selected_sector_key == "ZONE_SOUTH_CREST" else False,
    },
    "ZONE_RAMP_3": {
        "name": "Ramp 3 Switchback",
        "radar_velocity_mm_hr": preset["vel"] if selected_sector_key == "ZONE_RAMP_3" else 0.60,
        "rainfall_3h_mm": preset["rain"] if selected_sector_key == "ZONE_RAMP_3" else 0.8,
        "pore_pressure_kpa": preset["pore"] if selected_sector_key == "ZONE_RAMP_3" else 24.0,
        "vibration_ppv_mms": preset["ppv"] if selected_sector_key == "ZONE_RAMP_3" else 0.40,
        "crack_cm": preset["crack_cm"] if selected_sector_key == "ZONE_RAMP_3" else 0.0,
        "effective_level": preset["status"] if selected_sector_key == "ZONE_RAMP_3" else "GREEN",
        "directive": preset["directive"] if selected_sector_key == "ZONE_RAMP_3" else "Routine monitoring.",
        "dominance_override": preset["status"] == "RED" if selected_sector_key == "ZONE_RAMP_3" else False,
    },
    "ZONE_HAUL_ROAD": {
        "name": "West Access Road",
        "radar_velocity_mm_hr": preset["vel"] if selected_sector_key == "ZONE_HAUL_ROAD" else 0.45,
        "rainfall_3h_mm": preset["rain"] if selected_sector_key == "ZONE_HAUL_ROAD" else 0.5,
        "pore_pressure_kpa": preset["pore"] if selected_sector_key == "ZONE_HAUL_ROAD" else 20.0,
        "vibration_ppv_mms": preset["ppv"] if selected_sector_key == "ZONE_HAUL_ROAD" else 0.25,
        "crack_cm": preset["crack_cm"] if selected_sector_key == "ZONE_HAUL_ROAD" else 0.0,
        "effective_level": preset["status"] if selected_sector_key == "ZONE_HAUL_ROAD" else "GREEN",
        "directive": preset["directive"] if selected_sector_key == "ZONE_HAUL_ROAD" else "Routine monitoring.",
        "dominance_override": preset["status"] == "RED" if selected_sector_key == "ZONE_HAUL_ROAD" else False,
    },
}

# Run controller cycle
raw_frame = {
    "radar_velocity_mm_hr": sim_telemetries[selected_sector_key]["radar_velocity_mm_hr"],
    "rainfall_3h_mm": sim_telemetries[selected_sector_key]["rainfall_3h_mm"],
    "vibration_ppv_mms": sim_telemetries[selected_sector_key]["vibration_ppv_mms"],
    "timestamp": st.session_state.sim_time_hour * 3600.0,
}
cycle_res = ctrl.run_telemetry_cycle(selected_sector_key, raw_frame)
sim_telemetries[selected_sector_key]["effective_level"] = cycle_res["effective_level"]
sim_telemetries[selected_sector_key]["directive"] = cycle_res["final_alert_package"].get("advisory_directive", preset["directive"])

level_rank = {"GREEN": 1, "YELLOW": 2, "ORANGE": 3, "RED": 4}
highest_global_level = max([s["effective_level"] for s in sim_telemetries.values()], key=lambda l: level_rank.get(l, 1))

status_classes = {
    "GREEN": "status-pill-green",
    "YELLOW": "status-pill-yellow",
    "ORANGE": "status-pill-orange",
    "RED": "status-pill-red",
}
global_status_pill = status_classes.get(highest_global_level, "status-pill-green")


# ==============================================================================
# TOP MISSION CONTROL HEADER
# ==============================================================================
st.markdown(f"""
<div class="hud-header">
    <div class="hud-title">
        <span>⚡</span> ROCKWATCH // 3D OPEN-PIT MINE SAFETY COPILOT
    </div>
    <div style="display: flex; align-items: center; gap: 16px;">
        <div style="font-size: 0.85rem; color: #94a3b8; display: flex; align-items: center; gap: 6px;">
            <span class="pulse-dot"></span> AI Copilot & Sensors Active
        </div>
        <div>
            <span class="{global_status_pill}">MINE STATUS: LEVEL {level_rank[highest_global_level]} {highest_global_level}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# MAIN VIEWPORT: 3D MINE DIGITAL TWIN (LEFT) & PROMINENT AI COPILOT (RIGHT)
# ==============================================================================
col_mine, col_ai = st.columns([1.35, 1.0], gap="medium")


# ------------------------------------------------------------------------------
# 1. THE 3D MINE DIGITAL TWIN (THE VISUAL ANCHOR)
# ------------------------------------------------------------------------------
with col_mine:
    # Clean Top Controls Row
    top_c1, top_c2 = st.columns([1.2, 1.0])
    with top_c1:
        chosen_sector = st.radio(
            "Select Pit Sector to Inspect:",
            options=list(sim_telemetries.keys()),
            format_func=lambda k: f"{sim_telemetries[k]['name']} ({sim_telemetries[k]['effective_level']})",
            horizontal=True,
            key="selected_sector",
        )
    with top_c2:
        selected_mode = st.radio(
            "Simulate Safety Scenario:",
            options=list(SCENARIO_PRESETS.keys()),
            format_func=lambda m: SCENARIO_PRESETS[m]["label"],
            horizontal=True,
            key="hazard_mode",
        )

    # Perspective buttons
    view_angle = st.radio(
        "Camera View:",
        options=["🔭 Mine Lookout Deck (Default)", "🌐 Aerial 45° Perspective", "🎯 Target Zone Close-Up"],
        horizontal=True,
        key="view_angle_choice",
        label_visibility="collapsed",
    )

    cur_sec = sim_telemetries[chosen_sector]
    sec_level = cur_sec["effective_level"]
    sec_pill_class = status_classes.get(sec_level, "status-pill-green")

    # Generate and render the 3D model
    X_grid, Y_grid, Z_grid, s_map, R_pit = generate_clean_pit_topography(n_grid=85)
    fig_3d = build_clean_3d_mine_figure(
        X_grid, Y_grid, Z_grid, s_map, R_pit,
        sim_telemetries,
        chosen_sector,
        selected_mode,
        view_angle,
    )
    st.plotly_chart(fig_3d, width="stretch", config={"displayModeBar": False})

    # PROGRESSIVE DISCLOSURE: Simple Human-Friendly Sector Card
    # Directly answers: "Is everything going normal or not?"
    if sec_level == "GREEN":
        status_msg = "✅ <strong>Everything is Normal & Safe.</strong> All slope stability radar, pore pressure, and seismic sensors report stable ground conditions. Normal operations authorized."
    elif sec_level == "YELLOW":
        status_msg = "⚠️ <strong>Attention: Mild Movement Detected.</strong> Radar creep has accelerated slightly (2.8 mm/h) and a hairline crest crack (2.4 cm) has opened. Restrict non-essential foot traffic."
    elif sec_level == "ORANGE":
        status_msg = "🟠 <strong>Warning: Active Ground Fissure.</strong> Scarp crack expanded to 8.2 cm with bench slab detachment. Stand down haulage under highwall; dispatch inspection crew."
    else:
        status_msg = "🚨 <strong>CRITICAL EMERGENCY: Catastrophic Slope Detachment Underway!</strong> Radar runaway exceeded 15 mm/h safety floor. Sound pit alarms and evacuate 500m radius immediately!"

    st.markdown(f"""
    <div class="clean-card" style="margin-top: -6px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <span style="font-weight: 800; font-size: 1.05rem; color: #f1f5f9;">{cur_sec['name']}</span>
            <span class="{sec_pill_class}">STATUS: {sec_level}</span>
        </div>
        <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.45; margin-bottom: 8px;">
            {status_msg}
        </div>
        <div style="font-size: 0.82rem; color: #94a3b8;">
            <strong>Action Directive:</strong> {cur_sec['directive']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # OPTIONAL ON-DEMAND SENSOR DETAILS (Only if operator wants to see numbers)
    with st.expander("📊 Inspect Detailed Sensor Numbers (Radar, Water Pressure, Rain, Vibration)", expanded=False):
        d_c1, d_c2, d_c3, d_c4 = st.columns(4)
        with d_c1:
            st.metric("Radar Movement", f"{cur_sec['radar_velocity_mm_hr']:.2f} mm/h", help="Safe baseline is < 1.5 mm/h")
        with d_c2:
            st.metric("Pore Water Pressure", f"{cur_sec['pore_pressure_kpa']:.0f} kPa", help="Groundwater saturation in borehole")
        with d_c3:
            st.metric("Rainfall (Last 3h)", f"{cur_sec['rainfall_3h_mm']:.1f} mm", help="Antecedent 3-hour precipitation")
        with d_c4:
            st.metric("Tension Crack", f"{cur_sec['crack_cm']:.1f} cm", help="Crest wireline extensometer aperture")


# ------------------------------------------------------------------------------
# 2. PROMINENT AI SAFETY ASSISTANT (THE EXPERT COPILOT ON THE RIGHT)
# ------------------------------------------------------------------------------
with col_ai:
    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid rgba(59, 130, 246, 0.2); padding-bottom: 8px; margin-bottom: 10px;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 1.3rem;">🤖</span>
            <div>
                <div style="font-weight: 800; font-size: 1.05rem; color: #f1f5f9;">RockWatch AI Safety Assistant</div>
                <div style="font-size: 0.75rem; color: #38bdf8;">Always online • Analyzing multi-sensor slope conditions in real-time</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick One-Click Starter Questions
    st.markdown('<div style="font-size: 0.78rem; font-weight: 700; color: #94a3b8; margin-bottom: 6px;">QUICK QUESTIONS:</div>', unsafe_allow_html=True)
    q_c1, q_c2 = st.columns(2)
    trigger_prompt = None
    with q_c1:
        if st.button("🟢 Is everything safe right now?", use_container_width=True):
            trigger_prompt = "Is everything safe right now in the mine?"
        if st.button("⚠️ What if heavy rain starts?", use_container_width=True):
            trigger_prompt = "What happens if heavy rain starts falling?"
    with q_c2:
        if st.button("🔍 What is happening at North Wall?", use_container_width=True):
            trigger_prompt = f"What is happening right now at {cur_sec['name']}?"
        if st.button("📋 What are the evacuation rules?", use_container_width=True):
            trigger_prompt = "What are the mandatory evacuation rules for Level 4 Red?"

    # Chat Messages Window
    chat_box = st.container(height=360)
    with chat_box:
        for msg in st.session_state.chat_history:
            if msg["role"] == "assistant":
                st.markdown(f"""
                <div class="chat-bubble-ai">
                    <strong>🤖 RockWatch Copilot:</strong><br>{msg['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-bubble-user">
                    <strong>👤 Operator:</strong> {msg['content']}
                </div>
                """, unsafe_allow_html=True)

    # Chat Input Box
    user_input = st.chat_input("Ask the AI Copilot about mine safety, alerts, or procedures...")
    prompt_to_process = trigger_prompt or user_input

    if prompt_to_process:
        st.session_state.chat_history.append({"role": "user", "content": prompt_to_process})
        sop_info = retrieve_sop(sec_level)
        q_lower = prompt_to_process.lower()

        # Natural, human-friendly conversational responses
        if "safe" in q_lower or "normal" in q_lower or "status" in q_lower:
            if highest_global_level == "GREEN":
                reply = (
                    "✅ **Yes, everything is completely safe and normal!**\n\n"
                    "All 4 pit sectors are currently at **Level 1 Green**. "
                    f"Radar deformation is at a healthy baseline of **{cur_sec['radar_velocity_mm_hr']:.2f} mm/h**, "
                    "groundwater pore pressure is low, and no tension cracks have opened. "
                    "Normal haulage and excavation operations are authorized across all benches."
                )
            elif highest_global_level == "YELLOW":
                reply = (
                    f"⚠️ **Attention is required at {cur_sec['name']}.**\n\n"
                    f"The mine is currently at **Level 2 Yellow Caution**. "
                    f"Radar detected ground creep accelerating to **{cur_sec['radar_velocity_mm_hr']:.2f} mm/h**, "
                    f"and an extensometer registered a **{cur_sec['crack_cm']:.1f} cm** tension crack. "
                    "**Recommended action:** Restrict non-essential foot traffic on Bench 4 and notify the pit supervisor."
                )
            elif highest_global_level == "ORANGE":
                reply = (
                    f"🟠 **Warning: Elevated risk detected at {cur_sec['name']}!**\n\n"
                    f"We have entered **Level 3 Orange Warning**. Radar velocity reached **{cur_sec['radar_velocity_mm_hr']:.2f} mm/h**, "
                    f"and pore pressure climbed to **{cur_sec['pore_pressure_kpa']:.0f} kPa**. "
                    "**Recommended action:** Stand down haul trucks operating beneath the highwall and maintain a 150m exclusion perimeter."
                )
            else:
                reply = (
                    "🚨 **CRITICAL EMERGENCY: The mine is NOT safe!**\n\n"
                    f"**{cur_sec['name']}** has reached **Level 4 Red Emergency**. "
                    f"Radar acceleration has breached the **15.0 mm/h** safety floor (currently reading **{cur_sec['radar_velocity_mm_hr']:.2f} mm/h**). "
                    "**Immediate Action:** Sound pit sirens and execute immediate evacuation of all personnel within 500 meters!"
                )

        elif "north wall" in q_lower or "happening" in q_lower or "inspect" in q_lower:
            reply = (
                f"**Current Diagnostic for {cur_sec['name']}:**\n\n"
                f"- **Safety Status:** **Level {level_rank[sec_level]} {sec_level}**\n"
                f"- **Radar Movement:** {cur_sec['radar_velocity_mm_hr']:.2f} mm/h (Safe threshold is < 1.5 mm/h)\n"
                f"- **Crack Opening:** {cur_sec['crack_cm']:.1f} cm\n"
                f"- **Pore Water Pressure:** {cur_sec['pore_pressure_kpa']:.0f} kPa\n"
                f"- **Required Action:** *{cur_sec['directive']}*"
            )

        elif "evacuat" in q_lower or "rule" in q_lower or "sop" in q_lower or "procedure" in q_lower:
            reply = (
                f"**Mandatory Protocol for {sec_level} Status (Citation: `{sop_info.get('passage_id', '[TARP-SOP-01]')}`):**\n\n"
                f"{sop_info.get('text', 'Routine monitoring.')}\n\n"
                "**Operator Action Checklist:**\n"
                "1. Confirm alert on control room audio channel.\n"
                "2. Radio mobile equipment operators in target sector.\n"
                f"3. {'MANDATORY: Evacuate all personnel and equipment immediately beyond 500m perimeter.' if sec_level == 'RED' else 'Inspect crest catch berms and verify extensometer tension.'}"
            )

        elif "rain" in q_lower or "weather" in q_lower:
            reply = (
                "🌧️ **Rainfall Hazard Guidelines:**\n\n"
                "- **Normal:** < 5 mm in 3 hours (Current reading: "
                f"**{cur_sec['rainfall_3h_mm']:.1f} mm**).\n"
                "- **Caution:** 5 - 15 mm triggers heightened pore pressure monitoring.\n"
                "- **Warning:** > 15 mm in 3 hours automatically restricts haul ramp traffic due to reduced shear strength.\n"
                "- **Emergency:** > 50 mm antecedent rain triggers mandatory bench clearance."
            )
        else:
            reply = (
                f"**RockWatch Copilot:** {cur_sec['name']} is currently at **Level {level_rank[sec_level]} {sec_level}**. "
                f"Governing protocol is `{sop_info.get('passage_id', '[TARP-SOP-01]')}`: *{sop_info.get('text', 'Routine monitoring.')}*"
            )

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()


# ==============================================================================
# 3. BOTTOM ACCORDION: ADVANCED ADMIN & AUDIT LOGS (OUT OF THE WAY)
# ==============================================================================
st.markdown("---")
with st.expander("🔒 Geotechnical Administration, Dual-Role Authorization & Audit Ledger", expanded=False):
    st.caption("Enterprise management console: Geotechnical sign-off authority and cryptographic run log verification.")
    
    adm_c1, adm_c2 = st.columns([1.1, 0.9], gap="large")
    with adm_c1:
        st.markdown("**Dual-Role Geotechnical Sign-Off Console**")
        role_select = st.selectbox(
            "Active Session Profile:",
            options=["OPERATOR", "GEOTECH_ENGINEER"],
            format_func=lambda r: "👤 Control Room Operator (Monitoring Only)" if r == "OPERATOR" else "🛡️ Dr. P. Mehta, Principal Geotechnical Engineer (#GEO-7714)",
            key="user_role",
        )
        is_geotech = role_select == "GEOTECH_ENGINEER"
        perm_text = "FULL AUTHORIZATION (DESENSITIZE & PROMOTE)" if is_geotech else "RESTRICTED (CANNOT DESENSITIZE SAFETY FLOORS)"
        perm_color = "#34d399" if is_geotech else "#f59e0b"
        st.markdown(f"<div style='font-size: 0.78rem; font-family: monospace; color: {perm_color}; margin-bottom: 8px;'>PERMISSIONS: {perm_text}</div>", unsafe_allow_html=True)

        ticket = st.session_state.review_ticket
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 8px; padding: 10px; margin-bottom: 8px; font-size: 0.82rem;">
            <strong>Ticket #{ticket['id']}</strong> | Target: {cur_sec['name']} | Status: {'APPROVED' if ticket['approved'] else 'PENDING'}<br>
            <span style="color: #94a3b8;">Requested Action: Apply temporary 0.80x multiplier for thermal drift.</span><br>
            <span style="color: #f87171; font-family: monospace; font-size: 0.75rem;">Safety constraint: TARP Red 15.0 mm/h safety floor cannot be suppressed.</span>
        </div>
        """, unsafe_allow_html=True)

        sub_c1, sub_c2 = st.columns(2)
        with sub_c1:
            if st.button("✍️ Authorize Decision", use_container_width=True):
                if is_geotech:
                    ticket["approved"] = True
                    ticket["approved_by"] = "Dr. P. Mehta (#GEO-7714)"
                    ctrl.submit_desensitization_request(
                        zone_id=chosen_sector,
                        requested_by_role="GEOTECH_ENGINEER",
                        proposed_multiplier=0.80,
                        duration_hours=4.0,
                        justification="Thermal drift verified by extensometer.",
                    )
                    st.success("✅ APPROVED & DIGITALLY SIGNED by Principal Geotech Engineer!")
                else:
                    st.error("❌ ACCESS DENIED: Operators cannot authorize desensitization.")
        with sub_c2:
            if st.button("⏩ Fast-Forward +5h (Auto-Expiry)", use_container_width=True):
                st.session_state.sim_time_hour = min(24.0, st.session_state.sim_time_hour + 5.0)
                if ticket["approved"]:
                    ticket["approved"] = False
                    ticket["status"] = "EXPIRED"
                st.rerun()

    with adm_c2:
        st.markdown(f"**Cryptographic Tamper-Proof Audit Ledger (SHA-256)** — Anchor: `{bb.get_head_hash()[:14]}...`")
        if os.path.exists("logs/run_log.jsonl"):
            with open("logs/run_log.jsonl", "r", encoding="utf-8") as f:
                lines = [json.loads(line.strip()) for line in f if line.strip()]
            entries = []
            for e in lines[-4:]:
                entries.append({
                    "Seq": e.get("sequence_id"),
                    "Event": e.get("event_type"),
                    "Zone": e.get("zone_id"),
                    "Payload": e.get("payload_hash", "")[:10] + "...",
                    "Head": e.get("head_hash", "")[:10] + "...",
                })
            st.dataframe(pd.DataFrame(entries), width="stretch", hide_index=True)
