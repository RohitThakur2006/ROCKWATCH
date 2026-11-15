"""
app.py - RockWatch Mission-Control: 3D Geotechnical Digital Twin & Multi-Sensor Monitoring Center.

Fully customized industrial mission-control interface:
- High-tech dark aerospace command gradient with zero Streamlit chrome (headers/toolbars hidden).
- Realistic 3D Open-Pit Mine Digital Twin featured in the center: authentic multi-lobed topography,
  engineered bench terraces, spiral switchback haul roads, waste rock dump plateau, natural mountain ridges,
  geological lithology rock strata shading, and dynamic ground radar (InSAR/SSR) displacement heatmaps.
- Physical 3D equipment & sensor instrumentation: SSR-02 radar scanner trailer with 3D raycast fan beams,
  borehole piezometers, in-place inclinometers, crest extensometers, robotic total station prisms, and haul trucks.
- Real-time Multi-Sensor Telemetry Array on the left wing: Slope Stability Radar (SSR), Vibrating Wire Piezometers,
  In-Place Inclinometers, Surface Crack Extensometers, Blast Seismographs (PPV), and Meteorological Rain Gauges.
- TARP Early Warning, Slope Stability Physics (Bishop FoS & Saito 1/v Creep Law), and Multi-Agent Consensus on the right.
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
from plotly.subplots import make_subplots

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
    padding-top: 0.5rem !important;
    padding-bottom: 1.2rem !important;
    padding-left: 1.0rem !important;
    padding-right: 1.0rem !important;
    max-width: 100% !important;
}

/* 3. Deep space mission-control background gradient */
body, .stApp {
    background: radial-gradient(circle at 50% 10%, #0d1527 0%, #060913 65%, #020408 100%) !important;
    color: #e2e8f0 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
}

/* 4. Top Mission Control HUD Header */
.hud-header {
    background: linear-gradient(90deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.85) 50%, rgba(15, 23, 42, 0.95) 100%);
    border: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: 10px;
    padding: 10px 18px;
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.1);
}
.hud-title {
    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: 1.2px;
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

/* 5. Mission Card containers */
.mission-card {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(59, 130, 246, 0.22);
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 10px;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.35);
}
.card-title {
    font-size: 0.88rem;
    font-weight: 700;
    letter-spacing: 0.8px;
    color: #94a3b8;
    text-transform: uppercase;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
    border-bottom: 1px solid rgba(59, 130, 246, 0.15);
    padding-bottom: 4px;
}

/* 6. Metric readouts */
.metric-box {
    background: rgba(2, 6, 23, 0.65);
    border: 1px solid rgba(56, 189, 248, 0.18);
    border-radius: 8px;
    padding: 7px 10px;
    text-align: center;
}
.metric-box-val {
    font-size: 1.15rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    color: #38bdf8;
    line-height: 1.1;
}
.metric-box-label {
    font-size: 0.70rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 3px;
}

/* 7. TARP Status Badges */
.badge-green {
    background: rgba(16, 185, 129, 0.2);
    border: 1px solid #10b981;
    color: #34d399;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.8px;
}
.badge-yellow {
    background: rgba(245, 158, 11, 0.2);
    border: 1px solid #f59e0b;
    color: #fbbf24;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.8px;
}
.badge-orange {
    background: rgba(249, 115, 22, 0.2);
    border: 1px solid #f97316;
    color: #fb923c;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.8px;
}
.badge-red {
    background: rgba(239, 68, 68, 0.25);
    border: 1px solid #ef4444;
    color: #f87171;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.8px;
    animation: flash-red 1.5s infinite;
}
@keyframes flash-red {
    0%, 100% { border-color: #ef4444; box-shadow: 0 0 12px rgba(239, 68, 68, 0.6); }
    50% { border-color: #b91c1c; box-shadow: 0 0 4px rgba(239, 68, 68, 0.2); }
}

/* 8. Radio & Button styling */
.stRadio > div {
    background: rgba(15, 23, 42, 0.6) !important;
    padding: 4px 8px !important;
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


# ==============================================================================
# REALISTIC ASYMMETRIC OPEN-PIT MINE 3D TOPOGRAPHY (THE LIVE MODEL)
# ==============================================================================
@st.cache_data
def generate_realistic_pit_topography(n_grid: int = 100):
    """
    Generates authentic 3D open-pit mine topography:
    - Multi-lobed asymmetric pit geometry (North Pit Lobe + South-East Expansion Lobe)
    - 65° engineered bench terraces with 14m safety catch berms every 22m vertical drop
    - Spiral switchback haul road (24m wide, 8.5% grade) carved into the rock highwall
    - Natural surrounding mountain ridges and drainage gullies (Z = 310m -> 368m)
    - External Terraced Waste Rock Dump in South-West (3 distinct overburden lifts)
    - Deep mineralized pit sump retention lagoon (Z <= 75m)
    """
    x = np.linspace(-400, 400, n_grid)
    y = np.linspace(-400, 400, n_grid)
    X, Y = np.meshgrid(x, y)

    # 1. Multi-Lobed Asymmetric Open-Pit Geometry
    r1 = np.sqrt(((X - 25) / 1.35)**2 + ((Y - 45) / 0.95)**2)
    r2 = np.sqrt(((X + 45) / 1.15)**2 + ((Y + 85) / 1.25)**2)
    k = 0.035
    # Smooth minimum blend of the two pit lobes
    R_pit = -np.log(np.exp(-k * r1) + np.exp(-k * r2)) / k

    R_crest = 290.0
    R_floor = 50.0
    Z_crest = 320.0
    Z_floor = 65.0
    bench_h = 22.0
    bench_w = 24.0

    # 2. Stepped Bench Terraces & Catch Berms
    dr = np.clip(R_crest - R_pit, 0.0, R_crest - R_floor)
    bench_idx = np.floor(dr / bench_w)
    frac = (dr % bench_w) / bench_w
    bench_drop = np.where(frac < 0.68, (frac / 0.68) * bench_h, bench_h)
    Z = Z_crest - (bench_idx * bench_h + bench_drop)
    Z[R_pit < R_floor] = Z_floor

    # 3. Surrounding Natural Mountain Ridges & Valleys outside Crest
    ridge = 35.0 * np.sin(X / 110.0) * np.cos(Y / 130.0) + 18.0 * np.sin((X + 1.2 * Y) / 75.0) + 8.0 * np.cos(X / 40.0)
    crest_mask = R_pit >= R_crest
    blend = np.clip((R_pit - R_crest) / 50.0, 0.0, 1.0)
    Z[crest_mask] = Z_crest + (ridge * blend)[crest_mask]

    # 4. External Terraced Waste Dump (South-West Plateau)
    dump_dist = np.sqrt((X + 280)**2 + (Y + 240)**2)
    in_dump = (dump_dist < 115) & (X < -165) & (Y < -135)
    dump_lifts = np.floor(np.clip(115 - dump_dist, 0, 95) / 28) * 18.0
    Z[in_dump] = np.maximum(Z[in_dump], 305.0 + dump_lifts[in_dump])

    # 5. Spiral Switchback Haul Road (24m road width cut into pit highwall)
    Theta = np.arctan2(Y, X) % (2 * np.pi)
    ramp_radius = R_crest - (Theta / (2 * np.pi)) * (R_crest - R_floor)
    ramp_dist = np.abs(R_pit - ramp_radius)
    on_ramp = (ramp_dist < 16.0) & (R_pit < R_crest) & (R_pit > R_floor)
    ramp_elevation = Z_crest - (Theta / (2 * np.pi)) * (Z_crest - Z_floor)
    Z[on_ramp] = ramp_elevation[on_ramp]

    # 6. Pit Sector Zones
    sector_id_map = np.zeros_like(Z, dtype=int)
    sector_id_map[(Y > 40) & (R_pit < R_crest) & (R_pit > 55)] = 1   # North Wall
    sector_id_map[(Y < -50) & (R_pit < R_crest) & (R_pit > 55)] = 2  # South Crest
    sector_id_map[(X > 40) & (Y >= -40) & (Y <= 40) & (R_pit < R_crest)] = 3  # Ramp 3
    sector_id_map[(X < -45) & (Y >= -40) & (Y <= 40) & (R_pit < R_crest)] = 4  # Haul Road

    return X, Y, Z, sector_id_map, R_pit


# ==============================================================================
# BUILD REALISTIC 3D OPEN-PIT DIGITAL TWIN FIGURE
# ==============================================================================
def build_realistic_3d_mine_model(
    X, Y, Z, sector_id_map, R_pit,
    sim_telemetries: dict,
    chosen_sector: str,
    hazard_mode: str,
    render_mode: str,
    camera_view: str,
):
    """
    Constructs the high-fidelity 3D realistic mine model in Plotly:
    - Authentic geological rock strata shading (Overburden, Quartzite, Shale, Ore, Dolomite, Sump)
    - Dynamic SSR Radar interferometric displacement heatmaps draped over highwall
    - 3D physical sensor instrumentation (SSR trailer, raycasts, piezometers, inclinometers, extensometers, prisms, haul trucks)
    - Camera angle presets
    """
    n_grid = Z.shape[0]
    fig = go.Figure()

    # --- 1. COMPUTE SURFACE COLORING & HOVER DATA ---
    custom_data = np.empty((n_grid, n_grid, 4), dtype=object)
    
    # Calculate localized radar deformation field
    radar_vel_grid = np.full_like(Z, 0.45)  # baseline stable background creep
    active_vel = float(sim_telemetries[chosen_sector]["radar_velocity_mm_hr"])
    
    # Center of active sector's hazard scarp
    sector_centers = {
        "ZONE_NORTH_WALL": (15.0, 185.0),
        "ZONE_SOUTH_CREST": (-25.0, -195.0),
        "ZONE_RAMP_3": (175.0, 0.0),
        "ZONE_HAUL_ROAD": (-175.0, 0.0),
    }
    cx, cy = sector_centers.get(chosen_sector, (15.0, 185.0))
    dist_to_scarp = np.sqrt((X - cx)**2 + (Y - cy)**2)
    scarp_influence = np.exp(-((dist_to_scarp / 85.0)**2))
    
    if hazard_mode != "NORMAL":
        radar_vel_grid += (active_vel - 0.45) * scarp_influence

    for i in range(n_grid):
        for j in range(n_grid):
            z_val = Z[i, j]
            v_val = radar_vel_grid[i, j]
            s_id = sector_id_map[i, j]
            
            # Lithology tag
            if z_val >= 295:
                lith = "Overburden Regolith / Topsoil"
            elif z_val >= 245:
                lith = "Jointed Quartzite & Sandstone"
            elif z_val >= 185:
                lith = "Carbonaceous Shale / Shear Zone"
            elif z_val >= 120:
                lith = "Mineralized Porphyry Ore Seam"
            elif z_val >= 75:
                lith = "Footwall Dolomite & Breccia"
            else:
                lith = "Pit Sump Retention Lagoon"
                
            custom_data[i, j, 0] = lith
            custom_data[i, j, 1] = f"{z_val:.1f} m RL"
            custom_data[i, j, 2] = f"{v_val:.2f} mm/h"
            custom_data[i, j, 3] = "ZONE: " + chosen_sector.replace("ZONE_", "") if dist_to_scarp[i, j] < 90 else "Mine Field"

    # --- 2. COLORSCALE DEFINITIONS ---
    # A) Realistic Geological Lithology Strata Colorscale
    lithology_colorscale = [
        [0.00, "#0891b2"],  # 65m - Turquoise Pit Sump Lagoon
        [0.03, "#0e7490"],  # 75m
        [0.04, "#52525b"],  # Dolomite
        [0.18, "#71717a"],  # 120m
        [0.19, "#994d38"],  # Terracotta-Ochre Copper Ore Horizon
        [0.39, "#7c2d12"],  # 185m
        [0.40, "#3f3f46"],  # Carbonaceous Graphite Shale
        [0.59, "#27272a"],  # 245m
        [0.60, "#8a837b"],  # Sandstone & Quartzite Benches
        [0.75, "#78716c"],  # 295m
        [0.76, "#b59d79"],  # Overburden Weathered Regolith
        [1.00, "#8c7355"],  # 370m Surrounding Mountain Crest
    ]

    # B) Radar Interferometric (InSAR / SSR) Displacement Heatmap Colorscale
    radar_colorscale = [
        [0.00, "#10b981"],  # Emerald Normal (< 1.5 mm/hr)
        [0.08, "#10b981"],
        [0.20, "#f59e0b"],  # Amber Caution (1.5 - 5 mm/hr)
        [0.55, "#f97316"],  # Orange Warning (5 - 15 mm/hr)
        [0.80, "#ef4444"],  # Crimson Critical
        [1.00, "#7f1d1d"],  # Burgundy Failure
    ]

    # Choose surfacecolor & colorscale
    if "Geological" in render_mode:
        surface_color = Z
        cur_cmin, cur_cmax = 65.0, 370.0
        cur_cscale = lithology_colorscale
        cbar_title = "Elevation (m RL)"
    else:
        surface_color = radar_vel_grid
        cur_cmin, cur_cmax = 0.0, 20.0
        cur_cscale = radar_colorscale
        cbar_title = "Radar Velocity (mm/h)"

    # --- 3. ADD MAIN 3D OPEN-PIT TERRAIN MESH ---
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=surface_color,
        cmin=cur_cmin, cmax=cur_cmax,
        colorscale=cur_cscale,
        customdata=custom_data,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Elevation: %{customdata[1]}<br>"
            "Deformation Velocity: %{customdata[2]}<br>"
            "%{customdata[3]}"
            "<extra></extra>"
        ),
        showscale=True,
        colorbar=dict(
            title=dict(text=cbar_title, font=dict(color="#cbd5e1", size=10)),
            tickfont=dict(color="#94a3b8", size=9),
            thickness=12,
            len=0.68,
            x=0.98,
        ),
        lighting=dict(ambient=0.58, diffuse=0.85, roughness=0.88, specular=0.18),
    ))

    # --- 4. 3D EQUIPMENT & PHYSICAL SENSOR HARDWARE ---
    # A) SSR-02 Ground-Based Radar Scanner Trailer (South Crest Scanner Pad)
    radar_pos = (130.0, -225.0, 318.0)
    fig.add_trace(go.Scatter3d(
        x=[radar_pos[0]], y=[radar_pos[1]], z=[radar_pos[2]],
        mode="markers+text",
        text=["📡 SSR-02 RADAR TRAILER"],
        textposition="top center",
        textfont=dict(color="#38bdf8", size=10, family="monospace"),
        marker=dict(size=9, color="#38bdf8", symbol="diamond", line=dict(color="#ffffff", width=1.5)),
        name="SSR-02 Radar Station",
        hoverinfo="text",
        hovertext="<b>Slope Stability Radar SSR-02</b><br>Azimuth: 342° | Line-of-Sight Range: 640m<br>Scan Frequency: Ku-band (17.2 GHz)<br>Status: ONLINE (94 dB)",
    ))

    # B) Translucent 3D Radar Raycast Scan Fan Beam (sweeping North Highwall)
    target_pos = (cx, cy, 248.0)
    fan_left = (cx - 55, cy - 15, 252.0)
    fan_right = (cx + 55, cy + 15, 244.0)
    ray_x = [radar_pos[0], fan_left[0], None, radar_pos[0], target_pos[0], None, radar_pos[0], fan_right[0]]
    ray_y = [radar_pos[1], fan_left[1], None, radar_pos[1], target_pos[1], None, radar_pos[1], fan_right[1]]
    ray_z = [radar_pos[2], fan_left[2], None, radar_pos[2], target_pos[2], None, radar_pos[2], fan_right[2]]
    
    fig.add_trace(go.Scatter3d(
        x=ray_x, y=ray_y, z=ray_z,
        mode="lines",
        line=dict(color="rgba(56, 189, 248, 0.45)", width=2.5, dash="dash"),
        name="SSR Radar Line-of-Sight Raycasts",
        hoverinfo="none",
        showlegend=False,
    ))

    # C) Vibrating Wire Piezometers (Vertical Borehole Strings PZ-01..04)
    pz_stations = [
        {"id": "PZ-01", "x": 18.0, "y": 190.0, "z_top": 270.0, "z_bot": 215.0, "pore": sim_telemetries["ZONE_NORTH_WALL"]["pore_pressure_kpa"]},
        {"id": "PZ-02", "x": -22.0, "y": -185.0, "z_top": 268.0, "z_bot": 218.0, "pore": sim_telemetries["ZONE_SOUTH_CREST"]["pore_pressure_kpa"]},
        {"id": "PZ-03", "x": 165.0, "y": 15.0, "z_top": 225.0, "z_bot": 178.0, "pore": sim_telemetries["ZONE_RAMP_3"]["pore_pressure_kpa"]},
        {"id": "PZ-04", "x": -165.0, "y": 15.0, "z_top": 225.0, "z_bot": 180.0, "pore": sim_telemetries["ZONE_HAUL_ROAD"]["pore_pressure_kpa"]},
    ]
    pz_lines_x, pz_lines_y, pz_lines_z = [], [], []
    pz_pts_x, pz_pts_y, pz_pts_z, pz_texts = [], [], [], []
    for pz in pz_stations:
        pz_lines_x.extend([pz["x"], pz["x"], None])
        pz_lines_y.extend([pz["y"], pz["y"], None])
        pz_lines_z.extend([pz["z_top"], pz["z_bot"], None])
        pz_pts_x.append(pz["x"])
        pz_pts_y.append(pz["y"])
        pz_pts_z.append(pz["z_bot"])
        pz_texts.append(f"<b>Borehole Piezometer {pz['id']}</b><br>Pore Pressure: {pz['pore']:.0f} kPa<br>Water Table: {pz['z_bot']:.0f}m RL")

    fig.add_trace(go.Scatter3d(
        x=pz_lines_x, y=pz_lines_y, z=pz_lines_z,
        mode="lines",
        line=dict(color="#06b6d4", width=3.5),
        name="Piezometer Boreholes",
        hoverinfo="none",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=pz_pts_x, y=pz_pts_y, z=pz_pts_z,
        mode="markers",
        marker=dict(size=6, color="#06b6d4", symbol="circle", line=dict(color="#ffffff", width=1.0)),
        hoverinfo="text",
        hovertext=pz_texts,
        name="Piezometers (Water Table)",
    ))

    # D) In-Place Inclinometers (IPI-01 Shear Casing)
    fig.add_trace(go.Scatter3d(
        x=[28.0, 28.0], y=[170.0, 170.0], z=[265.0, 225.0],
        mode="lines+markers",
        line=dict(color="#f59e0b", width=3.5),
        marker=dict(size=5, color="#f59e0b"),
        hoverinfo="text",
        hovertext="<b>In-Place Inclinometer IPI-01</b><br>Subsurface Shear Slip: 14.6 mm @ 32m depth<br>Status: Active Shear Discontinuity",
        name="In-Place Inclinometers",
        showlegend=False,
    ))

    # E) Surface Tension Crack Wireline Extensometer (EXT-01)
    if sim_telemetries[chosen_sector].get("crack_cm", 0.0) > 0.1:
        fig.add_trace(go.Scatter3d(
            x=[32.0, 42.0], y=[205.0, 215.0], z=[292.0, 294.0],
            mode="lines+markers",
            line=dict(color="#ef4444", width=4.5),
            marker=dict(size=6, color="#ef4444", symbol="square"),
            hoverinfo="text",
            hovertext=f"<b>Wireline Extensometer EXT-01</b><br>Crest Tension Fissure: {sim_telemetries[chosen_sector]['crack_cm']:.1f} cm<br>Dilation Rate: High",
            name="Crest Extensometer",
            showlegend=False,
        ))

    # F) Scaled Mining Haul Trucks (Haul Road Navigation)
    fig.add_trace(go.Scatter3d(
        x=[120.0, -115.0], y=[110.0, -75.0], z=[198.0, 235.0],
        mode="markers+text",
        text=["🚛 HT-04 (CAT 797)", "🚛 HT-09 (CAT 797)"],
        textposition="top right",
        textfont=dict(color="#fbbf24", size=9, family="monospace"),
        marker=dict(size=7, color="#f59e0b", symbol="cross"),
        hoverinfo="text",
        hovertext=["<b>Haul Truck HT-04</b><br>Payload: 360T Copper Ore<br>Speed: 18 km/h Ascending", "<b>Haul Truck HT-09</b><br>Payload: Empty<br>Speed: 24 km/h Descending"],
        name="Active Haulage Equipment",
    ))

    # --- 5. CAMERA PRESETS ---
    camera_presets = {
        "Aerial 3D Overview": dict(eye=dict(x=1.35, y=-1.45, z=0.95), center=dict(x=0, y=0, z=-0.1)),
        "North Highwall Hazard Focus": dict(eye=dict(x=0.35, y=1.25, z=0.65), center=dict(x=0.05, y=0.45, z=-0.1)),
        "Haul Ramp Switchback": dict(eye=dict(x=1.45, y=0.35, z=0.75), center=dict(x=0.1, y=0.0, z=-0.2)),
        "SSR Radar Line-of-Sight": dict(eye=dict(x=0.45, y=-1.45, z=0.85), center=dict(x=0.05, y=0.35, z=-0.15)),
        "Top-Down Geotechnical Plan": dict(eye=dict(x=0.01, y=0.01, z=2.25), center=dict(x=0, y=0, z=-0.1)),
    }
    cur_cam = camera_presets.get(camera_view, camera_presets["Aerial 3D Overview"])

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(font=dict(color="#94a3b8", size=9), orientation="h", y=1.02, x=0.0),
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
            aspectratio=dict(x=1.15, y=1.15, z=0.48),
            camera=cur_cam,
        ),
        height=570,
    )
    return fig


# ==============================================================================
# GEOTECHNICAL 2D CROSS-SECTION & FAILURE ARC (FoS)
# ==============================================================================
def build_geotech_cross_section_figure(selected_sector_name: str, hazard_mode: str, cur_sec: dict):
    fig = go.Figure()

    # Highwall profile steps
    x_prof = [0, 45, 45, 75, 75, 105, 105, 135, 135, 165, 165, 230]
    z_prof = [320, 320, 296, 296, 272, 272, 248, 248, 224, 224, 200, 200]
    fig.add_trace(go.Scatter(
        x=x_prof, y=z_prof,
        mode="lines",
        name="Highwall Face & Catch Berms",
        line=dict(color="#94a3b8", width=3),
        fill="tozeroy",
        fillcolor="rgba(30, 41, 59, 0.35)",
    ))

    # Tension crack at crest
    crack_depth = cur_sec.get("crack_cm", 0.0) / 10.0  # scaled to meters
    if crack_depth > 0.1:
        crack_x = 42
        fig.add_trace(go.Scatter(
            x=[crack_x, crack_x], y=[320, 320 - crack_depth],
            mode="lines+markers",
            name=f"Tension Crack ({cur_sec.get('crack_cm'):.1f} cm)",
            line=dict(color="#ef4444", width=4),
            marker=dict(size=6, color="#ef4444"),
        ))

    # Phreatic water surface line
    fig.add_trace(go.Scatter(
        x=[0, 165], y=[295, 215],
        mode="lines",
        name="Phreatic Water Table",
        line=dict(color="#38bdf8", dash="dash", width=2),
    ))

    # Bishop Critical Failure Slip Arc
    theta = np.linspace(0.85, 2.15, 50)
    r_arc = 125.0
    xc, zc = 125.0, 325.0
    x_arc = xc - r_arc * np.cos(theta)
    z_arc = zc - r_arc * np.sin(theta)

    arc_colors = {"NORMAL": "#10b981", "MILD": "#f59e0b", "MODERATE": "#f97316", "SEVERE": "#ef4444"}
    fos_values = {"NORMAL": 1.62, "MILD": 1.34, "MODERATE": 1.08, "SEVERE": 0.86}
    cur_fos = fos_values.get(hazard_mode, 1.62)

    fig.add_trace(go.Scatter(
        x=x_arc, y=z_arc,
        mode="lines",
        name=f"Critical Slip Surface (FoS: {cur_fos:.2f})",
        line=dict(color=arc_colors.get(hazard_mode, "#10b981"), width=3.5, dash="dot"),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.7)",
        title=dict(
            text=f"📐 Geotechnical Limit-Equilibrium Cross-Section: {selected_sector_name} (FoS = {cur_fos:.2f})",
            font=dict(color="#f1f5f9", size=13),
        ),
        xaxis=dict(title="Section Distance (m)", gridcolor="#1e293b", color="#94a3b8"),
        yaxis=dict(title="Elevation (m RL)", gridcolor="#1e293b", color="#94a3b8", range=[180, 340]),
        legend=dict(font=dict(color="#cbd5e1", size=10), orientation="h", y=1.12),
        margin=dict(l=40, r=20, b=40, t=50),
        height=570,
    )
    return fig


# ==============================================================================
# SAITO TERTIARY CREEP INVERSE VELOCITY (1/v) FORECAST
# ==============================================================================
def build_saito_forecast_figure(hazard_mode: str, cur_vel: float):
    t = np.linspace(0, 12, 50)
    tf = 14.5 if hazard_mode in ("MILD", "MODERATE") else (12.4 if hazard_mode == "SEVERE" else 30.0)
    v0 = 0.8
    A = 3.5 if hazard_mode != "NORMAL" else 0.2
    v = v0 + A / np.maximum(0.1, (tf - t))**1.1
    inv_v = 1.0 / np.maximum(0.01, v)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=v,
        mode="lines",
        name="Radar Creep Velocity v(t) [mm/h]",
        line=dict(color="#f97316", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=t, y=inv_v,
        mode="lines+markers",
        name="Inverse Velocity 1/v [h/mm]",
        yaxis="y2",
        line=dict(color="#38bdf8", width=2),
        marker=dict(size=4),
    ))

    time_to_fail = max(0.0, tf - 12.0)
    fail_label = f"Estimated Detachment in {time_to_fail:.1f} hours" if hazard_mode in ("MILD", "MODERATE") else ("IMMINENT DETACHMENT / COLLAPSE UNDERWAY" if hazard_mode == "SEVERE" else "Slope Stable — Normal Creep")

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.7)",
        title=dict(
            text=f"📈 Saito Inverse Velocity (1/v) Creep Law: {fail_label}",
            font=dict(color="#f1f5f9", size=12),
        ),
        xaxis=dict(title="Time (hours)", gridcolor="#1e293b", color="#94a3b8"),
        yaxis=dict(title="Velocity v (mm/h)", gridcolor="#1e293b", color="#f97316"),
        yaxis2=dict(title="1/v (h/mm)", overlaying="y", side="right", color="#38bdf8"),
        legend=dict(font=dict(color="#cbd5e1", size=9), orientation="h", y=1.15),
        margin=dict(l=35, r=35, b=35, t=45),
        height=220,
    )
    return fig


# ==============================================================================
# SENSOR TIME-SERIES TREND CHART (LEFT WING)
# ==============================================================================
def build_sensor_trend_chart(cur_sec: dict, hazard_mode: str):
    """
    Renders 12-hour synchronized multi-sensor trend (Radar mm/h, Pore kPa, Rain mm).
    """
    t = np.linspace(0, 12, 25)
    base_v = cur_sec["radar_velocity_mm_hr"]
    base_pore = cur_sec["pore_pressure_kpa"]
    base_rain = cur_sec["rainfall_3h_mm"]

    # Trajectories
    mult = 1.0 if hazard_mode == "NORMAL" else (1.8 if hazard_mode == "MILD" else (3.5 if hazard_mode == "MODERATE" else 8.0))
    v_hist = np.clip(0.4 + (base_v - 0.4) * (t / 12.0)**1.8, 0.2, 35.0)
    pore_hist = np.clip(20.0 + (base_pore - 20.0) * (t / 12.0)**1.2, 10.0, 180.0)
    rain_hist = np.clip(base_rain * (t / 12.0)**1.4, 0.0, 90.0)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=t, y=v_hist, name="Radar Velocity (mm/h)", line=dict(color="#38bdf8", width=2.2)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=t, y=pore_hist, name="Pore Pressure (kPa)", line=dict(color="#f59e0b", width=1.8, dash="dot")),
        secondary_y=True,
    )
    fig.add_trace(
        go.Bar(x=t, y=rain_hist, name="Rain 3h (mm)", marker_color="rgba(14, 165, 233, 0.25)", width=0.3),
        secondary_y=False,
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.65)",
        title=dict(text="12-Hour Multi-Sensor Historical Trend", font=dict(color="#cbd5e1", size=11)),
        xaxis=dict(title="Simulated Time (h)", gridcolor="#1e293b", color="#64748b"),
        yaxis=dict(title="Velocity (mm/h) / Rain (mm)", gridcolor="#1e293b", color="#38bdf8"),
        yaxis2=dict(title="Pore (kPa)", color="#f59e0b", showgrid=False),
        legend=dict(font=dict(color="#94a3b8", size=9), orientation="h", y=1.2),
        margin=dict(l=35, r=35, b=30, t=35),
        height=190,
    )
    return fig


# ==============================================================================
# SYSTEM STATE & INITIALIZATION
# ==============================================================================
if "bb" not in st.session_state:
    st.session_state.bb = MineBlackboard()
    st.session_state.ctrl = RockWatchController(st.session_state.bb)
    st.session_state.sim_engine = GeotechnicalSimulationEngine(time_horizon_hours=24.0, step_hours=0.5)
    st.session_state.hazard_mode = "MILD"  # Default scenario mode
    st.session_state.sim_time_hour = 12.0
    st.session_state.user_role = "OPERATOR"
    st.session_state.selected_sector = "ZONE_NORTH_WALL"
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": "RockWatch Geotechnical Safety Copilot active. Monitoring 4 pit sectors across primary radar, piezometers, inclinometers, and extensometers. All 8 safety agents are synchronized with the blackboard."
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
sim_engine: GeotechnicalSimulationEngine = st.session_state.sim_engine

# --- HAZARD SEVERITY PRESETS (USER-REQUESTED SEVERITY SELECTOR) ---
HAZARD_PRESETS = {
    "NORMAL": {
        "name": "🟢 1. Normal Operations (Stable Highwall)",
        "desc": "Baseline mining operations. Benches stable, background radar creep (0.75 mm/hr), Level 1 Green.",
        "vel": 0.75,
        "rain": 1.2,
        "pore": 24.0,
        "ppv": 0.35,
        "crack_cm": 0.0,
        "status": "GREEN",
        "directive": "Routine automated monitoring. Normal mining operations authorized across all benches.",
    },
    "MILD": {
        "name": "🟡 2. Mild Slope Ravelling & Loose Stones",
        "desc": "Hairline crest tension crack opens (2.4 cm). Small gravel rolling down face. Radar creep accelerates to 2.85 mm/hr. Level 2 Yellow Caution.",
        "vel": 2.85,
        "rain": 4.5,
        "pore": 42.0,
        "ppv": 0.85,
        "crack_cm": 2.4,
        "status": "YELLOW",
        "directive": "Heightened caution. Restrict non-essential foot traffic; alert pit supervisor; increase scan rate.",
    },
    "MODERATE": {
        "name": "🟠 3. Moderate Wedge Fissure Dilation",
        "desc": "Scarp fissure dilates to 8.2 cm. Benches exhibit slab shearing. Radar creep reaches 7.85 mm/hr. Level 3 Orange Warning.",
        "vel": 7.85,
        "rain": 18.0,
        "pore": 72.0,
        "ppv": 2.4,
        "crack_cm": 8.2,
        "status": "ORANGE",
        "directive": "Warning status. Stand down haulage under highwall; dispatch geotechnical inspection crew; 150m exclusion zone.",
    },
    "SEVERE": {
        "name": "🔴 4. Severe Detachment & Massive Landslide",
        "desc": "Major scarp collapse. Tension fissure exceeds 18 cm. Radar tertiary runaway at 22.40 mm/hr. Level 4 Red Emergency.",
        "vel": 22.40,
        "rain": 62.0,
        "pore": 135.0,
        "ppv": 6.8,
        "crack_cm": 18.5,
        "status": "RED",
        "directive": "CRITICAL EMERGENCY: Mandatory immediate pit evacuation. Sound sirens. Evacuate all personnel within 500m radius.",
    },
}

# Sector Cam mappings
SECTOR_CAMS = {
    "ZONE_NORTH_WALL": "CAM-01 [North Highwall Bench 4]",
    "ZONE_SOUTH_CREST": "CAM-02 [South Crest Overburden]",
    "ZONE_RAMP_3": "CAM-03 [Main Haulage Switchback Ramp 3]",
    "ZONE_HAUL_ROAD": "CAM-04 [West Pit Access Haul Road]",
}

# --- COMPUTE CURRENT MULTI-SECTOR STATE ---
active_mode = st.session_state.hazard_mode
preset = HAZARD_PRESETS[active_mode]
selected_sector_key = st.session_state.get("selected_sector", "ZONE_NORTH_WALL")

sim_telemetries = {
    "ZONE_NORTH_WALL": {
        "name": "North Wall Bench 4",
        "radar_velocity_mm_hr": preset["vel"] if selected_sector_key == "ZONE_NORTH_WALL" else 0.8,
        "rainfall_3h_mm": preset["rain"] if selected_sector_key == "ZONE_NORTH_WALL" else 1.5,
        "pore_pressure_kpa": preset["pore"] if selected_sector_key == "ZONE_NORTH_WALL" else 25.0,
        "vibration_ppv_mms": preset["ppv"] if selected_sector_key == "ZONE_NORTH_WALL" else 0.4,
        "crack_cm": preset["crack_cm"] if selected_sector_key == "ZONE_NORTH_WALL" else 0.0,
        "effective_level": preset["status"] if selected_sector_key == "ZONE_NORTH_WALL" else "GREEN",
        "directive": preset["directive"] if selected_sector_key == "ZONE_NORTH_WALL" else "Routine monitoring.",
        "dominance_override": preset["status"] == "RED" if selected_sector_key == "ZONE_NORTH_WALL" else False,
    },
    "ZONE_SOUTH_CREST": {
        "name": "South Crest Ramp",
        "radar_velocity_mm_hr": preset["vel"] if selected_sector_key == "ZONE_SOUTH_CREST" else 0.5,
        "rainfall_3h_mm": preset["rain"] if selected_sector_key == "ZONE_SOUTH_CREST" else 1.0,
        "pore_pressure_kpa": preset["pore"] if selected_sector_key == "ZONE_SOUTH_CREST" else 22.0,
        "vibration_ppv_mms": preset["ppv"] if selected_sector_key == "ZONE_SOUTH_CREST" else 0.3,
        "crack_cm": preset["crack_cm"] if selected_sector_key == "ZONE_SOUTH_CREST" else 0.0,
        "effective_level": preset["status"] if selected_sector_key == "ZONE_SOUTH_CREST" else "GREEN",
        "directive": preset["directive"] if selected_sector_key == "ZONE_SOUTH_CREST" else "Routine monitoring.",
        "dominance_override": preset["status"] == "RED" if selected_sector_key == "ZONE_SOUTH_CREST" else False,
    },
    "ZONE_RAMP_3": {
        "name": "Main Haulage Ramp 3",
        "radar_velocity_mm_hr": preset["vel"] if selected_sector_key == "ZONE_RAMP_3" else 0.6,
        "rainfall_3h_mm": preset["rain"] if selected_sector_key == "ZONE_RAMP_3" else 0.8,
        "pore_pressure_kpa": preset["pore"] if selected_sector_key == "ZONE_RAMP_3" else 24.0,
        "vibration_ppv_mms": preset["ppv"] if selected_sector_key == "ZONE_RAMP_3" else 0.5,
        "crack_cm": preset["crack_cm"] if selected_sector_key == "ZONE_RAMP_3" else 0.0,
        "effective_level": preset["status"] if selected_sector_key == "ZONE_RAMP_3" else "GREEN",
        "directive": preset["directive"] if selected_sector_key == "ZONE_RAMP_3" else "Routine monitoring.",
        "dominance_override": preset["status"] == "RED" if selected_sector_key == "ZONE_RAMP_3" else False,
    },
    "ZONE_HAUL_ROAD": {
        "name": "West Pit Access Road",
        "radar_velocity_mm_hr": preset["vel"] if selected_sector_key == "ZONE_HAUL_ROAD" else 0.4,
        "rainfall_3h_mm": preset["rain"] if selected_sector_key == "ZONE_HAUL_ROAD" else 0.5,
        "pore_pressure_kpa": preset["pore"] if selected_sector_key == "ZONE_HAUL_ROAD" else 20.0,
        "vibration_ppv_mms": preset["ppv"] if selected_sector_key == "ZONE_HAUL_ROAD" else 0.25,
        "crack_cm": preset["crack_cm"] if selected_sector_key == "ZONE_HAUL_ROAD" else 0.0,
        "effective_level": preset["status"] if selected_sector_key == "ZONE_HAUL_ROAD" else "GREEN",
        "directive": preset["directive"] if selected_sector_key == "ZONE_HAUL_ROAD" else "Routine monitoring.",
        "dominance_override": preset["status"] == "RED" if selected_sector_key == "ZONE_HAUL_ROAD" else False,
    },
}

# Run controller cycle on active sector
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

badge_classes = {
    "GREEN": "badge-green",
    "YELLOW": "badge-yellow",
    "ORANGE": "badge-orange",
    "RED": "badge-red",
}
global_badge_class = badge_classes.get(highest_global_level, "badge-green")
head_hash = bb.get_head_hash()[:12] + "..."


# ==============================================================================
# 1. TOP MISSION CONTROL HUD HEADER
# ==============================================================================
hud_html = f"""
<div class="hud-header">
    <div class="hud-title">
        <span>⚡</span> ROCKWATCH // 3D GEOTECHNICAL MISSION CONTROL DIGITAL TWIN
    </div>
    <div style="display: flex; align-items: center; gap: 20px;">
        <div style="display: flex; align-items: center; gap: 8px; font-size: 0.85rem; color: #94a3b8;">
            <span class="pulse-dot"></span> 8 AGENTS SYNCHRONIZED
        </div>
        <div style="font-size: 0.82rem; color: #64748b; font-family: monospace;">
            LEDGER: <span style="color: #38bdf8;">{head_hash}</span>
        </div>
        <div style="font-size: 0.82rem; color: #64748b; font-family: monospace;">
            MODE: <span style="color: #f1f5f9; font-weight: 700;">{active_mode} SCENARIO</span>
        </div>
        <div>
            <span class="{global_badge_class}">GLOBAL STATUS: LEVEL {level_rank[highest_global_level]} {highest_global_level}</span>
        </div>
    </div>
</div>
"""
st.markdown(hud_html, unsafe_allow_html=True)


# ==============================================================================
# 2. MAIN COCKPIT VIEWPORT (SENSORS ON LEFT | LIVE MODEL IN THE MIDDLE | TARP ON RIGHT)
# ==============================================================================
col_left, col_mid, col_right = st.columns([1.0, 1.95, 1.05], gap="small")


# ------------------------------------------------------------------------------
# LEFT WING: REAL-TIME SENSOR INSTRUMENTATION & TELEMETRY ARRAY
# ------------------------------------------------------------------------------
with col_left:
    st.markdown('<div class="card-title">📡 Sensor Instrumentation Array</div>', unsafe_allow_html=True)

    # Sector Selector Radio Pills
    sec_names = {k: v["name"] for k, v in sim_telemetries.items()}
    chosen_sector = st.radio(
        "Active Pit Sector Target:",
        options=list(sec_names.keys()),
        format_func=lambda k: f"{sec_names[k]} ({sim_telemetries[k]['effective_level']})",
        horizontal=True,
        key="selected_sector",
    )
    cur_sec = sim_telemetries[chosen_sector]
    sec_level = cur_sec["effective_level"]
    sec_badge_class = badge_classes.get(sec_level, "badge-green")

    # Multimodal Sensor Metric Boxes
    st.markdown(f"""
    <div class="mission-card" style="padding: 10px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-weight: 700; font-size: 0.9rem; color: #f1f5f9;">{cur_sec['name']}</span>
            <span class="{sec_badge_class}">{sec_level}</span>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
            <div class="metric-box">
                <div class="metric-box-val">{cur_sec['radar_velocity_mm_hr']:.2f}</div>
                <div class="metric-box-label">SSR Radar (mm/h)</div>
            </div>
            <div class="metric-box">
                <div class="metric-box-val">{cur_sec['pore_pressure_kpa']:.0f}</div>
                <div class="metric-box-label">Piezometer (kPa)</div>
            </div>
            <div class="metric-box">
                <div class="metric-box-val">{cur_sec['crack_cm']:.1f}</div>
                <div class="metric-box-label">Crack Meter (cm)</div>
            </div>
            <div class="metric-box">
                <div class="metric-box-val">{cur_sec['vibration_ppv_mms']:.2f}</div>
                <div class="metric-box-label">Seismic PPV (mm/s)</div>
            </div>
            <div class="metric-box">
                <div class="metric-box-val">{cur_sec['rainfall_3h_mm']:.1f}</div>
                <div class="metric-box-label">Rain 3h (mm)</div>
            </div>
            <div class="metric-box">
                <div class="metric-box-val">{(1.0 / max(0.01, cur_sec['radar_velocity_mm_hr'])):.2f}</div>
                <div class="metric-box-label">Inv Vel 1/v (h/mm)</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Historical Sensor Trend Chart
    st.plotly_chart(build_sensor_trend_chart(cur_sec, active_mode), width="stretch", config={"displayModeBar": False})

    # Sensor Array Operational Diagnostics Matrix
    st.markdown('<div class="card-title" style="margin-top: 6px;">🩺 Sensor Diagnostic Health</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.76rem; font-family: monospace; background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(59, 130, 246, 0.18); border-radius: 8px; padding: 8px 10px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
            <span>🟢 SSR-02 Radar</span><span style="color: #38bdf8;">ONLINE (94 dB)</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
            <span>🟢 PZ-01/04 Piezos</span><span style="color: #34d399;">ONLINE (24V OK)</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
            <span>🟢 IPI-01 Inclinometer</span><span style="color: #34d399;">CALIBRATED</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
            <span>🟢 EXT-01 Extensometer</span><span style="color: #34d399;">TENSION NOMINAL</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>🟢 MET-01 Weather</span><span style="color: #38bdf8;">SOLAR 13.8V</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# CENTER STAGE: THE REALISTIC 3D MINE DIGITAL TWIN (HERO IN THE MIDDLE)
# ------------------------------------------------------------------------------
with col_mid:
    # 3D Digital Twin Viewport Title & Controls
    mid_top_c1, mid_top_c2 = st.columns([1.1, 1.0])
    with mid_top_c1:
        st.markdown('<div class="card-title">🌐 Realistic 3D Open-Pit Digital Twin</div>', unsafe_allow_html=True)
    with mid_top_c2:
        render_mode = st.selectbox(
            "Display Layer Mode:",
            options=[
                "🛰️ Realistic Geological Strata (True Rock Textures DEM)",
                "📡 Dynamic InSAR / SSR Radar Displacement Heatmap",
            ],
            key="mine_render_mode",
            label_visibility="collapsed",
        )

    # Scenario Selector Pills (Trigger Hazard Severity)
    mode_options = list(HAZARD_PRESETS.keys())
    selected_mode = st.radio(
        "Trigger Hazard Severity Level:",
        options=mode_options,
        format_func=lambda m: HAZARD_PRESETS[m]["name"],
        horizontal=True,
        key="hazard_mode",
    )

    # Camera Preset Radio Pills
    cam_preset = st.radio(
        "3D Camera Perspective:",
        options=["Aerial 3D Overview", "North Highwall Hazard Focus", "Haul Ramp Switchback", "SSR Radar Line-of-Sight", "Top-Down Geotechnical Plan"],
        horizontal=True,
        key="camera_perspective_choice",
    )

    # Generate and Render the Realistic 3D Model
    X_grid, Y_grid, Z_grid, s_map, R_pit = generate_realistic_pit_topography(n_grid=100)
    fig_3d = build_realistic_3d_mine_model(
        X_grid, Y_grid, Z_grid, s_map, R_pit,
        sim_telemetries,
        chosen_sector,
        selected_mode,
        render_mode,
        cam_preset,
    )
    st.plotly_chart(fig_3d, width="stretch", config={"displayModeBar": False})

    # Sector Inspection HUD Banner
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 8px; padding: 7px 14px; margin-top: -6px; font-size: 0.80rem; font-family: monospace;">
        <div>TARGET: <span style="color: #f1f5f9; font-weight: 700;">{cur_sec['name'].upper()}</span></div>
        <div>COORDINATES: <span style="color: #38bdf8;">E: 145.2m | N: 182.4m | Z: 248.0m RL</span></div>
        <div>TARP: <span class="{sec_badge_class}">{sec_level}</span></div>
        <div>SSR VELOCITY: <span style="color: #f97316; font-weight: 700;">{cur_sec['radar_velocity_mm_hr']:.2f} mm/h</span></div>
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# RIGHT WING: TARP DECISION ENGINE, SLOPE STABILITY & MULTI-AGENT COPILOT
# ------------------------------------------------------------------------------
with col_right:
    st.markdown('<div class="card-title">🛡️ TARP Early Warning & Decision Engine</div>', unsafe_allow_html=True)

    # Active TARP Action Directive Card
    st.markdown(f"""
    <div class="mission-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <span style="font-weight: 800; font-size: 1.05rem; color: #f1f5f9;">{cur_sec['name']}</span>
            <span class="{sec_badge_class}">{sec_level}</span>
        </div>
        <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.4;">
            <strong>TARP Action Directive:</strong> {cur_sec['directive']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Dominance Override Alert Card
    if cur_sec["dominance_override"] or sec_level == "RED":
        st.markdown("""
        <div style="background: rgba(239, 68, 68, 0.22); border: 1px solid #ef4444; border-radius: 8px; padding: 8px 12px; margin-bottom: 10px;">
            <span style="color: #f87171; font-weight: 700; font-size: 0.82rem;">⚡ DOMINANCE OVERRIDE ACTIVATED</span><br>
            <span style="color: #cbd5e1; font-size: 0.75rem;">Radar velocity breached 15.0 mm/h safety floor. Alert escalated to RED without multi-sensor dilution.</span>
        </div>
        """, unsafe_allow_html=True)

    # Geotechnical Slope Stability (Bishop FoS & Saito Creep)
    fos_values = {"NORMAL": 1.62, "MILD": 1.34, "MODERATE": 1.08, "SEVERE": 0.86}
    cur_fos = fos_values.get(selected_mode, 1.62)
    fos_color = "#10b981" if cur_fos >= 1.5 else ("#f59e0b" if cur_fos >= 1.2 else ("#f97316" if cur_fos >= 1.0 else "#ef4444"))
    
    st.markdown(f"""
    <div style="display: flex; gap: 8px; margin-bottom: 8px;">
        <div class="metric-box" style="flex: 1;">
            <div class="metric-box-val" style="color: {fos_color};">{cur_fos:.2f}</div>
            <div class="metric-box-label">Bishop FoS</div>
        </div>
        <div class="metric-box" style="flex: 1.5;">
            <div class="metric-box-val" style="font-size: 0.95rem; color: {'#34d399' if cur_fos >= 1.5 else ('#fbbf24' if cur_fos >= 1.2 else '#f87171')};">
                {'STABLE' if cur_fos >= 1.5 else ('CAUTION' if cur_fos >= 1.2 else ('MARGINAL' if cur_fos >= 1.0 else 'UNSTABLE / FAILURE'))}
            </div>
            <div class="metric-box-label">Slope Stability State</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Saito Inverse Velocity 1/v Creep Forecast
    st.plotly_chart(build_saito_forecast_figure(selected_mode, cur_sec["radar_velocity_mm_hr"]), width="stretch", config={"displayModeBar": False})

    # Interactive Multi-Agent Safety Copilot
    st.markdown('<div class="card-title" style="margin-top: 6px;">💬 Geotechnical AI Copilot</div>', unsafe_allow_html=True)
    chat_container = st.container(height=180)
    with chat_container:
        for msg in st.session_state.chat_history[-4:]:
            role_icon = "🤖" if msg["role"] == "assistant" else "👤"
            st.markdown(f"**{role_icon} {msg['role'].upper()}:** {msg['content']}")

    user_query = st.chat_input("Query sensor readings, SOP actions, or risk...")
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        sop_info = retrieve_sop(sec_level)
        q_l = user_query.lower()

        if "why" in q_l or "reason" in q_l or "alert" in q_l or "trigger" in q_l:
            response = (
                f"**Multi-Agent Assessment for {cur_sec['name']}:**\n\n"
                f"- **Active Status:** **{sec_level}**\n"
                f"- **Sensor Readout:** Radar creep is **{cur_sec['radar_velocity_mm_hr']:.2f} mm/h**, "
                f"pore pressure is **{cur_sec['pore_pressure_kpa']:.0f} kPa**, and 3h rainfall is **{cur_sec['rainfall_3h_mm']:.1f} mm**.\n"
                f"- **Geotechnical Status:** Factor of Safety is **{cur_fos:.2f}**; tension fissure is at **{cur_sec['crack_cm']:.1f} cm**.\n"
                f"- **Grounded SOP Rule:** {sop_info.get('passage_id', '[TARP-SOP-01]')}: *{sop_info.get('action', 'Routine monitoring.')}*"
            )
        elif "evacuat" in q_l or "sop" in q_l or "action" in q_l:
            response = (
                f"**Mandatory Protocol for {sec_level} Status:**\n\n"
                f"**Citation:** `{sop_info.get('passage_id', '[TARP-SOP-01]')}`\n"
                f"{sop_info.get('text', 'Normal monitoring.')}\n\n"
                f"**Operational Directives:**\n"
                f"1. Acknowledge notice and notify pit geotechnical engineer.\n"
                f"2. Radio mobile equipment operators in {cur_sec['name']}.\n"
                f"3. {'MANDATORY PIT EVACUATION: Clear all personnel from pit floor within 500m radius immediately.' if sec_level == 'RED' else 'Inspect crest tension cracks and verify piezometers.'}"
            )
        elif "sensor" in q_l or "health" in q_l or "radar" in q_l:
            response = (
                f"**Sensor Health Audit (Agent 1 - HealthAgent):**\n\n"
                f"- Slope Stability Radar (SSR-02): ONLINE | Signal: 94 dB | Velocity: {cur_sec['radar_velocity_mm_hr']:.2f} mm/h\n"
                f"- Piezometer (PZ-01): ONLINE | Pore Pressure: {cur_sec['pore_pressure_kpa']:.0f} kPa\n"
                f"- Inclinometer (IPI-01): ONLINE | Shear Slip: 14.6 mm @ 32m depth\n"
                f"- Weather Station (MET-01): ONLINE | Precipitation: {cur_sec['rainfall_3h_mm']:.1f} mm / 3h"
            )
        else:
            response = (
                f"**RockWatch Copilot:** {cur_sec['name']} is currently at **LEVEL {level_rank[sec_level]} {sec_level}**. "
                f"Governing citation is `{sop_info.get('passage_id', '[TARP-SOP-01]')}`: {sop_info.get('text', 'Routine monitoring.')}"
            )

        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()


# ==============================================================================
# 3. BOTTOM UTILITY DOCK: GEOTECHNICAL AUTHORIZATION CONSOLE & AUDIT LEDGER
# ==============================================================================
st.markdown("---")
b_c1, b_c2 = st.columns([1.1, 0.9], gap="large")

with b_c1:
    st.markdown('<div class="card-title">⚖️ Enterprise Geotechnical Authorization & Desensitization Console</div>', unsafe_allow_html=True)
    
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
        st.dataframe(pd.DataFrame(recent_entries), width="stretch", hide_index=True)
