"""
app.py - RockWatch Mission-Control: 3D Geotechnical Digital Twin & Monitoring Center.

Fully customized industrial mission-control interface:
- High-tech dark aerospace command gradient with zero Streamlit chrome (headers/toolbars hidden).
- Stepped Open-Pit 3D Digital Twin with explicit spiral haul road, bench faces, and 3D floating sector station markers.
- Built-in Real-Life CCTV Surveillance MP4 Videos for each hazard stage (Normal, Mild Ravelling, Moderate Slabbing, Severe Collapse).
- Dynamic AI Detection Overlays tracking rockfall velocity, crack aperture dilation, and bounding boxes.
- Intuitive Scenario & Progressive Timeline Selection ("All Together" in lockstep).
- Interactive Conversational Multi-Agent Safety Copilot (RAG-grounded in TARP SOPs).
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
from PIL import Image
import inspect

import streamlit as st

def safe_render_video(data, key=None, **kwargs):
    """
    Renders st.video with unique key/alt parameters compatible with all Streamlit versions (Python 3.13, 3.14).
    Prevents StreamlitDuplicateElementId while remaining robust across varying parameter signatures.
    """
    sig = inspect.signature(st.video)
    call_kwargs = dict(kwargs)
    if "key" in sig.parameters and key is not None:
        call_kwargs["key"] = key
    if "alt" in sig.parameters and key is not None:
        call_kwargs["alt"] = str(key)
    return st.video(data, **call_kwargs)

from blackboard import MineBlackboard
from controller import RockWatchController
from tools.rag_tools import retrieve_sop
from tools.simulation_engine import GeotechnicalSimulationEngine
from ml.video_analyzer import analyze_uploaded_video

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
    background: radial-gradient(circle at 50% 0%, #101c38 0%, #090e1a 50%, #03060c 100%) !important;
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

/* Chat container styling */
.chat-box {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 12px;
    padding: 12px 16px;
    height: 280px;
    overflow-y: auto;
    margin-bottom: 10px;
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


# --- REALISTIC ASYMMETRIC OPEN-PIT MINE 3D GEOMETRY ---
@st.cache_data
def generate_realistic_pit_topography(n_grid: int = 85):
    x = np.linspace(-380, 380, n_grid)
    y = np.linspace(-380, 380, n_grid)
    X, Y = np.meshgrid(x, y)

    # 1. Multi-Lobed Asymmetric Open-Pit Geometry (North Pit Lobe + South Expansion Lobe)
    r1 = np.sqrt(((X - 15) / 1.25)**2 + ((Y - 45) / 0.88)**2)
    r2 = np.sqrt(((X + 30) / 1.05)**2 + ((Y + 75) / 1.18)**2)
    k = 0.04
    # Smooth minimum blend of the two pit lobes
    R_pit = -np.log(np.exp(-k * r1) + np.exp(-k * r2)) / k

    R_crest = 280.0
    R_floor = 55.0
    Z_crest = 310.0
    Z_floor = 60.0
    bench_h = 24.0
    bench_w = 26.0

    # 2. Stepped Bench Terraces & Catch Berms
    dr = np.clip(R_crest - R_pit, 0.0, R_crest - R_floor)
    bench_idx = np.floor(dr / bench_w)
    frac = (dr % bench_w) / bench_w
    bench_drop = np.where(frac < 0.65, (frac / 0.65) * bench_h, bench_h)
    Z = Z_crest - (bench_idx * bench_h + bench_drop)
    Z[R_pit < R_floor] = Z_floor

    # 3. Surrounding Natural Mountain Ridges & Valleys outside Crest
    ridge = 32.0 * np.sin(X / 115.0) * np.cos(Y / 125.0) + 16.0 * np.sin((X + Y) / 85.0)
    crest_mask = R_pit >= R_crest
    blend = np.clip((R_pit - R_crest) / 45.0, 0.0, 1.0)
    Z[crest_mask] = Z_crest + (ridge * blend)[crest_mask]

    # 4. External Terraced Waste Dump (South-West Plateau)
    dump_dist = np.sqrt((X + 270)**2 + (Y + 230)**2)
    in_dump = (dump_dist < 110) & (X < -160) & (Y < -130)
    dump_lifts = np.floor(np.clip(110 - dump_dist, 0, 90) / 28) * 18.0
    Z[in_dump] = np.maximum(Z[in_dump], 305.0 + dump_lifts[in_dump])

    # 5. Spiral Switchback Haul Road (22m road width cut into pit face)
    Theta = np.arctan2(Y, X) % (2 * np.pi)
    ramp_radius = R_crest - (Theta / (2 * np.pi)) * (R_crest - R_floor)
    ramp_dist = np.abs(R_pit - ramp_radius)
    on_ramp = (ramp_dist < 15.0) & (R_pit < R_crest) & (R_pit > R_floor)
    ramp_elevation = Z_crest - (Theta / (2 * np.pi)) * (Z_crest - Z_floor)
    Z[on_ramp] = ramp_elevation[on_ramp]

    # Sector ID Assignment across pit domains
    sector_id_map = np.zeros_like(Z, dtype=int)
    sector_id_map[(Y > 40) & (R_pit < R_crest) & (R_pit > 60)] = 1   # North Wall
    sector_id_map[(Y < -50) & (R_pit < R_crest) & (R_pit > 60)] = 2  # South Crest
    sector_id_map[(X > 35) & (Y >= -40) & (Y <= 40) & (R_pit < R_crest)] = 3  # Ramp 3
    sector_id_map[(X < -45) & (Y >= -40) & (Y <= 40) & (R_pit < R_crest)] = 4  # Haul Road

    return X, Y, Z, sector_id_map


def build_3d_pit_figure(X, Y, Z, sector_id_map, sector_telemetries, selected_sector_key):
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
                custom_data[i, j, 0] = "Surrounding Crest / Waste Dump"
                custom_data[i, j, 1] = "0.40 mm/hr"
                custom_data[i, j, 2] = "GREEN"

    tarp_colorscale = [
        [0.00, "#059669"],   # Emerald Normal
        [0.08, "#10b981"],
        [0.22, "#f59e0b"],   # Amber Caution (1.5 - 5 mm/hr)
        [0.55, "#f97316"],   # Orange Warning (5 - 15 mm/hr)
        [0.80, "#ef4444"],   # Crimson Critical
        [1.00, "#991b1b"],   # Deep Red Failure
    ]

    fig = go.Figure()

    # 1. Main Multi-Lobe Stepped Pit Surface
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
        {"key": "ZONE_NORTH_WALL", "label": "📍 NORTH WALL BENCH 4", "x": 15, "y": 185, "z": 240},
        {"key": "ZONE_SOUTH_CREST", "label": "📍 SOUTH CREST RAMP", "x": -25, "y": -195, "z": 240},
        {"key": "ZONE_RAMP_3", "label": "📍 MAIN HAULAGE RAMP 3", "x": 175, "y": 0, "z": 200},
        {"key": "ZONE_HAUL_ROAD", "label": "📍 WEST PIT ACCESS ROAD", "x": -175, "y": 0, "z": 200},
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
        height=520,
    )
    return fig


def build_geotech_cross_section_figure(selected_sector_name: str, hazard_mode: str, cur_sec: dict):
    """
    Renders 2D engineered geotechnical cross-section showing highwall slope profile,
    tension crack depth, Bishop circular failure slip arc, water table, and Factor of Safety (FoS).
    """
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
        height=520,
    )
    return fig


def build_saito_forecast_figure(hazard_mode: str, cur_vel: float):
    """
    Renders Saito's tertiary creep inverse velocity (1/v) time-of-failure forecast.
    """
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

    # Countdown to failure
    time_to_fail = max(0.0, tf - 12.0)
    fail_label = f"Estimated Detachment in {time_to_fail:.1f} hours" if hazard_mode in ("MILD", "MODERATE") else ("IMMINENT DETACHMENT / COLLAPSE UNDERWAY" if hazard_mode == "SEVERE" else "Slope Stable — Normal Creep")

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.7)",
        title=dict(
            text=f"📈 Saito Inverse Velocity (1/v) Creep Law: {fail_label}",
            font=dict(color="#f1f5f9", size=13),
        ),
        xaxis=dict(title="Time (hours)", gridcolor="#1e293b", color="#94a3b8"),
        yaxis=dict(title="Velocity v (mm/h)", gridcolor="#1e293b", color="#f97316"),
        yaxis2=dict(title="Inverse Velocity 1/v (h/mm)", overlaying="y", side="right", color="#38bdf8"),
        legend=dict(font=dict(color="#cbd5e1", size=10), orientation="h", y=1.12),
        margin=dict(l=40, r=40, b=40, t=50),
        height=520,
    )
    return fig


def build_sensor_topology_figure(sim_telemetries: dict):
    """
    Renders 2D plan-view sensor network topology showing Slope Stability Radar scanner,
    piezometers, inclinometers, and robotic prism total stations.
    """
    fig = go.Figure()

    # Pit contour rings
    for r in [280, 220, 160, 100, 50]:
        theta = np.linspace(0, 2 * np.pi, 60)
        fig.add_trace(go.Scatter(
            x=r * 1.2 * np.cos(theta),
            y=r * 0.9 * np.sin(theta),
            mode="lines",
            line=dict(color="#334155", width=1, dash="dot"),
            showlegend=False,
            hoverinfo="none",
        ))

    # Radar scanner location (SSR-02 on south-east ridge)
    fig.add_trace(go.Scatter(
        x=[240], y=[-260],
        mode="markers+text",
        name="SSR-02 Ground Radar",
        text=["📡 SSR-02 RADAR (Scanning 360°)"],
        textposition="bottom right",
        textfont=dict(color="#38bdf8", size=11, family="monospace"),
        marker=dict(size=14, color="#38bdf8", symbol="star"),
    ))

    # Radar scan fan lines
    for angle in [75, 105, 135, 165]:
        rad = np.radians(angle)
        fig.add_trace(go.Scatter(
            x=[240, 240 - 450 * np.cos(rad)],
            y=[-260, -260 + 450 * np.sin(rad)],
            mode="lines",
            line=dict(color="rgba(56, 189, 248, 0.18)", width=1),
            showlegend=False,
            hoverinfo="none",
        ))

    # Monitoring prisms and piezometers
    sensors = [
        {"type": "VWP Piezometer", "x": 0, "y": 190, "code": "PZ-01 (42 kPa)", "color": "#10b981"},
        {"type": "VWP Piezometer", "x": 160, "y": 20, "code": "PZ-02 (26 kPa)", "color": "#10b981"},
        {"type": "Inclinometer", "x": 20, "y": 140, "code": "IPI-01 (Shear: 3.2mm)", "color": "#f59e0b"},
        {"type": "Robotic Prism", "x": -150, "y": 40, "code": "PRISM-04 (1.2mm)", "color": "#a855f7"},
        {"type": "Robotic Prism", "x": 0, "y": -160, "code": "PRISM-07 (0.8mm)", "color": "#a855f7"},
    ]
    for s in sensors:
        fig.add_trace(go.Scatter(
            x=[s["x"]], y=[s["y"]],
            mode="markers+text",
            name=s["type"],
            text=[s["code"]],
            textposition="top center",
            textfont=dict(color="#f8fafc", size=9, family="monospace"),
            marker=dict(size=9, color=s["color"], symbol="triangle-up"),
            showlegend=False,
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.7)",
        title=dict(
            text="📡 Multi-Sensor Network Topology & GroundProbe SSR Line-of-Sight Coverage",
            font=dict(color="#f1f5f9", size=13),
        ),
        xaxis=dict(title="Easting (m)", gridcolor="#1e293b", color="#94a3b8"),
        yaxis=dict(title="Northing (m)", gridcolor="#1e293b", color="#94a3b8"),
        margin=dict(l=40, r=40, b=40, t=50),
        height=520,
    )
    return fig


# --- SYSTEM STATE & INITIALIZATION ---
if "bb" not in st.session_state:
    st.session_state.bb = MineBlackboard()
    st.session_state.ctrl = RockWatchController(st.session_state.bb)
    st.session_state.sim_engine = GeotechnicalSimulationEngine(time_horizon_hours=24.0, step_hours=0.5)
    st.session_state.hazard_mode = "MILD"  # Default scenario mode
    st.session_state.sim_time_hour = 12.0
    st.session_state.user_role = "OPERATOR"
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": "RockWatch Multi-Agent Safety Copilot active. Monitoring 4 pit sectors. All agents (Health, Perception, TARP, Forecaster, Advisor, Arbiter) are online and synchronized with the blackboard."
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
        "desc": "Baseline mining operations. Benches stable, background radar creep (0.7 mm/hr), Level 1 Green.",
        "video": "data/videos/cam01_north_wall_normal.mp4",
        "vel": 0.75,
        "rain": 1.2,
        "pore": 24.0,
        "ppv": 0.35,
        "crack_cm": 0.0,
        "status": "GREEN",
        "directive": "Routine automated monitoring. Normal operations authorized.",
    },
    "MILD": {
        "name": "🟡 2. Mild Slope Ravelling & Loose Stones",
        "desc": "Hairline crest tension crack opens. Small gravel rolling down face. Radar creep accelerates to 2.8 mm/hr. Level 2 Yellow Caution.",
        "video": "data/videos/mild_rockfall.mp4",
        "vel": 2.85,
        "rain": 4.5,
        "pore": 42.0,
        "ppv": 0.85,
        "crack_cm": 2.4,
        "status": "YELLOW",
        "directive": "Heightened caution. Restrict non-essential foot traffic; alert pit supervisor.",
    },
    "MODERATE": {
        "name": "🟠 3. Moderate Wedge Fissure Dilation",
        "desc": "Scarp fissure dilates to 8cm. Moderate rock slabs detach with dust plume. Radar creep reaches 7.8 mm/hr. Level 3 Orange Warning.",
        "video": "data/videos/moderate_rockfall.mp4",
        "vel": 7.85,
        "rain": 18.0,
        "pore": 72.0,
        "ppv": 2.4,
        "crack_cm": 8.2,
        "status": "ORANGE",
        "directive": "Warning status. Stand down haulage under highwall; dispatch geotechnical inspection crew.",
    },
    "SEVERE": {
        "name": "🔴 4. Severe / Imminent Rockfall Collapse",
        "desc": "Structural shear failure. Tumbling boulder cascades caught on camera. Radar creep breaches 18.5 mm/hr. Level 4 Red Emergency Evacuation.",
        "video": "data/videos/severe_rockfall.mp4",
        "vel": 18.80,
        "rain": 42.0,
        "pore": 94.0,
        "ppv": 6.8,
        "crack_cm": 18.5,
        "status": "RED",
        "directive": "CRITICAL EMERGENCY: Mandatory immediate pit evacuation of all personnel and machinery.",
    },
}

SECTOR_NORMAL_VIDEOS = {
    "ZONE_NORTH_WALL": "data/videos/cam01_north_wall_normal.mp4",
    "ZONE_SOUTH_CREST": "data/videos/cam02_south_crest_normal.mp4",
    "ZONE_RAMP_3": "data/videos/cam03_ramp3_normal.mp4",
    "ZONE_HAUL_ROAD": "data/videos/cam04_haul_road_normal.mp4",
}

# Real-world authentic mine footage and actual geological rockfall/landslide video streams
REAL_WORLD_PRESETS = {
    "NORMAL": {
        "video": "data/videos/real_normal_pit_ops.mp4",
        "desc": "Real-world open-pit quarry and active highwall operations. Heavy haul equipment, nominal creep.",
    },
    "MILD": {
        "video": "data/videos/real_mild_ravelling.mp4",
        "desc": "Real-world mountain slope tension scarp opening and initial surface ravelling.",
    },
    "MODERATE": {
        "video": "data/videos/real_moderate_slabbing.mp4",
        "desc": "Real-world quarry highwall bench detachment, rock fracturing, and sliding slabs.",
    },
    "SEVERE": {
        "video": "data/videos/real_severe_collapse.mp4",
        "desc": "Real-world catastrophic mountain slope collapse and massive debris avalanche in Full HD 1080p.",
    },
}

SECTOR_REAL_VIDEOS = {
    "ZONE_NORTH_WALL": "data/videos/real_normal_pit_ops.mp4",
    "ZONE_SOUTH_CREST": "data/videos/real_openpit_haulage.mp4",
    "ZONE_RAMP_3": "data/videos/real_normal_quarry.mp4",
    "ZONE_HAUL_ROAD": "data/videos/real_openpit_haulage.mp4",
}

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

# Populate sector states
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


# ==============================================================================
# 1. TOP MISSION CONTROL HUD HEADER
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
# 2. MAIN COCKPIT VIEWPORT (60% 3D DIGITAL TWIN | 40% CCTV VIDEO & CHAT)
# ==============================================================================
col_left, col_right = st.columns([1.5, 1.0], gap="medium")

# --- LEFT COLUMN: 3D DIGITAL TWIN & REAL-TIME CONTROLS ---
with col_left:
    st.markdown('<div class="card-title">🌐 Geotechnical Mine Engineering Visualization Suite</div>', unsafe_allow_html=True)

    geotech_view = st.radio(
        "Active Spatial & Engineering View:",
        options=[
            "🌐 3D Multi-Lobed Open-Pit Digital Twin",
            "📐 2D Highwall Cross-Section & Failure Arc (FoS)",
            "📈 Saito Creep Inverse Velocity Forecast (1/v)",
            "📡 Sensor Network Topology & Ground Radar",
        ],
        horizontal=True,
        key="geotech_spatial_view",
    )

    if geotech_view == "🌐 3D Multi-Lobed Open-Pit Digital Twin":
        X_grid, Y_grid, Z_grid, s_map = generate_realistic_pit_topography()
        fig_3d = build_3d_pit_figure(X_grid, Y_grid, Z_grid, s_map, sim_telemetries, selected_sector_key)
        st.plotly_chart(fig_3d, width="stretch", config={"displayModeBar": False})
    elif geotech_view == "📐 2D Highwall Cross-Section & Failure Arc (FoS)":
        fig_sec = build_geotech_cross_section_figure(cur_sec['name'], selected_mode, cur_sec)
        st.plotly_chart(fig_sec, width="stretch", config={"displayModeBar": False})
    elif geotech_view == "📈 Saito Creep Inverse Velocity Forecast (1/v)":
        fig_saito = build_saito_forecast_figure(selected_mode, cur_sec['radar_velocity_mm_hr'])
        st.plotly_chart(fig_saito, width="stretch", config={"displayModeBar": False})
    else:
        fig_topo = build_sensor_topology_figure(sim_telemetries)
        st.plotly_chart(fig_topo, width="stretch", config={"displayModeBar": False})

    # Sector Selector Radio Pills
    sec_names = {k: v["name"] for k, v in sim_telemetries.items()}
    chosen_sector = st.radio(
        "Active Monitored Pit Sector:",
        options=list(sec_names.keys()),
        format_func=lambda k: f"{sec_names[k]} ({sim_telemetries[k]['effective_level']})",
        horizontal=True,
        key="selected_sector",
    )

    # --- USER-REQUESTED HAZARD SEVERITY SCENARIO SELECTOR ---
    st.markdown('<div class="card-title" style="margin-top: 14px;">🎛️ Live Hazard Scenario Progression ("All Together" in Lockstep)</div>', unsafe_allow_html=True)

    # Selection of Mild, Moderate, Severe, or Normal
    mode_options = list(HAZARD_PRESETS.keys())
    selected_mode = st.radio(
        "Trigger Hazard Severity Level:",
        options=mode_options,
        format_func=lambda m: HAZARD_PRESETS[m]["name"],
        horizontal=True,
        key="hazard_mode",
    )
    st.caption(HAZARD_PRESETS[selected_mode]["desc"])


# --- RIGHT COLUMN: REAL CCTV VIDEO, GAUGES & MULTI-AGENT CHAT ---
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
            <strong>TARP Action Directive:</strong> {cur_sec['directive']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. REAL CCTV SURVEILLANCE & AI DETECTION TABS
    st.markdown('<div class="card-title">🎥 Video Surveillance Feeds & Rockfall Detachment AI</div>', unsafe_allow_html=True)

    # Footage Source Selector: Real-World vs Synthetic AI Overlays
    footage_source = st.radio(
        "Surveillance Video Source:",
        options=["🎥 Real-Life World Footage (Actual Mines & Landslides)", "🤖 Computer Vision AI Tracking Overlay Stream"],
        horizontal=True,
        key="footage_source_mode",
    )

    tab_focused, tab_upload, tab_matrix = st.tabs([
        "📹 Monitored CCTV",
        "📤 Video Uploader & AI",
        "🎛️ 4-Camera Matrix",
    ])

    with tab_focused:
        if footage_source == "🎥 Real-Life World Footage (Actual Mines & Landslides)":
            if selected_mode == "NORMAL":
                video_path = SECTOR_REAL_VIDEOS.get(chosen_sector, "data/videos/real_normal_pit_ops.mp4")
            else:
                video_path = REAL_WORLD_PRESETS[selected_mode]["video"]
            footage_desc = REAL_WORLD_PRESETS[selected_mode]["desc"]
        else:
            if selected_mode == "NORMAL":
                video_path = SECTOR_NORMAL_VIDEOS.get(chosen_sector, "data/videos/cam01_north_wall_normal.mp4")
            else:
                video_path = HAZARD_PRESETS[selected_mode]["video"]
            footage_desc = HAZARD_PRESETS[selected_mode]["desc"]

        if os.path.exists(video_path):
            safe_render_video(video_path, autoplay=True, loop=True, key=f"vid_focused_{chosen_sector}_{selected_mode}_{footage_source[:4]}")
        else:
            st.info("Surveillance feed initializing...")

        # Visual Detection Details Card
        det_stats = {
            "NORMAL": {"frags": 0, "opt_v": 0.35, "vol": 0.0, "conf": 95, "banner": "🟢 STABLE HIGHWALL | Normal Baseline Creep | No Detachments", "alert": "TARP LEVEL 1 GREEN (Normal Operations)"},
            "MILD": {"frags": 4, "opt_v": 1.92, "vol": 0.3, "conf": 89, "banner": "🟡 TENSION CRACK OPENING + LOOSE GRAVEL RAVELLING DETECTED", "alert": "TARP LEVEL 2 YELLOW CAUTION (Foot Traffic Restricted)"},
            "MODERATE": {"frags": 12, "opt_v": 4.41, "vol": 3.4, "conf": 94, "banner": "🟠 SCARP FISSURE EXTENSION + SLAB DETACHMENT DETECTED", "alert": "TARP LEVEL 3 ORANGE WARNING (Stand Down Haul Ramp)"},
            "SEVERE": {"frags": 44, "opt_v": 10.55, "vol": 28.6, "conf": 99, "banner": "🔴 MASSIVE BOULDER CASCADE COLLAPSE DETECTED", "alert": "TARP LEVEL 4 RED EMERGENCY (Mandatory Pit Evacuation)"},
        }
        active_stat = det_stats[selected_mode]

        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(59, 130, 246, 0.25); border-radius: 8px; padding: 10px 14px; margin-top: 8px;">
            <div style="font-size: 0.82rem; font-weight: 700; color: #f1f5f9; margin-bottom: 4px;">
                {active_stat['banner']}
            </div>
            <div style="font-size: 0.78rem; color: #94a3b8; margin-bottom: 6px;">
                <strong>Footage Description:</strong> {footage_desc}
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; font-size: 0.75rem; color: #cbd5e1; font-family: monospace;">
                <div>FRAGMENTS: <span style="color:#38bdf8; font-weight:bold;">{active_stat['frags']}</span></div>
                <div>OPTICAL VEL: <span style="color:#38bdf8; font-weight:bold;">{active_stat['opt_v']} m/s</span></div>
                <div>VOL LOSS: <span style="color:#38bdf8; font-weight:bold;">{active_stat['vol']} m³</span></div>
                <div>AI CONF: <span style="color:#34d399; font-weight:bold;">{active_stat['conf']}%</span></div>
            </div>
            <div style="margin-top: 6px; font-size: 0.78rem; color: #f59e0b;">
                ⚡ <strong>ALERT GENERATED:</strong> {active_stat['alert']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tab_upload:
        st.markdown('<div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 6px;">Upload real-world open-pit drone or CCTV video to run live computer vision rockfall detection:</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload Pit Drone / CCTV Video (.mp4, .mov, .avi):",
            type=["mp4", "mov", "avi", "m4v"],
            key="custom_pit_video_uploader",
            label_visibility="collapsed"
        )

        sample_choice = st.selectbox(
            "Or test AI analysis on a preloaded real-world mine clip:",
            options=[
                "None",
                "Real Clip 1: Catastrophic Landslide & Slope Collapse (Full HD)",
                "Real Clip 2: Quarry Highwall Bench Detachment & Slabbing",
                "Real Clip 3: Mountain Slope Scarp & Initial Surface Ravelling",
                "Real Clip 4: Active Open-Pit Mining Benches & Equipment",
                "Real Clip 5: Garzweiler Surface Mine Heavy Haulage",
            ],
            key="sample_video_choice"
        )

        video_to_analyze = None
        if uploaded_file is not None:
            save_path = os.path.join("data/videos/uploads", uploaded_file.name)
            with open(save_path, "wb") as f_out:
                f_out.write(uploaded_file.getbuffer())
            video_to_analyze = save_path
        elif sample_choice == "Real Clip 1: Catastrophic Landslide & Slope Collapse (Full HD)":
            video_to_analyze = "data/videos/real_severe_collapse.mp4"
        elif sample_choice == "Real Clip 2: Quarry Highwall Bench Detachment & Slabbing":
            video_to_analyze = "data/videos/real_moderate_slabbing.mp4"
        elif sample_choice == "Real Clip 3: Mountain Slope Scarp & Initial Surface Ravelling":
            video_to_analyze = "data/videos/real_mild_ravelling.mp4"
        elif sample_choice == "Real Clip 4: Active Open-Pit Mining Benches & Equipment":
            video_to_analyze = "data/videos/real_normal_pit_ops.mp4"
        elif sample_choice == "Real Clip 5: Garzweiler Surface Mine Heavy Haulage":
            video_to_analyze = "data/videos/real_openpit_haulage.mp4"

        if video_to_analyze and os.path.exists(video_to_analyze):
            safe_render_video(video_to_analyze, autoplay=True, loop=True, key=f"vid_upload_player_{hash(video_to_analyze) % 10000}")
            with st.spinner("Executing Optical Flow & Volumetric Rockfall Analysis..."):
                analysis_res = analyze_uploaded_video(video_to_analyze)

            tier = analysis_res["detected_hazard_tier"]
            tier_badge = badge_classes.get(tier if tier != "NORMAL" else "GREEN", "badge-green")
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(59, 130, 246, 0.35); border-radius: 8px; padding: 12px; margin-top: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: 800; font-size: 0.95rem; color: #f1f5f9;">VIDEO AI DETECTOR OUTPUT</span>
                    <span class="{tier_badge}">TARP {analysis_res['tarp_level']}</span>
                </div>
                <div style="font-size: 0.82rem; color: #cbd5e1; margin-bottom: 8px;">
                    <strong>Alert:</strong> {analysis_res['alert_message']}
                </div>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; font-size: 0.75rem; color: #cbd5e1; font-family: monospace;">
                    <div>FRAGMENTS: <span style="color:#38bdf8;">{analysis_res['fragment_count']}</span></div>
                    <div>MAX VEL: <span style="color:#38bdf8;">{analysis_res['max_velocity_ms']} m/s</span></div>
                    <div>VOLUME: <span style="color:#38bdf8;">{analysis_res['volume_m3']} m³</span></div>
                    <div>CONFIDENCE: <span style="color:#34d399;">{int(analysis_res['confidence']*100)}%</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if analysis_res.get("annotated_frame_path") and os.path.exists(analysis_res["annotated_frame_path"]):
                st.image(analysis_res["annotated_frame_path"], caption="Keyframe Optical Motion AI Bounding Box Overlay", width="stretch")

    with tab_matrix:
        st.caption("Live Simultaneous Surveillance Matrix across all 4 Pit Fronts (Real-World Feeds):")
        mat_c1, mat_c2 = st.columns(2)
        with mat_c1:
            st.markdown(f"**CAM-01:** North Highwall ({sim_telemetries['ZONE_NORTH_WALL']['effective_level']})")
            v1 = "data/videos/real_normal_pit_ops.mp4" if sim_telemetries['ZONE_NORTH_WALL']['effective_level'] == "GREEN" else REAL_WORLD_PRESETS[selected_mode]["video"]
            safe_render_video(v1, autoplay=True, loop=True, key=f"mat_v1_{chosen_sector}_{selected_mode}")

            st.markdown(f"**CAM-02:** South Crest ({sim_telemetries['ZONE_SOUTH_CREST']['effective_level']})")
            v2 = "data/videos/real_openpit_haulage.mp4" if sim_telemetries['ZONE_SOUTH_CREST']['effective_level'] == "GREEN" else REAL_WORLD_PRESETS[selected_mode]["video"]
            safe_render_video(v2, autoplay=True, loop=True, key=f"mat_v2_{chosen_sector}_{selected_mode}")
        with mat_c2:
            st.markdown(f"**CAM-03:** Ramp 3 Switchback ({sim_telemetries['ZONE_RAMP_3']['effective_level']})")
            v3 = "data/videos/real_normal_quarry.mp4" if sim_telemetries['ZONE_RAMP_3']['effective_level'] == "GREEN" else REAL_WORLD_PRESETS[selected_mode]["video"]
            safe_render_video(v3, autoplay=True, loop=True, key=f"mat_v3_{chosen_sector}_{selected_mode}")

            st.markdown(f"**CAM-04:** West Pit Haul Road ({sim_telemetries['ZONE_HAUL_ROAD']['effective_level']})")
            v4 = "data/videos/real_openpit_haulage.mp4" if sim_telemetries['ZONE_HAUL_ROAD']['effective_level'] == "GREEN" else REAL_WORLD_PRESETS[selected_mode]["video"]
            safe_render_video(v4, autoplay=True, loop=True, key=f"mat_v4_{chosen_sector}_{selected_mode}")

    # 2. MULTIMODAL SENSOR TELEMETRY METRIC BOXES
    st.markdown('<div class="card-title" style="margin-top: 10px;">📊 Multimodal Telemetry Streams</div>', unsafe_allow_html=True)
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
    if cur_sec["dominance_override"] or sec_level == "RED":
        st.markdown("""
        <div style="background: rgba(239, 68, 68, 0.18); border: 1px solid #ef4444; border-radius: 8px; padding: 10px 14px; margin-top: 8px;">
            <span style="color: #f87171; font-weight: 700; font-size: 0.85rem;">⚡ DOMINANCE OVERRIDE ACTIVATED</span><br>
            <span style="color: #cbd5e1; font-size: 0.78rem;">Single-sensor anomaly breached 15.0 mm/hr safety floor. Alert escalated directly without dilution.</span>
        </div>
        """, unsafe_allow_html=True)

    # 3. INTERACTIVE MULTI-AGENT CHAT COPILOT
    st.markdown('<div class="card-title" style="margin-top: 12px;">💬 Multi-Agent Geotechnical Copilot (Ask the Agents)</div>', unsafe_allow_html=True)

    # Chat history container
    chat_container = st.container(height=220)
    with chat_container:
        for msg in st.session_state.chat_history[-6:]:
            role_icon = "🤖" if msg["role"] == "assistant" else "👤"
            st.markdown(f"**{role_icon} {msg['role'].upper()}:** {msg['content']}")

    # User Query Input
    user_query = st.chat_input("Ask why an alert triggered, request SOP actions, or audit sensors...")
    if user_query:
        # Save user message
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # Multi-Agent Collaborative Reasoning
        sop_info = retrieve_sop(sec_level)
        q_l = user_query.lower()

        if "why" in q_l or "reason" in q_l or "alert" in q_l or "trigger" in q_l:
            response = (
                f"**Multi-Agent Assessment for {cur_sec['name']}:**\n\n"
                f"- **Active Alert Level:** **{sec_level}**\n"
                f"- **Telemetry Breakdown:** Radar creep is **{cur_sec['radar_velocity_mm_hr']:.2f} mm/hr**, "
                f"pore pressure is **{cur_sec['pore_pressure_kpa']:.0f} kPa**, and 3h rainfall is **{cur_sec['rainfall_3h_mm']:.1f} mm**.\n"
                f"- **Visual Confirmation:** CCTV CAM-01 confirmed {HAZARD_PRESETS[active_mode]['desc'].lower()}.\n"
                f"- **Grounded SOP Rule:** {sop_info.get('passage_id', '[TARP-SOP-01]')}: *{sop_info.get('action', 'Routine monitoring.')}*\n"
                f"- **Arbiter Consensus:** Discrepancy check: {'DISCREPANCY FLAGGED' if cur_sec.get('discrepancy') else 'IN CONSENSUS'}. Caution elevated asymmetrically."
            )
        elif "evacuat" in q_l or "sop" in q_l or "protocol" in q_l or "action" in q_l:
            response = (
                f"**Mandatory Protocol for {sec_level} Status:**\n\n"
                f"**Citation:** `{sop_info.get('passage_id', '[TARP-SOP-01]')}`\n"
                f"{sop_info.get('text', 'Normal monitoring.')}\n\n"
                f"**Geotechnical Action Items:**\n"
                f"1. Acknowledge and dispatch notice to pit operations manager.\n"
                f"2. Radio equipment operators on {cur_sec['name']}.\n"
                f"3. {'MANDATORY: Evacuate all personnel and heavy machinery immediately. Do not enter exclusion zone.' if sec_level == 'RED' else 'Inspect crest tension cracks and verify extensometers.'}"
            )
        elif "sensor" in q_l or "health" in q_l or "radar" in q_l:
            response = (
                f"**Sensor Health Audit (Agent 1 - HealthAgent):**\n\n"
                f"- Slope Stability Radar (SSR-02): ONLINE | Signal: 94 dB | Velocity: {cur_sec['radar_velocity_mm_hr']:.2f} mm/hr\n"
                f"- Piezometer (PZ-04): ONLINE | Pore Pressure: {cur_sec['pore_pressure_kpa']:.0f} kPa\n"
                f"- Weather Station: ONLINE | Precipitation: {cur_sec['rainfall_3h_mm']:.1f} mm / 3h\n"
                f"- Surveillance CCTV: STREAMING 25 FPS | AI Detachment Detector Active"
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
        st.dataframe(recent_entries, width="stretch")
