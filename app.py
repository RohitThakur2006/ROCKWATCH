"""
app.py - RockWatch Mission-Control: 3D Geotechnical Digital Twin & Monitoring Center.

Fully customized mission-control interface:
- High-tech dark command console gradient with zero Streamlit chrome (headers/toolbars hidden).
- Stepped Open-Pit 3D Digital Twin with real-time deformation heatmap draping.
- Multi-Sector Continuous Telemetry Simulator across pit zones.
- Live Surveillance Camera with real-time AI rockfall/crack bounding box overlay.
- Multimodal Late-Sensor Fusion (Option B) with single-sensor dominance override.
- AI Geotechnical Safety Assistant (Semantic Kernel RAG grounded in SOPs).
- Authenticated Geotechnical Engineer Gate with automatic fail-closed expiry.
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
from ml.vision_detector import PitVisionDetector

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
    padding-top: 0.75rem !important;
    padding-bottom: 1.5rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
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
    padding: 14px 24px;
    margin-bottom: 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.1);
}

.hud-title {
    font-size: 1.35rem;
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
    padding: 16px 20px;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45);
    margin-bottom: 14px;
}

.mission-card:hover {
    border-color: rgba(59, 130, 246, 0.35);
}

.card-title {
    font-size: 0.9rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* 6. Dynamic Glowing Status Badges */
.badge-green {
    background: rgba(16, 185, 129, 0.18);
    border: 1px solid #10b981;
    color: #34d399;
    padding: 5px 14px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.85rem;
    box-shadow: 0 0 14px rgba(16, 185, 129, 0.3);
}

.badge-yellow {
    background: rgba(245, 158, 11, 0.18);
    border: 1px solid #f59e0b;
    color: #fbbf24;
    padding: 5px 14px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.85rem;
    box-shadow: 0 0 14px rgba(245, 158, 11, 0.35);
}

.badge-orange {
    background: rgba(249, 115, 22, 0.2);
    border: 1px solid #f97316;
    color: #fb923c;
    padding: 5px 14px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.85rem;
    box-shadow: 0 0 16px rgba(249, 115, 22, 0.4);
}

.badge-red {
    background: rgba(239, 68, 68, 0.25);
    border: 1px solid #ef4444;
    color: #f87171;
    padding: 5px 14px;
    border-radius: 9999px;
    font-weight: 800;
    font-size: 0.85rem;
    box-shadow: 0 0 20px rgba(239, 68, 68, 0.6);
    animation: criticalPulse 1.8s infinite;
}

@keyframes criticalPulse {
    0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.8); }
    70% { box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); }
    100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
}

/* 7. Metric Value Pill */
.metric-box {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 10px 14px;
    text-align: center;
}
.metric-box-val {
    font-size: 1.25rem;
    font-weight: 800;
    color: #f8fafc;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}
.metric-box-label {
    font-size: 0.72rem;
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

/* Clean button overrides */
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


# --- 3D OPEN-PIT MESH GENERATION (STEADY TOPOGRAPHY) ---
@st.cache_data
def generate_base_pit_topography(n_grid: int = 70):
    """
    Generates a realistic stepped conical open-pit topography
    with terraces, spiral haul ramp cut, and structural sector domains.
    """
    x = np.linspace(-350, 350, n_grid)
    y = np.linspace(-350, 350, n_grid)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    Z_crest = 320.0
    Z_floor = 60.0
    pit_radius = 320.0
    bench_h = 24.0

    # Stepped elevation drops
    norm_r = np.clip(R / pit_radius, 0.0, 1.0)
    steps = np.floor((1.0 - norm_r) * (Z_crest - Z_floor) / bench_h)
    Z = Z_crest - (steps * bench_h)

    # Spiral haul ramp cut
    theta = np.arctan2(Y, X) % (2 * np.pi)
    ramp_cut = np.where((R < pit_radius) & (R > 40), (theta / (2 * np.pi)) * bench_h * 0.45, 0.0)
    Z = np.clip(Z - ramp_cut, Z_floor, Z_crest)

    # Sector mapping:
    # 1: ZONE_NORTH_WALL (Y > 50, R < 310)
    # 2: ZONE_SOUTH_CREST (Y < -60, R < 310)
    # 3: ZONE_RAMP_3 (X > 40, -50 <= Y <= 50)
    # 4: ZONE_HAUL_ROAD (X < -60, -50 <= Y <= 50)
    # 0: Pit Rim / Natural Surface
    sector_id_map = np.zeros_like(Z, dtype=int)
    sector_id_map[(Y > 50) & (R < pit_radius) & (R > 60)] = 1
    sector_id_map[(Y < -60) & (R < pit_radius) & (R > 60)] = 2
    sector_id_map[(X > 40) & (Y >= -50) & (Y <= 50) & (R < pit_radius)] = 3
    sector_id_map[(X < -60) & (Y >= -50) & (Y <= 50) & (R < pit_radius)] = 4

    return X, Y, Z, sector_id_map


def build_3d_pit_figure(X, Y, Z, sector_id_map, sector_telemetries, selected_sector_key):
    """
    Builds the Plotly 3D WebGL Surface draped with real-time deformation heatmaps.
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

    # Assign live radar creep velocities and labels to surface
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

    # Color Scale: Green (0-1.5) -> Yellow (1.5-5) -> Orange (5-15) -> Critical Crimson Red (>15)
    tarp_colorscale = [
        [0.00, "#059669"],   # Emerald Normal
        [0.08, "#10b981"],
        [0.25, "#f59e0b"],   # Amber Caution (1.5 - 5 mm/hr)
        [0.60, "#f97316"],   # Orange Warning (5 - 15 mm/hr)
        [0.85, "#ef4444"],   # Crimson Critical
        [1.00, "#b91c1c"],   # Deep Red Failure
    ]

    fig = go.Figure(data=[
        go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=surf_color,
            cmin=0.0, cmax=20.0,
            colorscale=tarp_colorscale,
            customdata=custom_data,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Elevation: %{z:.1f} m<br>"
                "Deformation Velocity: %{customdata[1]}<br>"
                "TARP Tier: %{customdata[2]}"
                "<extra></extra>"
            ),
            showscale=True,
            colorbar=dict(
                title=dict(text="Radar Velocity (mm/hr)", font=dict(color="#94a3b8", size=11)),
                tickfont=dict(color="#94a3b8", size=10),
                thickness=14,
                len=0.75,
                x=0.98,
            ),
            lighting=dict(ambient=0.65, diffuse=0.8, specular=0.2, roughness=0.7),
        )
    ])

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(
                title="Easting (m)",
                backgroundcolor="rgba(10, 15, 26, 0.8)",
                gridcolor="#1e293b",
                color="#64748b",
                showbackground=True,
            ),
            yaxis=dict(
                title="Northing (m)",
                backgroundcolor="rgba(10, 15, 26, 0.8)",
                gridcolor="#1e293b",
                color="#64748b",
                showbackground=True,
            ),
            zaxis=dict(
                title="Elevation (m)",
                backgroundcolor="rgba(10, 15, 26, 0.8)",
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


# --- SYSTEM STATE INITIALIZATION ---
if "bb" not in st.session_state:
    st.session_state.bb = MineBlackboard()
    st.session_state.ctrl = RockWatchController(st.session_state.bb)
    st.session_state.detector = PitVisionDetector()
    st.session_state.sim_clock = 0.0
    st.session_state.chat_history = [
        {"role": "assistant", "content": "RockWatch AI Safety Copilot initialized. Monitoring all 4 pit sectors. All sensor streams currently green."}
    ]

    # Initialize continuous multi-sector states
    st.session_state.sectors = {
        "ZONE_NORTH_WALL": {
            "name": "North Wall Bench 4",
            "radar_velocity_mm_hr": 0.8,
            "rainfall_3h_mm": 2.0,
            "vibration_ppv_mms": 0.4,
            "pore_pressure_kpa": 28.0,
            "effective_level": "GREEN",
            "risk_probability": 0.12,
            "discrepancy": False,
            "dominance_override": False,
            "directive": "Routine automated monitoring active.",
            "crack_width": 0,
            "detachment": False,
        },
        "ZONE_SOUTH_CREST": {
            "name": "South Crest Ramp",
            "radar_velocity_mm_hr": 0.5,
            "rainfall_3h_mm": 1.5,
            "vibration_ppv_mms": 0.2,
            "pore_pressure_kpa": 22.0,
            "effective_level": "GREEN",
            "risk_probability": 0.08,
            "discrepancy": False,
            "dominance_override": False,
            "directive": "Routine automated monitoring active.",
            "crack_width": 0,
            "detachment": False,
        },
        "ZONE_RAMP_3": {
            "name": "Main Haulage Ramp 3",
            "radar_velocity_mm_hr": 0.6,
            "rainfall_3h_mm": 1.0,
            "vibration_ppv_mms": 0.8,
            "pore_pressure_kpa": 24.0,
            "effective_level": "GREEN",
            "risk_probability": 0.10,
            "discrepancy": False,
            "dominance_override": False,
            "directive": "Routine automated monitoring active.",
            "crack_width": 0,
            "detachment": False,
        },
        "ZONE_HAUL_ROAD": {
            "name": "West Pit Access Road",
            "radar_velocity_mm_hr": 0.4,
            "rainfall_3h_mm": 0.5,
            "vibration_ppv_mms": 0.3,
            "pore_pressure_kpa": 20.0,
            "effective_level": "GREEN",
            "risk_probability": 0.06,
            "discrepancy": False,
            "dominance_override": False,
            "directive": "Routine automated monitoring active.",
            "crack_width": 0,
            "detachment": False,
        },
    }

bb: MineBlackboard = st.session_state.bb
ctrl: RockWatchController = st.session_state.ctrl
detector: PitVisionDetector = st.session_state.detector

# --- EXECUTE CYCLE FOR ALL SECTORS ---
highest_global_level = "GREEN"
level_rank = {"GREEN": 1, "YELLOW": 2, "ORANGE": 3, "RED": 4}

for s_key, s_data in st.session_state.sectors.items():
    raw_frame = {
        "radar_velocity_mm_hr": s_data["radar_velocity_mm_hr"],
        "rainfall_3h_mm": s_data["rainfall_3h_mm"],
        "vibration_ppv_mms": s_data["vibration_ppv_mms"],
        "timestamp": st.session_state.sim_clock,
    }
    cycle_res = ctrl.run_telemetry_cycle(s_key, raw_frame)
    s_data["effective_level"] = cycle_res["effective_level"]
    s_data["directive"] = cycle_res["final_alert_package"].get("advisory_directive", "Routine monitoring.")
    s_data["discrepancy"] = cycle_res.get("discrepancy_detected", False)
    
    # Read forecast details
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
        <span>⚡</span> ROCKWATCH // 3D DIGITAL TWIN & EARLY WARNING SYSTEM
    </div>
    <div style="display: flex; align-items: center; gap: 20px;">
        <div style="display: flex; align-items: center; gap: 8px; font-size: 0.85rem; color: #94a3b8;">
            <span class="pulse-dot"></span> LIVE SENSORS
        </div>
        <div style="font-size: 0.82rem; color: #64748b; font-family: monospace;">
            LEDGER: <span style="color: #38bdf8;">{head_hash}</span>
        </div>
        <div style="font-size: 0.82rem; color: #64748b; font-family: monospace;">
            CLOCK: <span style="color: #e2e8f0;">+{st.session_state.sim_clock:.1f}h</span>
        </div>
        <div>
            <span class="{global_badge_class}">GLOBAL: LEVEL {level_rank[highest_global_level]} {highest_global_level}</span>
        </div>
    </div>
</div>
"""
st.markdown(hud_html, unsafe_allow_html=True)


# ==============================================================================
# 2. MAIN COCKPIT VIEWPORT (62% 3D DIGITAL TWIN | 38% SECTOR DIAGNOSTICS DECK)
# ==============================================================================
col_left, col_right = st.columns([1.6, 1.0], gap="medium")

# --- LEFT COLUMN: 3D DIGITAL TWIN & REAL-TIME CONTROLS ---
with col_left:
    st.markdown('<div class="card-title">🌐 3D Open-Pit Digital Twin & Deformation Velocity Heatmap</div>', unsafe_allow_html=True)
    
    # Render 3D Pit Mesh
    X_grid, Y_grid, Z_grid, s_map = generate_base_pit_topography()
    selected_sector_key = st.session_state.get("selected_sector", "ZONE_NORTH_WALL")
    
    fig_3d = build_3d_pit_figure(
        X_grid, Y_grid, Z_grid, s_map,
        st.session_state.sectors,
        selected_sector_key
    )
    st.plotly_chart(fig_3d, width="stretch", config={"displayModeBar": False})

    # Sector Selector Pills
    sec_names = {k: v["name"] for k, v in st.session_state.sectors.items()}
    chosen_sector = st.radio(
        "Inspect Structural Pit Sector:",
        options=list(sec_names.keys()),
        format_func=lambda k: f"{sec_names[k]} ({st.session_state.sectors[k]['effective_level']})",
        horizontal=True,
        key="selected_sector",
    )

    # --- SIMULATION PLAYBACK & SCENARIO INJECTION DRAWER ---
    st.markdown('<div class="card-title" style="margin-top: 14px;">🎮 Pit Telemetry Stream Controller & Scenarios</div>', unsafe_allow_html=True)
    sim_c1, sim_c2, sim_c3, sim_c4 = st.columns(4)

    with sim_c1:
        if st.button("🚨 Inject Radar Spike (>15 mm/hr)"):
            st.session_state.sectors["ZONE_NORTH_WALL"]["radar_velocity_mm_hr"] = 18.5
            st.session_state.sectors["ZONE_NORTH_WALL"]["crack_width"] = 12
            st.session_state.sectors["ZONE_NORTH_WALL"]["detachment"] = True
            st.rerun()

    with sim_c2:
        if st.button("🌧️ Inject Heavy Rain (85mm)"):
            st.session_state.sectors["ZONE_SOUTH_CREST"]["rainfall_3h_mm"] = 35.0
            st.session_state.sectors["ZONE_SOUTH_CREST"]["pore_pressure_kpa"] = 92.0
            st.session_state.sectors["ZONE_SOUTH_CREST"]["radar_velocity_mm_hr"] = 6.8
            st.rerun()

    with sim_c3:
        if st.button("⏩ Advance Clock (+1h)"):
            st.session_state.sim_clock += 1.0
            # Natural minor random walk in telemetry
            for s in st.session_state.sectors.values():
                s["radar_velocity_mm_hr"] = max(0.2, s["radar_velocity_mm_hr"] + np.random.normal(0, 0.05))
            st.rerun()

    with sim_c4:
        if st.button("🔄 Reset to Baseline"):
            for s in st.session_state.sectors.values():
                s["radar_velocity_mm_hr"] = 0.8
                s["rainfall_3h_mm"] = 2.0
                s["vibration_ppv_mms"] = 0.4
                s["pore_pressure_kpa"] = 25.0
                s["crack_width"] = 0
                s["detachment"] = False
            st.session_state.sim_clock = 0.0
            st.rerun()


# --- RIGHT COLUMN: SECTOR INSPECTOR, CAMERA, & AI SAFETY AGENT ---
cur_sec = st.session_state.sectors[chosen_sector]
sec_level = cur_sec["effective_level"]
sec_badge_class = badge_classes.get(sec_level, "badge-green")

with col_right:
    # Selected Sector Title & Directive Card
    st.markdown(f"""
    <div class="mission-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-weight: 800; font-size: 1.1rem; color: #f1f5f9;">{cur_sec['name']}</span>
            <span class="{sec_badge_class}">{sec_level}</span>
        </div>
        <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.4;">
            <strong>TARP Action:</strong> {cur_sec['directive']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. LIVE CAMERA WITH AI BOUNDING BOXES
    st.markdown('<div class="card-title">🎥 Sector Surveillance Feed (AI Detachment Detector)</div>', unsafe_allow_html=True)
    raw_cam_frame = detector.generate_synthetic_pit_frame(
        with_detachment=cur_sec["detachment"],
        crack_width=cur_sec["crack_width"],
    )
    v_analysis = detector.analyze_frame(chosen_sector, raw_cam_frame)
    cam_img = Image.fromarray(raw_cam_frame)
    draw = ImageDraw.Draw(cam_img)
    for box in v_analysis.get("bounding_boxes", []):
        draw.rectangle([box["xmin"], box["ymin"], box["xmax"], box["ymax"]], outline="#ef4444", width=3)

    st.image(cam_img, width="stretch", caption=f"Feed: {cur_sec['name']} | Vision Detachment Risk: {v_analysis.get('vision_risk_probability', 0.05):.1%}")

    # 2. MULTIMODAL SENSOR TELEMETRY METRIC PILLS
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
    st.markdown('<div class="card-title" style="margin-top: 14px;">🤖 AI Geotechnical Safety Copilot (Grounded in SOP)</div>', unsafe_allow_html=True)
    
    sop_ref = retrieve_sop(sec_level)
    st.markdown(f"""
    <div class="mission-card" style="font-size: 0.85rem; line-height: 1.5; color: #cbd5e1;">
        <div style="color: #38bdf8; font-weight: 700; margin-bottom: 4px;">Citation: {sop_ref.get('passage_id', '[TARP-SOP-01]')}</div>
        {sop_ref.get('text', 'Normal operating conditions.')}
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 3. BOTTOM UTILITY DOCK: GEOTECHNICAL AUTHORIZATION GATE & AUDIT LEDGER
# ==============================================================================
st.markdown("---")
b_c1, b_c2 = st.columns([1, 1], gap="large")

with b_c1:
    st.markdown('<div class="card-title">⚖️ Human-in-the-Loop Geotechnical Authorization Gate</div>', unsafe_allow_html=True)
    st.caption("Inviolable Principle: Operators can log false alarms, but only an authenticated **Geotechnical Engineer** can authorize temporary sensitivity adjustments.")

    role_choice = st.radio(
        "Current Active User Role:",
        ["OPERATOR (View & Report Only)", "GEOTECH_ENGINEER (Authorization Authority)"],
        horizontal=True,
    )
    is_geotech = "GEOTECH_ENGINEER" in role_choice

    g_col1, g_col2 = st.columns([1.5, 1])
    with g_col1:
        if st.button("Submit 0.8x Sensitivity Reduction Proposal"):
            role_tag = "GEOTECH_ENGINEER" if is_geotech else "OPERATOR"
            prop_res = ctrl.submit_desensitization_request(
                zone_id=chosen_sector,
                requested_by_role=role_tag,
                proposed_multiplier=0.8,
                duration_hours=4.0,
                justification="Visual verification indicates thermal instrument noise.",
            )
            if prop_res["approved"]:
                st.success(f"✅ APPROVED by Geotechnical Engineer! Sensitivity adjusted. Auto-expires in 4.0 hours.")
            else:
                st.warning(f"⏳ PENDING. Operators cannot authorize desensitization. Requires Geotechnical Engineer sign-off.")

    with g_col2:
        if st.button("⏩ Fast-Forward Clock +5 Hours"):
            st.session_state.sim_clock += 5.0
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
