"""
generate_scenario_videos.py - Generates realistic open-pit mine CCTV surveillance MP4 videos.

Generates 4 realistic video streams for the dashboard:
1. normal_ops.mp4: Stable rock face, distant haul truck passing, ambient dust, Green Level 1.
2. mild_rockfall.mp4: Hairline tension crack, small pebbles rolling down, Yellow Level 2 Caution.
3. moderate_rockfall.mp4: Widening fissure, rock slabs detaching, Orange Level 3 Warning.
4. severe_rockfall.mp4: Major bench collapse, cascading boulders with red AI bounding boxes, Red Level 4 Emergency.
"""

import os
import shutil
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

VIDEOS_DIR = "data/videos"
os.makedirs(VIDEOS_DIR, exist_ok=True)
TMP_FRAMES = os.path.join(VIDEOS_DIR, "tmp_frames")


def draw_realistic_highwall(draw: ImageDraw.Draw, w: int, h: int, seed: int):
    """Draws a textured open-pit rock highwall with geological jointing and benches."""
    np.random.seed(seed)
    # Natural rock face strata
    strata_colors = [(52, 46, 38), (44, 40, 34), (58, 52, 44), (40, 36, 30)]
    for y in range(0, h, 20):
        c = strata_colors[(y // 20) % len(strata_colors)]
        draw.rectangle([(0, y), (w, y + 20)], fill=c)

    # Major horizontal bench terraces (safety berms)
    draw.line([(0, int(h * 0.28)), (w, int(h * 0.28))], fill=(88, 80, 70), width=6)
    draw.line([(0, int(h * 0.60)), (w, int(h * 0.60))], fill=(82, 75, 65), width=6)
    draw.line([(0, int(h * 0.90)), (w, int(h * 0.90))], fill=(75, 68, 60), width=8)

    # Geological joint fracture network (steeply dipping joint sets)
    for x in range(25, w, 40):
        offset = np.random.randint(-15, 15)
        draw.line([(x, 0), (x + 35 + offset, h)], fill=(28, 24, 20), width=2)


def render_normal_ops_video(out_path: str, n_frames: int = 45):
    """Renders stable mining operations with a haul truck passing on the lower bench."""
    os.makedirs(TMP_FRAMES, exist_ok=True)
    w, h = 640, 360

    for i in range(n_frames):
        img = Image.new("RGB", (w, h), color=(44, 40, 34))
        draw = ImageDraw.Draw(img)
        draw_realistic_highwall(draw, w, h, seed=42)

        # Ambient dust particles drifting horizontally
        np.random.seed(i * 10)
        for _ in range(40):
            dx = (np.random.randint(0, w) + i * 4) % w
            dy = np.random.randint(0, h)
            draw.ellipse([(dx, dy), (dx + 2, dy + 2)], fill=(160, 150, 135))

        # Haul truck moving along lower haul bench (y=210)
        truck_x = int((i / n_frames) * (w + 120) - 60)
        truck_y = int(h * 0.58) - 16
        if -50 <= truck_x <= w + 50:
            # Truck body
            draw.rectangle([(truck_x, truck_y), (truck_x + 34, truck_y + 16)], fill=(234, 179, 8))
            draw.rectangle([(truck_x + 22, truck_y - 6), (truck_x + 34, truck_y + 10)], fill=(202, 138, 4))
            draw.ellipse([(truck_x + 4, truck_y + 12), (truck_x + 12, truck_y + 20)], fill=(15, 23, 42))
            draw.ellipse([(truck_x + 24, truck_y + 12), (truck_x + 32, truck_y + 20)], fill=(15, 23, 42))

        # CCTV Industrial HUD
        draw.rectangle([(0, 0), (w, 24)], fill=(10, 15, 26))
        draw.rectangle([(0, h - 22), (w, h)], fill=(10, 15, 26))
        draw.text((12, 5), f"● REC | CAM-01 [NORTH WALL BENCH 4] | 2026-11-15 12:00:{i:02d} | 25 FPS | HD PTZ", fill=(56, 189, 248))
        draw.text((12, h - 17), "STATUS: NORMAL OPERATIONS | SCENE STABLE | TARP: LEVEL 1 GREEN", fill=(52, 211, 153))

        img.save(os.path.join(TMP_FRAMES, f"frame_{i:03d}.png"))

    subprocess.run([
        "ffmpeg", "-y", "-framerate", "15",
        "-i", os.path.join(TMP_FRAMES, "frame_%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path
    ], capture_output=True, check=True)
    shutil.rmtree(TMP_FRAMES, ignore_errors=True)
    print(f"[✓] Created: {out_path} ({os.path.getsize(out_path)} bytes)")


def render_mild_rockfall_video(out_path: str, n_frames: int = 45):
    """Renders mild slope ravelling: small pebbles rolling down, hairline crest tension crack."""
    os.makedirs(TMP_FRAMES, exist_ok=True)
    w, h = 640, 360

    for i in range(n_frames):
        img = Image.new("RGB", (w, h), color=(44, 40, 34))
        draw = ImageDraw.Draw(img)
        draw_realistic_highwall(draw, w, h, seed=42)

        # Hairline crest tension crack
        crack_x = int(w * 0.45)
        draw.line([(crack_x, 40), (crack_x + 10, 85), (crack_x + 4, 130)], fill=(15, 12, 10), width=3)
        draw.rectangle([crack_x - 12, 35, crack_x + 22, 135], outline="#f59e0b", width=2)
        draw.text((crack_x - 12, 20), "CRACK DILATION [2.4cm]", fill="#f59e0b")

        # Small loose pebbles and gravel tumbling down face
        t_frac = (i / n_frames)
        for p_idx, p_offset in enumerate([-15, 0, 18]):
            peb_y = int(95 + ((t_frac * 1.5 + p_idx * 0.25) % 1.0) * 110)
            peb_x = crack_x + p_offset + int(np.sin(peb_y * 0.08) * 8)
            draw.ellipse([(peb_x - 5, peb_y - 4), (peb_x + 5, peb_y + 4)], fill=(120, 110, 95))
            draw.line([(peb_x, peb_y - 8), (peb_x, peb_y)], fill="#f59e0b", width=1)
            # Bounding box around active pebble group
            if peb_y > 110 and peb_y < 190:
                draw.rectangle([peb_x - 12, peb_y - 10, peb_x + 12, peb_y + 10], outline="#f59e0b", width=1)
                draw.text((peb_x - 12, peb_y - 20), "RAVELLING [1.8m/s]", fill="#f59e0b")

        # CCTV HUD
        draw.rectangle([(0, 0), (w, 24)], fill=(10, 15, 26))
        draw.rectangle([(0, h - 22), (w, h)], fill=(10, 15, 26))
        draw.text((12, 5), f"● REC | CAM-01 [NORTH WALL BENCH 4] | 2026-11-15 12:12:{i:02d} | 25 FPS", fill=(56, 189, 248))
        draw.text((12, h - 17), "STATUS: MILD SLOPE RAVELLING | LOOSE GRAVEL DETECTED | TARP: LEVEL 2 YELLOW", fill="#fbbf24")

        img.save(os.path.join(TMP_FRAMES, f"frame_{i:03d}.png"))

    subprocess.run([
        "ffmpeg", "-y", "-framerate", "15",
        "-i", os.path.join(TMP_FRAMES, "frame_%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path
    ], capture_output=True, check=True)
    shutil.rmtree(TMP_FRAMES, ignore_errors=True)
    print(f"[✓] Created: {out_path} ({os.path.getsize(out_path)} bytes)")


def render_moderate_rockfall_video(out_path: str, n_frames: int = 45):
    """Renders moderate wedge dilation: widening fissure and sliding rock slabs."""
    os.makedirs(TMP_FRAMES, exist_ok=True)
    w, h = 640, 360

    for i in range(n_frames):
        img = Image.new("RGB", (w, h), color=(44, 40, 34))
        draw = ImageDraw.Draw(img)
        draw_realistic_highwall(draw, w, h, seed=42)

        # Widened scarp fissure (8cm)
        crack_x = int(w * 0.44)
        draw.line([(crack_x, 30), (crack_x + 16, 90), (crack_x + 8, 160)], fill=(12, 8, 6), width=6)
        draw.rectangle([crack_x - 16, 25, crack_x + 30, 165], outline="#f97316", width=2)
        draw.text((crack_x - 16, 10), "SCARP EXTENSION [8.2cm]", fill="#f97316")

        # Sliding rock slabs and trailing dust plume
        t_frac = (i / n_frames)
        slab_y = int(105 + (t_frac ** 1.3) * 140)
        slab_x = crack_x + int(np.sin(t_frac * 2 * np.pi) * 14) + 12

        # Dust plume
        for d in range(25):
            dx = slab_x + np.random.randint(-30, 30)
            dy = max(0, slab_y - np.random.randint(5, 50))
            draw.ellipse([(dx, dy), (dx + 6, dy + 6)], fill=(145, 135, 120))

        # Tumbling rock slabs
        draw.polygon([(slab_x - 18, slab_y - 12), (slab_x + 20, slab_y - 8), (slab_x + 14, slab_y + 16), (slab_x - 16, slab_y + 12)], fill=(90, 82, 72))
        draw.line([(slab_x, slab_y - 22), (slab_x, slab_y)], fill="#f97316", width=2)

        # AI Detection Bounding Box
        draw.rectangle([slab_x - 26, slab_y - 20, slab_x + 28, slab_y + 22], outline="#f97316", width=2)
        draw.text((slab_x - 26, slab_y - 34), "SLABBING [4.2 m/s | CONF: 91%]", fill="#f97316")

        # CCTV HUD
        draw.rectangle([(0, 0), (w, 24)], fill=(10, 15, 26))
        draw.rectangle([(0, h - 22), (w, h)], fill=(10, 15, 26))
        draw.text((12, 5), f"● REC | CAM-01 [NORTH WALL BENCH 4] | 2026-11-15 12:18:{i:02d} | 25 FPS", fill=(56, 189, 248))
        draw.text((12, h - 17), "STATUS: MODERATE WEDGE DILATION | ROCK DETACHMENT | TARP: LEVEL 3 ORANGE", fill="#fb923c")

        img.save(os.path.join(TMP_FRAMES, f"frame_{i:03d}.png"))

    subprocess.run([
        "ffmpeg", "-y", "-framerate", "15",
        "-i", os.path.join(TMP_FRAMES, "frame_%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path
    ], capture_output=True, check=True)
    shutil.rmtree(TMP_FRAMES, ignore_errors=True)
    print(f"[✓] Created: {out_path} ({os.path.getsize(out_path)} bytes)")


def render_severe_rockfall_video(out_path: str, n_frames: int = 45):
    """Renders major bench collapse: tumbling boulder cascades with red AI bounding boxes."""
    os.makedirs(TMP_FRAMES, exist_ok=True)
    w, h = 640, 360

    for i in range(n_frames):
        img = Image.new("RGB", (w, h), color=(44, 40, 34))
        draw = ImageDraw.Draw(img)
        draw_realistic_highwall(draw, w, h, seed=42)

        # Massive structural tension chasm
        crack_x = int(w * 0.42)
        draw.line([(crack_x, 20), (crack_x + 24, 100), (crack_x + 12, 190)], fill=(10, 6, 4), width=10)
        draw.rectangle([crack_x - 22, 15, crack_x + 40, 195], outline="#ef4444", width=3)
        draw.text((crack_x - 22, 2), "CRITICAL SHEAR SCARP [18.5cm]", fill="#ef4444")

        # Heavy cascading boulders with motion blur
        t_frac = (i / n_frames)
        boulder_y = int(90 + (t_frac ** 1.6) * 190)
        boulder_x = crack_x + int(np.sin(t_frac * 3 * np.pi) * 28) + 16

        # Massive dust cloud
        for d in range(50):
            dx = boulder_x + np.random.randint(-55, 55)
            dy = max(0, boulder_y - np.random.randint(10, 80))
            draw.ellipse([(dx, dy), (dx + 10, dy + 10)], fill=(165, 152, 136))

        # Main falling boulder mass
        draw.ellipse([(boulder_x - 24, boulder_y - 18), (boulder_x + 24, boulder_y + 18)], fill=(75, 68, 58))
        draw.ellipse([(boulder_x - 16, boulder_y - 12), (boulder_x + 16, boulder_y + 12)], fill=(105, 96, 82))
        draw.line([(boulder_x, boulder_y - 35), (boulder_x, boulder_y)], fill="#ef4444", width=3)

        # Trailing secondary rock blocks
        for b_sub in range(3):
            sub_y = max(0, boulder_y - 35 - b_sub * 25)
            sub_x = boulder_x + (b_sub - 1) * 22
            draw.rectangle([(sub_x - 10, sub_y - 8), (sub_x + 10, sub_y + 8)], fill=(85, 78, 68))

        # AI Detection Bounding Box (Crimson Critical)
        draw.rectangle([boulder_x - 36, boulder_y - 30, boulder_x + 38, boulder_y + 32], outline="#ef4444", width=3)
        draw.text((boulder_x - 36, boulder_y - 46), "ROCKFALL CASCADE [8.4 m/s | CONF: 98%]", fill="#ef4444")

        # CCTV HUD Alert Banner
        draw.rectangle([(0, 0), (w, 24)], fill=(10, 15, 26))
        draw.rectangle([(0, h - 22), (w, h)], fill=(10, 15, 26))
        draw.text((12, 5), f"● REC | CAM-01 [NORTH WALL BENCH 4] | 2026-11-15 12:24:{i:02d} | 25 FPS", fill=(56, 189, 248))
        draw.text((12, h - 17), "🚨 CRITICAL ROCKFALL DETACHMENT IN PROGRESS | EVACUATION MANDATED | TARP: LEVEL 4 RED", fill="#f87171")

        img.save(os.path.join(TMP_FRAMES, f"frame_{i:03d}.png"))

    subprocess.run([
        "ffmpeg", "-y", "-framerate", "15",
        "-i", os.path.join(TMP_FRAMES, "frame_%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path
    ], capture_output=True, check=True)
    shutil.rmtree(TMP_FRAMES, ignore_errors=True)
    print(f"[✓] Created: {out_path} ({os.path.getsize(out_path)} bytes)")


if __name__ == "__main__":
    print("[*] Generating high-definition surveillance MP4 video suite...")
    render_normal_ops_video(os.path.join(VIDEOS_DIR, "normal_ops.mp4"))
    render_mild_rockfall_video(os.path.join(VIDEOS_DIR, "mild_rockfall.mp4"))
    render_moderate_rockfall_video(os.path.join(VIDEOS_DIR, "moderate_rockfall.mp4"))
    render_severe_rockfall_video(os.path.join(VIDEOS_DIR, "severe_rockfall.mp4"))
    print("[✓] All 4 surveillance video feeds generated successfully.")
