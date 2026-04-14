"""
Simple Simulator demo — boots the simulator and moves the mouse around.

Usage:
    cd hidagent/examples
    python sim_demo.py
"""

import os
import sys
# Allow running from any directory by adding parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from hidagent.simulator import SimulatorBackend

sim = SimulatorBackend()
sim.boot()

# Get device dimensions
img = sim.screenshot()
w, h = img.size
print(f"Device screen: {w}x{h}")

# Use 10% margins to stay well inside the curved corners
margin_x = int(w * 0.10)
margin_y = int(h * 0.10)

positions = [
    (w // 2, h // 2, "center"),
    (margin_x, margin_y, "top-left"),
    (w - margin_x, margin_y, "top-right"),
    (w - margin_x, h - margin_y, "bottom-right"),
    (margin_x, h - margin_y, "bottom-left"),
    (margin_x, margin_y, "back to top-left"),
    (w // 2, h // 2, "center"),
]

for x, y, label in positions:
    print(f"  Moving to {label} ({x}, {y})...")
    sim.move(x, y)
    time.sleep(0.7)

# Click center
print("Clicking center...")
sim.click(w // 2, h // 2)
time.sleep(1)

print("Done!")
