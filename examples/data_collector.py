"""
Universal UI Data Collection

Automated pipeline for collecting visual UI datasets across any platform.
Captures screenshots, parses UI elements with Omniparser, and randomly
navigates to discover new screens.

Uses gui_diff to detect when the screen has changed after a click,
indicating a new state worth recording.

Usage:
    python data_collector.py --output dataset/ --num-screens 100
"""

from dotenv import load_dotenv
load_dotenv()

import os
import sys
# Allow running from examples/ directory by adding parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import random
import argparse
import base64
from io import BytesIO
from PIL import Image
from hidagent.hidagent import (
    get_screenshot, click_mouse, recognize_gui_elements, gui_diff
)


def click_random_element(parsed_elements, image_size):
    """Click a random parsed UI element."""
    if not parsed_elements:
        return False

    element = random.choice(parsed_elements)
    width, height = image_size

    if "location" in element:
        x = element["location"].get("x", 0.5)
        y = element["location"].get("y", 0.5)
    elif "bbox" in element:
        bbox = element["bbox"]
        x = (bbox[0] + bbox[2]) / 2
        y = (bbox[1] + bbox[3]) / 2
    else:
        x, y = random.random(), random.random()

    px = int(x * width)
    py = int(y * height)

    # click_mouse() expects calibration-space coordinates. Calibration was
    # done at half-resolution (image_resize_factor=2), so scale down.
    click_mouse(int(px * 0.5), int(py * 0.5))
    print(f"  Clicked element at ({px}, {py}): {element.get('content', 'unknown')[:50]}")
    return True


def save_screen(output_dir, screen_id, image, parsed_elements):
    """Save a screen capture and its parsed elements to the dataset."""
    image.save(os.path.join(output_dir, f"screen_{screen_id:04d}.png"))

    with open(os.path.join(output_dir, f"screen_{screen_id:04d}.json"), "w") as f:
        json.dump({
            "screen_id": screen_id,
            "num_elements": len(parsed_elements),
            "elements": parsed_elements,
        }, f, indent=2)


def collect_data(output_dir, num_screens=100, max_clicks_per_screen=5):
    """Main data collection loop."""
    os.makedirs(output_dir, exist_ok=True)

    collected = 0
    total_clicks = 0

    print(f"Collecting {num_screens} screens to {output_dir}/")

    while collected < num_screens:
        print(f"\n--- Screen {collected + 1}/{num_screens} ---")

        image = get_screenshot()
        parsed_elements = recognize_gui_elements(image)
        print(f"  Parsed {len(parsed_elements)} elements")

        save_screen(output_dir, collected, image, parsed_elements)
        collected += 1
        print(f"  Saved screen {collected}")

        for click_attempt in range(max_clicks_per_screen):
            prev_image = image

            if not click_random_element(parsed_elements, image.size):
                break
            total_clicks += 1
            time.sleep(1)

            image = get_screenshot()

            changes = gui_diff(prev_image, image)
            if changes:
                print(f"  Screen changed! New state detected.")
                break
            else:
                print(f"  No significant change, trying another click...")

    print(f"\nDone! Collected {collected} screens with {total_clicks} clicks.")
    print(f"Dataset saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect UI dataset via HIDAgent")
    parser.add_argument("--output", default="dataset", help="Output directory")
    parser.add_argument("--num-screens", type=int, default=100,
                        help="Number of screens to collect")
    args = parser.parse_args()
    collect_data(args.output, args.num_screens)
