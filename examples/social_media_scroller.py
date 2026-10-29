"""
The Social Media Scroller

Records a social media feed by automating screen recording and scrolling
on the target device (e.g., iPhone). The resulting video can be processed
to extract individual posts, remove ads, and reorder content.

Workflow:
  1. Start screen recording on the target device
  2. Open the social media app (e.g., Instagram)
  3. Navigate to the main feed
  4. Scroll at regular intervals for a set duration
  5. Stop recording

Usage:
    python social_media_scroller.py --app instagram --duration 120
"""

from dotenv import load_dotenv
load_dotenv()

import os
import sys
# Allow running from examples/ directory by adding parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse
from PIL import Image
from hidagent.hidagent import (
    get_screenshot, click_mouse, run_application, patch_location, drag_down, drag_up
)

# iOS screen recording button template (Control Center, gray circle w/ dot).
_RECORD_BUTTON_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "record-button.png",
)


def _load_record_button():
    """Load the record button template image from the repo root."""
    return Image.open(_RECORD_BUTTON_PATH)


def start_screen_recording():
    """Start screen recording on iOS via Control Center.

    Uses patch_location to find the screen recording button in Control Center,
    so it works regardless of where the button is positioned on different devices.
    """
    print("  Starting screen recording...")
    # Swipe down from top-right for Control Center (iPhone with notch)
    drag_down(440, 0)
    time.sleep(1)

    # Take screenshot and find the record button
    screenshot = get_screenshot()
    record_btn = _load_record_button()
    location = patch_location(record_btn, screenshot)

    if location:
        # Click the center of the detected button (screenshot pixels).
        btn_x = location["x"] + location["width"] // 2
        btn_y = location["y"] + location["height"] // 2
        print(f"  Found record button at ({btn_x}, {btn_y}) "
              f"confidence={location['confidence']:.2f}")
        click_mouse(btn_x, btn_y)
    else:
        # Fallback to approximate position if button not found
        print("  Record button not found, using fallback position")
        click_mouse(400, 800)

    time.sleep(1)  # iOS gives a 3-second countdown


def stop_screen_recording():
    """Stop screen recording by tapping the red status bar."""
    print("  Stopping screen recording...")
    # Tap the red recording indicator at the top of screen
    click_mouse(400, 10)
    time.sleep(1)

    # Confirm stop
    click_mouse(400, 600)
    time.sleep(1)


def open_app(app_name):
    """Launch an app using spotlight/search."""
    print(f"  Opening {app_name}...")
    run_application(app_name)
    time.sleep(1)


def scroll_feed(scroll_amount_y, pause_between=1.0):
    """Scroll the feed by a specified amount."""
    drag_up(300, 300)
    time.sleep(pause_between)


def navigate_to_feed(app_name):
    """Navigate to the main feed within the app.

    Taps the home/feed icon (typically bottom-left of screen).
    """
    print("  Navigating to main feed...")
    # Bottom-left icon is typically the home/feed tab
    click_mouse(60, 940)
    time.sleep(2)


def record_feed(app_name="instagram", duration=20, scroll_interval=1.5):
    """Main recording workflow."""
    print(f"Recording {app_name} feed for {duration}s")
    print(f"  Scroll interval: {scroll_interval}s")
    print()

    # Step 1: Start recording
    start_screen_recording()
    print("  Recording started.")

    # Step 2: Open the app
    open_app(app_name)

    # Step 3: Navigate to feed
    navigate_to_feed(app_name)

    # Step 4: Scroll for the specified duration
    print(f"  Scrolling for {duration}s...")
    start_time = time.time()
    scroll_count = 0

    while time.time() - start_time < duration:
        scroll_feed(scroll_amount_y=250, pause_between=scroll_interval)
        scroll_count += 1
        elapsed = int(time.time() - start_time)
        if scroll_count % 10 == 0:
            print(f"    {elapsed}s / {duration}s ({scroll_count} scrolls)")

    print(f"  Scrolling complete: {scroll_count} scrolls in {duration}s")

    # Step 5: Stop recording
    stop_screen_recording()
    print("  Recording stopped.")

    # Step 6: Open scrollshen app
    print("  Opening scrollshen app...")
    open_app("scrollshen")
    print("  scrollshen app opened.")

    print(f"\nDone! Video saved to device camera roll.")
    print("Use a video parser to extract individual posts from the recording.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record social media feed via HIDAgent")
    parser.add_argument("--app", default="instagram",
                        help="App to record (default: instagram)")
    parser.add_argument("--duration", type=int, default=120,
                        help="Recording duration in seconds (default: 120)")
    parser.add_argument("--scroll-interval", type=float, default=1.5,
                        help="Seconds between scrolls (default: 1.5)")
    args = parser.parse_args()

    record_feed(args.app, args.duration, args.scroll_interval)
