from dotenv import load_dotenv
load_dotenv()

import os
import io
import sys
import json
import time
import glob
import threading
import base64
from io import BytesIO

import numpy as np
import cv2
from PIL import Image
import serial

# Optional imports — loaded lazily when features that need them are used.
# This lets the core library (click_mouse, get_screenshot, etc.) work without
# installing heavy dependencies like Omniparser, Ollama, AVFoundation, etc.
try:
    import util.omniparser  # noqa: F401
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import ollama
except ImportError:
    ollama = None

try:
    import AVFoundation
    import objc
except ImportError:
    AVFoundation = None
    objc = None

try:
    import usb.core
    import usb.util
except ImportError:
    pass

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from flask import Flask, send_file, render_template_string
except ImportError:
    Flask = None

# HID-to-pixel ratios (set by calibrate() or manually)
# Default values for laptop with studio display, with 2x resize for Anthropic
hid_x_ratio = 2.1 * 2
hid_y_ratio = 2.15 * 2

# Calibration file path — stores ratios so calibration persists across runs
_CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "calibration.json")


def _detect_current_screen():
    """Capture a frame and return (screen_w, screen_h) or None.

    Uses the same detection logic as test_calibration.detect_screen_bounds
    so saved calibrations are found consistently.
    """
    try:
        cap = cv2.VideoCapture(0)
        for _ in range(5):
            cap.read()
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Match test_calibration.detect_screen_bounds: low threshold + closing
        _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        _, _, w, h = cv2.boundingRect(largest)
        return w, h
    except Exception:
        return None


def _load_calibrations():
    """Load all saved calibrations from file."""
    if not os.path.exists(_CALIBRATION_FILE):
        return []
    with open(_CALIBRATION_FILE) as f:
        data = json.load(f)
    # Support both old format (single dict) and new format (list)
    if isinstance(data, dict):
        return [data]
    return data


def _save_calibrations(calibrations):
    """Save all calibrations to file."""
    with open(_CALIBRATION_FILE, "w") as f:
        json.dump(calibrations, f, indent=2)


def _find_matching_calibration(calibrations, screen_w, screen_h):
    """Find a saved calibration matching the given screen dimensions (±5%)."""
    for cal in calibrations:
        if "screen_w" not in cal or "screen_h" not in cal:
            continue
        w_match = abs(screen_w - cal["screen_w"]) / max(cal["screen_w"], 1) < 0.05
        h_match = abs(screen_h - cal["screen_h"]) / max(cal["screen_h"], 1) < 0.05
        if w_match and h_match:
            return cal
    return None


def _run_calibration_script():
    """Run test_calibration.py as a subprocess to calibrate the current device."""
    import subprocess

    # Find the calibration script — it lives at the repo root
    script_candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "test_calibration.py"),
        os.path.join(os.path.dirname(__file__), "..", "..", "test_calibration.py"),
    ]
    script_path = None
    for candidate in script_candidates:
        candidate = os.path.abspath(candidate)
        if os.path.exists(candidate):
            script_path = candidate
            break

    if not script_path:
        print("ERROR: Could not find test_calibration.py to auto-calibrate.")
        return False

    print(f"\nRunning auto-calibration: {script_path}")
    print("This will take a few minutes...\n")

    try:
        result = subprocess.run(
            [sys.executable, "-u", script_path],
            cwd=os.path.dirname(script_path),
            check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"Calibration failed: {e}")
        return False


def _load_calibration(auto_calibrate=True):
    """Load calibration matching the currently connected screen.

    Stores multiple calibrations keyed by screen resolution, so switching
    between devices reuses previous calibrations automatically. If no
    matching calibration is found and auto_calibrate is True, runs the
    calibration procedure automatically.
    """
    global hid_x_ratio, hid_y_ratio

    calibrations = _load_calibrations()
    current = _detect_current_screen()

    if current is None:
        if calibrations:
            print("No screen detected — using default calibration ratios.")
        return False

    current_w, current_h = current
    match = _find_matching_calibration(calibrations, current_w, current_h)

    if match:
        hid_x_ratio = match["hid_x_ratio"]
        hid_y_ratio = match["hid_y_ratio"]
        print(f"Loaded calibration for {match['screen_w']}x{match['screen_h']} screen: "
              f"x_ratio={hid_x_ratio:.4f}, y_ratio={hid_y_ratio:.4f}")
        return True

    # No match — offer to auto-calibrate
    saved_sizes = [f"{c['screen_w']}x{c['screen_h']}" for c in calibrations
                   if "screen_w" in c]
    print(f"No calibration for {current_w}x{current_h} screen. "
          f"Saved: {', '.join(saved_sizes) or 'none'}.")

    if auto_calibrate:
        if _run_calibration_script():
            # Reload after calibration
            calibrations = _load_calibrations()
            match = _find_matching_calibration(calibrations, current_w, current_h)
            if match:
                hid_x_ratio = match["hid_x_ratio"]
                hid_y_ratio = match["hid_y_ratio"]
                print(f"Loaded new calibration: "
                      f"x_ratio={hid_x_ratio:.4f}, y_ratio={hid_y_ratio:.4f}")
                return True

    print("Run calibrate() for this device.")
    return False


def _save_calibration(x_ratio, y_ratio, metadata=None):
    """Save calibration, adding or updating the entry for this screen size."""
    cal = {
        "hid_x_ratio": x_ratio,
        "hid_y_ratio": y_ratio,
    }
    if metadata:
        cal.update(metadata)

    calibrations = _load_calibrations()

    # Update existing entry for this screen size, or append new one
    screen_w = cal.get("screen_w")
    screen_h = cal.get("screen_h")
    if screen_w and screen_h:
        existing = _find_matching_calibration(calibrations, screen_w, screen_h)
        if existing:
            calibrations.remove(existing)
    calibrations.append(cal)

    _save_calibrations(calibrations)


# Try to load saved calibration on import. Don't auto-calibrate here: that
# sweeps the pointer for minutes, and a dark app screen is easily mistaken
# for an unknown device. Call calibrate() explicitly instead.
_load_calibration(auto_calibrate=False)


# ============================================================
# PUBLIC API
# ============================================================

def get_screenshot():
    """Capture a screenshot from the target device via HDMI capture.

    Identifies the connected USB capture device, captures one frame,
    and returns it as a PIL Image object.
    """
    return get_image_from_usb()


def move_mouse(x, y):
    """Move the mouse to location (x, y) on the target device.

    Coordinates are in the target device's screen pixel space
    (after applying the calibrated HID ratio).
    """
    cmd = json.dumps({
        "type": "move",
        "x": int(x * hid_x_ratio),
        "y": int(y * hid_y_ratio),
    })
    send_serial_message(f"{cmd}\n")


def click_mouse(x, y, button="left"):
    """Move the mouse to (x, y) and click.

    Args:
        x, y: Target device screen coordinates.
        button: "left" (default) or "right".
    """
    cmd = json.dumps({
        "type": "click",
        "x": int(x * hid_x_ratio),
        "y": int(y * hid_y_ratio),
    })
    send_serial_message(f"{cmd}\n")


def move_mouse_visual(x, y, tol=1.0):
    """Move the pointer onto (x, y) by watching it in the capture.

    Unlike move_mouse(), this doesn't depend on a calibration: it finds the
    pointer on screen and corrects until it is within tol pixels (see
    cursor.py). Coordinates are in get_screenshot() pixels.

    Returns {"cmd", "pos", "err", "iters", "verified"}, or None if the
    pointer was never found (e.g. the target is under the iPhone's Dynamic
    Island). verified is False when the pointer was invisible at the final
    spot (e.g. over a black icon) and pos was estimated from nearby landings.
    """
    from .cursor import get_positioner
    return get_positioner().goto(x, y, tol=tol)


def click_mouse_visual(x, y, tol=1.0, max_err=5.0):
    """Click at (x, y) after steering the pointer there visually.

    The click replays the exact HID command that was seen to land on the
    target, so it hits the same spot. Returns the same dict as
    move_mouse_visual() plus "clicked". It does not click (clicked=False)
    if the pointer ended more than max_err pixels from the target, and
    returns None (without clicking) if the pointer was never found.
    """
    from .cursor import get_positioner, hid_click
    result = get_positioner().goto(x, y, tol=tol)
    if result is not None:
        result["clicked"] = result["err"] <= max_err
        if result["clicked"]:
            hid_click(*result["cmd"])
        else:
            print(f"click_mouse_visual: pointer ended {result['err']:.1f}px "
                  f"from ({x}, {y}); not clicking")
    return result


def drag_down(x, y):
    """Drag downward starting from (x, y) on the target device.

    Moves to (x, y), presses left button, drags down, and releases. Used
    for gestures like swiping down to open Control Center on iPhone.
    """
    cmd = json.dumps({
        "type": "dragdown",
        "x": int(x * hid_x_ratio),
        "y": int(y * hid_y_ratio),
    })
    send_serial_message(f"{cmd}\n")


def drag_up(x, y):
    """Drag upward starting from (x, y) on the target device.

    Moves to (x, y), presses left button, drags up, and releases. Used
    for gestures like scrolling a feed.
    """
    cmd = json.dumps({
        "type": "dragup",
        "x": int(x * hid_x_ratio),
        "y": int(y * hid_y_ratio),
    })
    send_serial_message(f"{cmd}\n")


def type_text(text):
    """Send a string of characters to the target device.

    Args:
        text: The string to type.
    """
    cmd = json.dumps({"type": "type", "text": text})
    send_serial_message(f"{cmd}\n")


def keypress(keys):
    """Perform a single keypress (keydown + keyup) on the target device.

    Args:
        keys: A list of keys to press simultaneously. Keys can be
            characters or control key names (e.g., ["ctrl", "c"],
            ["cmd", "space"], ["enter"]).
    """
    if isinstance(keys, str):
        keys = [keys]
    # Join keys with "+" for the HID bridge to parse
    key_str = "+".join(keys)
    cmd = json.dumps({"type": "key", "text": key_str})
    send_serial_message(f"{cmd}\n")


def run_application(application_name):
    """Launch an application on the target device.

    Works cross-platform by issuing: cmd/win + spacebar, typing the
    application name, then pressing enter.

    Args:
        application_name: Name of the application to launch
            (e.g., "Safari", "Notes", "Chrome").
    """
    cmd = json.dumps({"type": "run", "text": application_name})
    send_serial_message(f"{cmd}\n")


def hid_calibration(steps=None, step_size=None):
    """Run the HID-to-pixel calibration procedure.

    Uses optical flow to detect cursor motion and computes the
    mapping ratio between HID coordinates and screen pixels.
    Results are saved to calibration.json for future sessions.

    Args:
        steps: Number of measurement steps per axis (default: ~14 for X, ~15 for Y).
        step_size: HID units between measurement points (default: 40 for X, 80 for Y).

    Returns:
        dict with 'hid_x_ratio', 'hid_y_ratio', 'x_r2', 'y_r2' or None on failure.
    """
    return calibrate(image_resize_factor=2, verbose=True)


def recognize_gui_elements(screenshot_img):
    """Parse a screenshot to identify UI elements.

    Takes a screenshot image of a GUI and returns a JSON list of
    recognized user interface elements with their location, type,
    and content (e.g., labels).

    Currently implemented as a wrapper around Omniparser.

    Args:
        screenshot_img: PIL Image of the screenshot.

    Returns:
        List of dicts, each with keys like 'type', 'location', 'content'.
    """
    # Locate the repo root (where `util/` and `weights/` live).
    repo_root = None
    for candidate in [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    ]:
        if os.path.isdir(os.path.join(candidate, "util")):
            repo_root = candidate
            break
    if repo_root is None:
        raise RuntimeError("Could not locate repo root with util/ and weights/")

    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    config = {
        "som_model_path": os.path.join(repo_root, "weights/icon_detect/model.pt"),
        "caption_model_name": "florence2",
        "caption_model_path": os.path.join(repo_root, "weights/icon_caption_florence"),
        "BOX_TRESHOLD": 0.05,
    }
    import util.omniparser
    omni_parser = util.omniparser.Omniparser(config=config)

    # Upscale small images before passing to Omniparser. The YOLO icon
    # detection model works much better on images ~1000+ pixels on the
    # long edge. iPhone screenshots from HDMI capture are often ~450x980
    # which is too small for reliable detection.
    upscale = 1
    target_long_edge = 1600
    long_edge = max(screenshot_img.size)
    if long_edge < target_long_edge:
        upscale = target_long_edge / long_edge
        new_w = int(screenshot_img.width * upscale)
        new_h = int(screenshot_img.height * upscale)
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        img_for_parser = screenshot_img.resize((new_w, new_h), resample)
    else:
        img_for_parser = screenshot_img

    buffered = BytesIO()
    img_for_parser.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    _, parsed_content_list = omni_parser.parse(image_base64)
    return parsed_content_list


def llm_screenshot_query(screenshot_img, query, model=None):
    """Query an LLM about a screenshot.

    Takes a screenshot image and a text query, and returns a response.
    By default, runs the screenshot through recognize_gui_elements and
    passes the results + image to Gemma:27b locally via Ollama.

    Args:
        screenshot_img: PIL Image of the screenshot.
        query: Text query about the screenshot.
        model: LLM to use. Options:
            - None or "gemma" (default): Gemma:27b via Ollama (local)
            - "claude": Anthropic Claude API
            - "gpt5": OpenAI GPT-5 API

    Returns:
        String response from the LLM.
    """
    buffered = BytesIO()
    screenshot_img.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    if model == "claude":
        from anthropic import Anthropic
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model="claude-opus-4-1-20250805",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_base64,
                    }},
                ],
            }],
        )
        return response.content[0].text

    elif model == "gpt5":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }},
                ],
            }],
        )
        return response.choices[0].message.content

    else:
        # Default: local Gemma via Ollama with GUI element context
        parsed_elements = recognize_gui_elements(screenshot_img)
        prompt = f"{query}\n\nUI Elements:\n{json.dumps(parsed_elements, indent=2)}"
        response = ollama.chat(
            model="gemma:27b",
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]


def gui_diff(screenshot1, screenshot2):
    """Compare two screenshots and return the location/size of differences.

    Args:
        screenshot1: PIL Image (first screenshot).
        screenshot2: PIL Image (second screenshot, same resolution).

    Returns:
        List of dicts with keys 'x', 'y', 'width', 'height' for each
        changed region, or empty list if no differences detected.
    """
    arr1 = np.array(screenshot1)
    arr2 = np.array(screenshot2)

    # Crop both to matching dimensions. The HDMI capture's border detection
    # can vary by 1-2 pixels between captures, which shouldn't cause a hard
    # error — just align the images and diff the overlapping area.
    min_h = min(arr1.shape[0], arr2.shape[0])
    min_w = min(arr1.shape[1], arr2.shape[1])
    arr1 = arr1[:min_h, :min_w]
    arr2 = arr2[:min_h, :min_w]

    diff = cv2.absdiff(arr1, arr2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

    # Use a higher pixel threshold to ignore HDMI capture noise, JPEG
    # compression artifacts, and minor anti-aliasing flicker.
    _, thresh = cv2.threshold(gray_diff, 50, 255, cv2.THRESH_BINARY)

    # Morphological opening removes isolated noise pixels; a large close
    # then merges nearby changed pixels into coherent regions so that
    # many small changes in the same area (e.g., a re-rendered widget)
    # collapse into one big region rather than producing a swarm of
    # tiny bounding boxes.
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)

    # If the total changed area is a small fraction of the image, it's
    # almost certainly sensor/compression noise rather than a real UI
    # change — bail out early.
    total_pixels = min_h * min_w
    changed_pixels = int(np.count_nonzero(thresh))
    if changed_pixels < max(2000, int(total_pixels * 0.005)):
        return []

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # Require each region to be a meaningful chunk of the screen (~0.5%
    # of total pixels or at least 1500 px) AND have substantial mean
    # contrast in the underlying diff.
    min_region_area = max(1500, int(total_pixels * 0.005))
    regions = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_region_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        region_diff = gray_diff[y:y + h, x:x + w]
        if region_diff.size == 0 or float(region_diff.mean()) < 20:
            continue
        regions.append({"x": x, "y": y, "width": w, "height": h})

    return regions


def patch_location(patch_img, screenshot):
    """Find the location of a patch image within a screenshot.

    Uses template matching (similar to Sikuli) to find where a small
    image appears within a larger screenshot.

    Args:
        patch_img: PIL Image of the patch to find.
        screenshot: PIL Image of the full screenshot to search in.

    Returns:
        Dict with 'x', 'y', 'width', 'height' of the match location,
        or None if not found.
    """
    patch_arr = cv2.cvtColor(np.array(patch_img), cv2.COLOR_RGB2BGR)
    screen_arr = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    # Convert to grayscale for matching
    patch_gray = cv2.cvtColor(patch_arr, cv2.COLOR_BGR2GRAY)
    screen_gray = cv2.cvtColor(screen_arr, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(screen_gray, patch_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Threshold for a good match
    if max_val < 0.7:
        return None

    h, w = patch_gray.shape
    return {
        "x": max_loc[0],
        "y": max_loc[1],
        "width": w,
        "height": h,
        "confidence": float(max_val),
    }


# ============================================================
# INTERNAL / LEGACY
# ============================================================

data = {
    "type": "example",
    "x": 10,
    "y": 20,
    "text": "Sample text"
}

def main():

    config = {
        "som_model_path": "weights/icon_detect/model.pt",
        "caption_model_name": "florence2",
        "caption_model_path": "weights/icon_caption_florence",
        "BOX_TRESHOLD": 0.05
    }

    image = get_image_from_usb()

    # The `util` package lives at the repo root (outside hidagent/).
    # Add candidate paths so `import util.omniparser` works regardless
    # of where this function is called from.
    _repo_root_candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    ]
    for _p in _repo_root_candidates:
        if os.path.isdir(os.path.join(_p, "util")) and _p not in sys.path:
            sys.path.insert(0, _p)
            break
    import util.omniparser
    omni_parser = util.omniparser.Omniparser(config=config)

    convert_to_html(image, omni_parser)
    #getnextaction(image, "click the 9 button", omni_parser)

    # Save the captured image to disk
    image.save("captured_image.png")
    print("Image saved as captured_image.png")

    app = Flask(__name__)

    @app.route("/")
    def index():
        # Serve a simple HTML page displaying the image
        html = """
                                                        </body>        </html>
        """
        return render_template_string(html)

    @app.route("/image")
    def serve_image():
        # Serve the captured image file
        return send_file("captured_image.png", mimetype="image/png")

    # Start the Flask server
    app.run(host="0.0.0.0", port=8080, debug=True)



    return

    # Example: Move mouse to coordinate (x=300, y=200) via serial command
    move_command = json.dumps({"type": "click", "x": int(20*hid_x_ratio), "y": int(10*hid_y_ratio)})
    send_serial_message(f"{move_command}\n")

    #calibrate()

    sampling_loop(
        model="claude-opus-4-1-20250805",
        messages=[
            {"role": "user", "content": "Open a web browser, find the website for CHI 2026, then find the paper submission instructions."}
#            {"role": "user", "content": "Open a new note and enter the text 'remember this!' in it."}
        ],
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=4096,
        tool_version="20250124",
        thinking_budget=1024,
        max_iterations=10  # Limit iterations to prevent infinite loops
    )

    return

    # {"type": "dragdown", "x": 200, "y": 7, "text": "dragdown”}
    data["type"] = "dragdown"
    data["x"] = 200  # Example normalized x coordinate
    data["y"] = 7  # Example normalized y coordinate

    data["type"] = "run"
    data["text"] = "notes"  # Example normalized y coordinate


    json_data = json.dumps(data)
    message = f"{json_data}\n"

    # '{"type": "dragdown", "x": 200, "y": 7, "text": "dragdown"}'

    send_serial_message(message)


    return

    config = {
        "som_model_path": "weights/icon_detect/model.pt",
        "caption_model_name": "florence2",
        "caption_model_path": "weights/icon_caption_florence",
        "BOX_TRESHOLD": 0.05
    }

    image = get_image_from_usb()
    image.show()

    #image_path = 'imgs/ig_screenshot1.png'
    #image = Image.open(image_path)
    #image.show()

    # The `util` package lives at the repo root (outside hidagent/).
    # Add candidate paths so `import util.omniparser` works regardless
    # of where this function is called from.
    _repo_root_candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    ]
    for _p in _repo_root_candidates:
        if os.path.isdir(os.path.join(_p, "util")) and _p not in sys.path:
            sys.path.insert(0, _p)
            break
    import util.omniparser
    omni_parser = util.omniparser.Omniparser(config=config)

    action_description = input("Enter action description (default: click the search icon): ").strip() or "click the search icon"

    getnextaction(image, action_description, omni_parser)



from anthropic import Anthropic
import uuid


def sampling_loop(
    *,
    model: str,
    messages: list[dict],
    api_key: str,
    max_tokens: int = 4096,
    tool_version: str,
    thinking_budget: int | None = None,
    max_iterations: int = 2,  # Add iteration limit to prevent infinite loops
):
    """
    A simple agent loop for Claude computer use interactions.

    This function handles the back-and-forth between:
    1. Sending user messages to Claude
    2. Claude requesting to use tools
    3. Your app executing those tools
    4. Sending tool results back to Claude
    """
    # Set up tools and API parameters
    client = Anthropic(api_key=api_key)
    beta_flag = "computer-use-2025-01-24" if "20250124" in tool_version else "computer-use-2024-10-22"

    # Configure tools - you should already have these initialized elsewhere
    tools = [
        {"type": f"computer_{tool_version}", "name": "computer", "display_width_px": 960, "display_height_px": 540},
        {
            "name": "runcommand",
            "description": "run a command on the local computer",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The name of the command to run, e.g., notes, safari, chrome, firefox, terminal, mail",
                    }
                },
                "required": ["command"],
            },
        }
        #{"type": f"text_editor_{tool_version}", "name": "str_replace_editor"},
        #{"type": f"bash_{tool_version}", "name": "bash"}
    ]

    # Main agent loop (with iteration limit to prevent runaway API costs)
    iterations = 0
    while True and iterations < max_iterations:
        iterations += 1

        print(f"\n--- Iteration {iterations} ---")

        # Set up optional thinking parameter (for Claude Sonnet 3.7)
        thinking = None
        if thinking_budget:
            thinking = {"type": "enabled", "budget_tokens": thinking_budget}

        # Call the Claude API
        response = client.beta.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            tools=tools,
            betas=[beta_flag],
            thinking=thinking
        )

        # Add Claude's response to the conversation history
        response_content = response.content

        print(response_content)

        messages.append({"role": "assistant", "content": response_content})

        # Check if Claude used any tools
        tool_results = []
        for block in response_content:
            if block.type == "tool_use":
                # run the tool
                result = run_tool(block.name, block.input)

                # Format the result for Claude
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })

                #print(f"Tool results: {tool_results}")

        # If no tools were used, Claude is done - return the final messages
        if not tool_results:
            print("No tools were used, so returning.")
            return messages

        print(f"Tool results being appended.")

        # Add tool results to messages for the next iteration with Claude
        messages.append({"role": "user", "content": tool_results})

def run_tool(tool_name, tool_input):
    print(f"Running tool: {tool_name} with input: {tool_input}")

    if tool_name == "computer" and "action" in tool_input and tool_input["action"] == "screenshot":
        img = get_image_from_usb()
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Resize image if it's too large
        if img.width > 1000 or img.height > 800:
            img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Write the screenshot to a unique filename
        filename = f"imgs/screenshot_{uuid.uuid4().hex}.png"
        with open(filename, "wb") as f:
            f.write(base64.b64decode(img_str))
        print(f"Screenshot saved to {filename}")

        print(f"Captured screenshot of size: {img.size}")
        # Return the content in the format Claude Computer Use API expects
        return [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_str
                }
            }
        ]
    elif tool_name == "runcommand" and "command" in tool_input:
        command = tool_input["command"]
        message = f'{{"type": "run", "text": "{command}"}}\n'
        send_serial_message(message)
        return [{"type": "text", "text": f"Command executed: {command}"}]

    elif tool_name == "type" and "text" in tool_input:
        text = tool_input["text"]
        message = f'{{"type": "type", "text": "{text}"}}\n'
        send_serial_message(message)
        return [{"type": "text", "text": f"Command executed: {text}"}]

    elif tool_name == "computer" and "action" in tool_input and tool_input["action"] == "left_click":
        x = tool_input["coordinate"][0]
        y = tool_input["coordinate"][1]
        if x is None or y is None:
            return [{"type": "text", "text": "Error: Missing x or y coordinate for left_click"}]
        move_command = json.dumps({"type": "click", "x": int(x*hid_x_ratio), "y": int(y*hid_y_ratio)})
        send_serial_message(f"{move_command}\n")
        return [{"type": "text", "text": f"Clicked at ({x}, {y})"}]

    elif tool_name == "computer" and "action" in tool_input and tool_input["action"] == "wait":
        time.sleep(tool_input.get("duration", 1))

    elif tool_name == "computer" and "action" in tool_input and tool_input["action"] == "key":
        keyname = tool_input["text"]
        message = f'{{"type": "key", "text": "{keyname}"}}\n'
        send_serial_message(message)
        return [{"type": "text", "text": f"Command executed: {keyname}"}]

    # Implement actual tool execution logic here
    if tool_name == "computer":
        return [{"type": "text", "text": "Computer action completed"}]
    elif tool_name == "str_replace_editor":
        return [{"type": "text", "text": "File edited successfully"}]
    elif tool_name == "bash":
        return [{"type": "text", "text": "Command executed: stdout here"}]
    else:
        return [{"type": "text", "text": f"Error: Unknown tool: {tool_name}"}]

_serial_conn = None  # persistent serial connection
_serial_lock = threading.Lock()

# Upper bound on waiting for the HID bridge's reply. Commands take 2-7 s on
# the device (long moves are slowest); readline() returns as soon as the
# reply arrives, so this only matters if the bridge stops responding.
_SERIAL_RESPONSE_TIMEOUT = 15


def _get_serial_conn():
    """Return a persistent serial connection, opening it on first call."""
    global _serial_conn

    if _serial_conn is not None and _serial_conn.is_open:
        return _serial_conn

    serial_devices = glob.glob('/dev/tty.usbserial*')
    if not serial_devices:
        raise FileNotFoundError("No /dev/tty.usbserial device found")
    serial_port = serial_devices[0]

    _serial_conn = serial.Serial(port=serial_port, baudrate=9600,
                                 timeout=_SERIAL_RESPONSE_TIMEOUT)
    return _serial_conn


def release_serial_conn():
    """Release the persistent serial connection (call on shutdown)."""
    global _serial_conn
    if _serial_conn is not None:
        _serial_conn.close()
        _serial_conn = None


def send_serial_message(message, wait_response=True):
    """Send a command to the HID bridge and wait for it to finish.

    The bridge replies "Success" (or "Error") once the command has been
    carried out on the target device, so waiting keeps actions in order and
    means a screenshot taken afterwards shows the finished result. Pass
    wait_response=False only when the caller doesn't care about either;
    unread replies are discarded before the next command is sent.
    """
    print(f"Sending message: {message.strip()}")

    with _serial_lock:
        ser = _get_serial_conn()

        try:
            ser.reset_input_buffer()
            ser.write(message.encode('utf-8'))
            ser.flush()
            print(f"Sent: {message.strip()}")

            if wait_response:
                # errors="replace": a corrupted byte on the UART line
                # shouldn't crash the caller.
                response = ser.readline().decode('utf-8', errors='replace').strip()
                if response:
                    print(f"Received: {response}")
                else:
                    print(f"Warning: no reply from HID bridge within "
                          f"{_SERIAL_RESPONSE_TIMEOUT}s")
                return response

        except KeyboardInterrupt:
            pass


def convert_to_html(image, omni_parser):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    print('Base64 encoded image:', image_base64[:100], '...')  # Print only the first 100 chars

    print('image size:', image.size)

    dino_labled_img, parsed_content_list = omni_parser.parse(image_base64)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY environment variable")

    #print(api_key)

    # Create a client instance
    client = OpenAI(api_key=api_key)

    starting_prompt = (
        "Given the parsed list of elements from a UI screenshot, convert to HTML\n\n" + json.dumps(parsed_content_list, indent=2) + "\n\nReturn only the HTML and nothing else."
    )


    response = client.chat.completions.create(
        model="gpt-5",  # latest multimodal GPT‑4 variant
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": f"{starting_prompt}"
                }
            ]
            }
        ]
        #,temperature=0.7     # optional: tweak output randomness
        #,max_tokens=512,      # optional: control response length
    )

    result = response.choices[0].message.content

    print("GPT-4o response:")
    print(result)

    return dino_labled_img, parsed_content_list


def getnextaction(image, action_description, omni_parser):

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    print('Base64 encoded image:', image_base64[:100], '...')  # Print only the first 100 chars

    print('image size:', image.size)

    dino_labled_img, parsed_content_list = omni_parser.parse(image_base64)

    with open("parsed_content_list.json", "w") as f:
        json.dump(parsed_content_list, f, indent=2)
    print("Parsed content list saved to parsed_content_list.json")



    # Save the labeled image at full size
    labeled_image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))

    labeled_image.show()


    #labeled_image.save("labeled_image.png")
    #print("Labeled image saved as labeled_image.png (full size)")
    #print(parsed_content_list)

    # Encode the labeled image as base64
    #with open("labeled_image.png", "rb") as img_file:
    #    labeled_image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

    buffered = BytesIO()
    labeled_image.save(buffered, format="PNG")
    labeled_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY environment variable")

    #print(api_key)

    # Create a client instance
    client = OpenAI(api_key=api_key)

    starting_prompt = (
        "Given the parsed list of elements from a UI screenshot, suggest a specific location to click with coordinates. Return your response as valid JSON, and nothing else. Do not include explanations or extra text. Output must begin with { and end with }." +
        "\n\n" + action_description + "\n\n" +
        "Report only center of the location to click, a recommendation to (click), and a description of the step in JSON (no starting quotes or json), examples: {\"location\": {\"x\": 0.893, \"y\": 0.32}, \"action\": \"click\"} and {\"location\": {\"x\": 0.893, \"y\": 0.32}, \"action\": \"type\", \"text\": \"search query\", \"description\": \"Click the search input box to enter your query\"}"
    )

    ending_prompt = (
        "What is the JSON? Report only the final action to take, and nothing else. Output must begin with { and end with }."
    )

    response = client.chat.completions.create(
        model="gpt-5",  # latest multimodal GPT‑4 variant
        messages=[
            {"role": "system",  "content": "You are a JSON-only API that outputs only JSON as specified."},
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": f"{starting_prompt}\n\n{parsed_content_list}\n\n{ending_prompt}"
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{labeled_image_base64}"
                }
                }
            ]
            }
        ]
        #,temperature=0.7     # optional: tweak output randomness
        #,max_tokens=512,      # optional: control response length
    )

    result = response.choices[0].message.content

    print("GPT-4o response:")
    print(result)

    with open("starting_prompt_and_parsed_content.txt", "w") as f:
        f.write(f"{starting_prompt}\n\n{parsed_content_list}\n\n{ending_prompt}")

    #result = get_ollama_response([
    #         {
    #             'role': 'user',
    #             'content': f"base64:{labeled_image_base64}\n{starting_prompt}\n\n{parsed_content_list}\n\nWhere should I click?"
    #             #,'images': [labeled_image_base64]
    #        }
    #    ])

    # Parse the result as JSON
    try:
        result_json = json.loads(result)
        x = result_json["location"]["x"]
        y = result_json["location"]["y"]
    except Exception as e:
        print("Failed to parse result as JSON:", e)
        return

    # Open the labeled image
    #labeled_image = Image.open("labeled_image.png")
    width, height = labeled_image.size

    # Calculate pixel coordinates
    px = int(x * width)
    py = int(y * height)

    # Draw an X at the location
    fig, ax = plt.subplots()
    ax.imshow(labeled_image)
    ax.plot(px, py, marker='x', markersize=20, color='red', markeredgewidth=4)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig("labeled_image_with_x.png", bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Drew an X at ({px}, {py}) and saved as labeled_image_with_x.png")


def get_ollama_response(messages, model='gpt-oss:20b'):
    # Chat with image and text
    response = ollama.chat(
        model=model,
        messages=messages
    )

    result = response['message']['content']

    print(result)

    return result

def list_cameras():
    devices = AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeVideo)

    camera_num = 0
    camera_num_i = 0
    print("Available cameras:")    
    for device in devices:
        print(device.localizedName())
        if "USB" in device.localizedName():
            camera_num = camera_num_i
            print(f"Selecting camera {camera_num} ({device.localizedName()})")
        camera_num_i += 1

    return camera_num

_usb_capture = None   # persistent VideoCapture handle
_usb_camera_name = None


def _get_usb_capture():
    """Return a persistent VideoCapture for the USB capture device.

    Opens the device and warms it up on first call; subsequent calls
    reuse the same handle so each frame grab is a single read (~33 ms
    at 30 fps) instead of re-opening and discarding 10 warm-up frames.
    """
    global _usb_capture, _usb_camera_name

    if _usb_capture is not None and _usb_capture.isOpened():
        return _usb_capture

    # Find cameras using the legacy method to avoid Continuity Camera warnings
    devices = AVFoundation.AVCaptureDevice.devicesWithMediaType_(
        AVFoundation.AVMediaTypeVideo)
    camera_num = 0
    usb_camera_found = False
    _usb_camera_name = None

    for idx, device in enumerate(devices):
        device_name = device.localizedName()
        if "Continuity" in device_name:
            continue
        if "USB" in device_name and not usb_camera_found:
            camera_num = idx
            usb_camera_found = True
            _usb_camera_name = device_name
            break

    if not usb_camera_found:
        for idx, device in enumerate(devices):
            device_name = device.localizedName()
            if "Continuity" not in device_name:
                camera_num = idx
                _usb_camera_name = device_name
                break

    camera_num = 0

    cap = cv2.VideoCapture(camera_num)
    if not cap.isOpened():
        raise ValueError('Webcam not found or cannot be opened')

    # Warm up the camera (only needed once)
    for _ in range(10):
        cap.read()

    _usb_capture = cap
    return _usb_capture


def release_usb_capture():
    """Release the persistent USB capture device (call on shutdown)."""
    global _usb_capture, _screen_rect
    if _usb_capture is not None:
        _usb_capture.release()
        _usb_capture = None
    _screen_rect = None


_screen_rect = None  # (x0, y0, x1, y1) of the device's screen in the frame
_SCREEN_RECT_FILE = os.path.join(os.path.dirname(__file__), "screen_rect.json")


def _load_screen_rect(frame_shape):
    try:
        with open(_SCREEN_RECT_FILE) as f:
            return tuple(json.load(f)[f"{frame_shape[1]}x{frame_shape[0]}"])
    except (OSError, ValueError, KeyError):
        return None


def _save_screen_rect(frame_shape, rect):
    try:
        with open(_SCREEN_RECT_FILE) as f:
            saved = json.load(f)
    except (OSError, ValueError):
        saved = {}
    saved[f"{frame_shape[1]}x{frame_shape[0]}"] = list(rect)
    with open(_SCREEN_RECT_FILE, "w") as f:
        json.dump(saved, f)


def get_screen_rect(refresh=False):
    """Bounding box of the target's screen inside the capture frame.

    HDMI capture letterboxes the device's picture in black. The extent is
    taken from the brightest value each pixel reaches over several frames.
    A dark app (black background reaching the screen edges) is
    indistinguishable from the letterbox, so detection can only shrink the
    true rect; it is therefore merged with the rect saved from earlier runs
    (screen_rect.json). Cached; pass refresh=True after switching devices
    or orientation to discard the saved rect.
    """
    global _screen_rect
    if _screen_rect is None or refresh:
        cap = _get_usb_capture()
        frames = []
        for _ in range(15):
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if not frames:
            raise ValueError('Failed to capture image from webcam')
        brightest = np.max(frames, axis=0)
        ys, xs = np.where(brightest > 10)
        if xs.size == 0:
            h, w = brightest.shape
            return (0, 0, w, h)  # all dark (screen off?); don't cache
        rect = (int(xs.min()), int(ys.min()),
                int(xs.max()) + 1, int(ys.max()) + 1)
        saved = None if refresh else _load_screen_rect(brightest.shape)
        if saved is not None:
            rect = (min(rect[0], saved[0]), min(rect[1], saved[1]),
                    max(rect[2], saved[2]), max(rect[3], saved[3]))
        if rect != saved:
            _save_screen_rect(brightest.shape, rect)
        _screen_rect = rect
    return _screen_rect


def get_image_from_usb():
    cap = _get_usb_capture()

    ret, frame = cap.read()
    if not ret:
        # Device may have been disconnected — reset and retry once
        release_usb_capture()
        cap = _get_usb_capture()
        ret, frame = cap.read()
        if not ret:
            raise ValueError('Failed to capture image from webcam')

    # Convert the image from OpenCV BGR format to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # If the picture is letterboxed (dark top-left corner), crop to the
    # device's screen. The crop is computed once and reused so that
    # screenshot coordinates stay stable from frame to frame.
    if np.all(frame_rgb[0, 0] < 30):
        x0, y0, x1, y1 = get_screen_rect()
        frame_rgb = frame_rgb[y0:y1, x0:x1]

    # Convert to PIL Image
    image = Image.fromarray(frame_rgb)

    print(f"Captured {image.size[0]}x{image.size[1]} from {_usb_camera_name or 'camera 0'}")

    return image

def _find_cursor_via_flow(img_before, img_after, expected_dir, screen_bounds=None):
    """Use dense optical flow (Farneback) to detect directional cursor motion.

    The cursor creates a small region with consistent directional flow when it
    moves, while screen noise (clock updates, animations) creates random flow.

    Args:
        img_before: BGR image before cursor move
        img_after: BGR image after cursor move
        expected_dir: 'right' or 'down'
        screen_bounds: (x0, y0, x1, y1) to restrict search area

    Returns ((cx, cy, confidence), global_flow) or (None, global_flow)
    """
    gray1 = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    flow_x, flow_y = flow[:, :, 0], flow[:, :, 1]
    magnitude = np.sqrt(flow_x**2 + flow_y**2)

    if screen_bounds:
        x0, y0, x1, y1 = screen_bounds
        global_flow = float(np.mean(magnitude[y0:y1, x0:x1]))
    else:
        global_flow = float(np.mean(magnitude))

    dir_score = flow_x if expected_dir == 'right' else flow_y

    mask = dir_score > 0.5
    if screen_bounds:
        x0, y0, x1, y1 = screen_bounds
        screen_mask = np.zeros_like(mask)
        screen_mask[y0:y1, x0:x1] = True
        mask = mask & screen_mask

    if not np.any(mask):
        return None, global_flow

    mask_uint8 = mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
    contours, _ = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    weighted_score = np.where(mask, dir_score, 0)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5:
            continue
        region_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(region_mask, [cnt], -1, 255, -1)
        region_mask = region_mask > 0
        ys, xs = np.where(region_mask)
        if len(xs) == 0:
            continue
        weights = weighted_score[ys, xs]
        total_w = np.sum(weights)
        if total_w == 0:
            continue
        cx = int(np.average(xs, weights=weights))
        cy = int(np.average(ys, weights=weights))
        mean_flow = float(np.mean(weights))
        candidates.append((cx, cy, area, mean_flow))

    if not candidates:
        return None, global_flow

    candidates.sort(key=lambda c: c[3] * min(c[2], 500), reverse=True)
    best = candidates[0]
    return (best[0], best[1], best[3]), global_flow


def _measure_calibration_axis(serial_port, axis, fixed_hid, hid_values, screen_bounds):
    """Measure cursor pixel positions along one axis using optical flow."""
    measurements = []

    def avg_capture(n=3):
        frames = []
        for _ in range(n):
            cap = cv2.VideoCapture(0)
            for __ in range(5):
                cap.read()
            ret, frame = cap.read()
            cap.release()
            if ret:
                frames.append(frame.astype(np.float32))
        if not frames:
            raise ValueError("Failed to capture calibration image")
        return np.mean(frames, axis=0).astype(np.uint8)

    def move(hid_x, hid_y):
        cmd = json.dumps({"type": "move", "x": hid_x, "y": hid_y, "text": ""})
        send_serial_message(f"{cmd}\n")
        time.sleep(0.3)

    if axis == 'x':
        move(hid_values[0], fixed_hid)
    else:
        move(fixed_hid, hid_values[0])
    time.sleep(0.2)
    prev_img = avg_capture()

    confidences = []

    for hid_val in hid_values[1:]:
        if axis == 'x':
            move(hid_val, fixed_hid)
            expected_dir = 'right'
        else:
            move(fixed_hid, hid_val)
            expected_dir = 'down'

        time.sleep(0.15)
        curr_img = avg_capture()

        result, global_flow = _find_cursor_via_flow(
            prev_img, curr_img, expected_dir, screen_bounds=screen_bounds)

        if global_flow > 5.0:
            print(f"    HID {axis}={hid_val}: screen change, skipping")
            prev_img = curr_img
            continue

        if result is None:
            prev_img = curr_img
            continue

        cx, cy, confidence = result
        pixel_val = cx if axis == 'x' else cy

        is_suspicious = False
        if len(confidences) >= 3:
            med_conf = np.median(confidences)
            if confidence > med_conf * 4 or confidence < med_conf * 0.1:
                is_suspicious = True

        if not is_suspicious:
            if len(measurements) >= 2:
                if abs(pixel_val - measurements[-2][1]) < 3:
                    break
            measurements.append((hid_val, pixel_val, confidence))
            confidences.append(confidence)

        prev_img = curr_img

    return measurements


def _robust_linear_fit(hid_arr, px_arr):
    """Iteratively fit hid = slope * pixel + intercept, removing outliers."""
    mask = np.ones(len(hid_arr), dtype=bool)

    for _ in range(5):
        h = hid_arr[mask]
        p = px_arr[mask]
        if len(h) < 3:
            break

        slope, intercept = np.polyfit(p, h, 1)
        predicted = slope * px_arr + intercept
        residuals = np.abs(hid_arr - predicted)

        med_res = np.median(residuals[mask])
        mad = np.median(np.abs(residuals[mask] - med_res))
        threshold = med_res + 3 * max(mad, 5)

        new_mask = residuals < threshold
        if np.sum(mask & ~new_mask) == 0:
            break
        mask = new_mask

    h = hid_arr[mask]
    p = px_arr[mask]
    slope, intercept = np.polyfit(p, h, 1)

    predicted = slope * p + intercept
    ss_res = np.sum((h - predicted)**2)
    ss_tot = np.sum((h - np.mean(h))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return slope, intercept, r2, mask


def calibrate(image_resize_factor=2, verbose=True):
    """Run automatic HID-to-pixel calibration using optical flow.

    Moves the cursor along X and Y axes independently, uses dense optical
    flow (Farneback) to detect cursor motion direction, and fits a linear
    model with robust outlier rejection. Takes ~2-3 minutes.

    Args:
        image_resize_factor: How much the image is resized before sending
            to the AI model (e.g., 2 means images are halved). The ratio
            is multiplied by this factor so pixel coordinates in the
            resized image map correctly to HID coordinates.
        verbose: Print progress to stdout.

    Returns:
        dict with 'hid_x_ratio', 'hid_y_ratio', 'x_r2', 'y_r2' or None on failure.
    """
    global hid_x_ratio, hid_y_ratio

    serial_port = send_serial_message.__defaults__  # not needed, we use send_serial_message directly

    # Detect screen bounds
    if verbose:
        print("Calibrating HID-to-pixel mapping...")

    # Move cursor to origin and capture
    move_cmd = json.dumps({"type": "move", "x": 0, "y": 0, "text": ""})
    send_serial_message(f"{move_cmd}\n")
    time.sleep(0.5)

    cap = cv2.VideoCapture(0)
    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("ERROR: Cannot capture from camera.")
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        print("ERROR: Cannot detect screen in camera image.")
        return None

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    screen_bounds = (x0, y0, x1, y1)
    screen_w, screen_h = x1 - x0, y1 - y0
    if verbose:
        print(f"  Screen: {screen_w}x{screen_h} camera pixels")

    # Find serial port for move commands
    serial_devices = glob.glob('/dev/tty.usbserial*')
    if not serial_devices:
        print("ERROR: No serial device found.")
        return None
    serial_port = serial_devices[0]

    # Measure X axis (~14 points, step=40)
    if verbose:
        print("  Measuring X axis...")
    x_measurements = _measure_calibration_axis(
        serial_port, 'x', fixed_hid=200,
        hid_values=list(range(60, 600, 40)),
        screen_bounds=screen_bounds)
    if verbose:
        print(f"    {len(x_measurements)} points")

    # Measure Y axis (~15 points, step=80)
    if verbose:
        print("  Measuring Y axis...")
    y_measurements = _measure_calibration_axis(
        serial_port, 'y', fixed_hid=100,
        hid_values=list(range(60, 1200, 80)),
        screen_bounds=screen_bounds)
    if verbose:
        print(f"    {len(y_measurements)} points")

    if len(x_measurements) < 3 or len(y_measurements) < 3:
        print(f"ERROR: Not enough measurements (x={len(x_measurements)}, "
              f"y={len(y_measurements)}). Need at least 3 per axis.")
        return None

    # Fit linear model with outlier rejection
    x_hid = np.array([m[0] for m in x_measurements], dtype=float)
    x_px = np.array([m[1] for m in x_measurements], dtype=float)
    y_hid = np.array([m[0] for m in y_measurements], dtype=float)
    y_px = np.array([m[1] for m in y_measurements], dtype=float)

    x_slope, x_intercept, x_r2, x_mask = _robust_linear_fit(x_hid, x_px)
    y_slope, y_intercept, y_r2, y_mask = _robust_linear_fit(y_hid, y_px)

    # Apply resize factor
    final_x_ratio = x_slope * image_resize_factor
    final_y_ratio = y_slope * image_resize_factor

    # Update global ratios
    hid_x_ratio = final_x_ratio
    hid_y_ratio = final_y_ratio

    # Save to file
    _save_calibration(final_x_ratio, final_y_ratio, {
        "x_r2": x_r2,
        "y_r2": y_r2,
        "x_points_used": int(np.sum(x_mask)),
        "x_points_total": len(x_mask),
        "y_points_used": int(np.sum(y_mask)),
        "y_points_total": len(y_mask),
        "image_resize_factor": image_resize_factor,
        "screen_w": screen_w,
        "screen_h": screen_h,
    })

    if verbose:
        print(f"\n  Calibration complete:")
        print(f"    hid_x_ratio = {final_x_ratio:.4f}  (R²={x_r2:.4f}, "
              f"{np.sum(x_mask)}/{len(x_mask)} points)")
        print(f"    hid_y_ratio = {final_y_ratio:.4f}  (R²={y_r2:.4f}, "
              f"{np.sum(y_mask)}/{len(y_mask)} points)")
        print(f"    Saved to {_CALIBRATION_FILE}")

    return {
        "hid_x_ratio": final_x_ratio,
        "hid_y_ratio": final_y_ratio,
        "x_r2": x_r2,
        "y_r2": y_r2,
    }

if __name__ == "__main__":
    main()