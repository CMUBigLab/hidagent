"""
Screen Reader Anywhere

Enables non-visual access to any device by capturing screenshots, parsing UI
elements with Omniparser, and rendering an accessible web interface on the
control computer. Actions from the web interface are relayed back to the
target device via HIDAgent.

A screen reader on the control computer can read the generated HTML, allowing
blind/low-vision users to interact with devices that lack a built-in screen reader.

Usage:
    python screen_reader.py [--port 8080]
"""

from dotenv import load_dotenv
load_dotenv()

import os
import sys
# Allow running from examples/ directory by adding parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import base64
import argparse
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, jsonify, render_template_string
from hidagent.hidagent import (
    get_screenshot, click_mouse, type_text, recognize_gui_elements
)


app = Flask(__name__)
current_elements = []
current_screenshot_b64 = ""
current_image_size = (1, 1)  # (width, height) of the latest screenshot
# Cache of previous elements for stability across refreshes
_stable_elements = []  # list of element dicts from last refresh
_stable_id_counter = [0]  # mutable counter for unique IDs


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Screen Reader Anywhere - HIDAgent</title>
    <style>
        body {
            font-family: -apple-system, sans-serif;
            margin: 0;
            padding: 0;
            background: #1e1e1e;
            color: #ddd;
        }
        .container {
            display: grid;
            grid-template-columns: 1fr 460px;
            gap: 20px;
            padding: 20px;
            max-width: 1600px;
            margin: 0 auto;
        }
        h1 { font-size: 18px; margin: 0 0 10px; }
        h2 { font-size: 14px; color: #888; text-transform: uppercase;
             margin: 0 0 10px; }
        .screenshot-pane {
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
        }
        .screenshot-wrap {
            position: relative;
            display: inline-block;
            max-width: 100%;
        }
        .screenshot-wrap img {
            width: 100%;
            display: block;
            border: 1px solid #444;
            border-radius: 4px;
        }
        /* Interactive overlay elements positioned via % so they scale
           with the image. */
        .overlay {
            position: absolute;
            box-sizing: border-box;
            background: transparent;
            cursor: pointer;
            padding: 0;
            margin: 0;
            color: #fff;
            font-family: inherit;
            font-size: 11px;
            text-align: left;
            transition: background 0.1s, outline 0.1s;
        }
        .overlay.button, .overlay.icon {
            border: 2px solid rgba(102, 170, 255, 0.9);
        }
        .overlay.button:hover, .overlay.icon:hover {
            background: rgba(102, 170, 255, 0.3);
            outline: 2px solid rgba(102, 170, 255, 1);
        }
        .overlay.text_input {
            border: 2px solid rgba(255, 170, 102, 0.9);
        }
        .overlay.text {
            border: 2px dashed rgba(160, 160, 160, 0.6);
            cursor: default;
        }
        .overlay:focus {
            outline: 3px solid #fff;
            outline-offset: 2px;
        }
        .overlay.typing-active {
            background: rgba(255, 255, 0, 0.25) !important;
            outline: 3px solid yellow !important;
            outline-offset: 2px;
        }
        .overlay.typing-active .overlay-badge {
            background: yellow !important;
            color: black !important;
        }
        #typing-banner {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background: #ffeb3b;
            color: #000;
            padding: 10px 20px;
            font-size: 14px;
            z-index: 100;
            align-items: center;
            gap: 10px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        }
        #typing-banner a { color: #004; }

        /* Refresh indicator — shows countdown bar and status */
        #refresh-indicator {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: transparent;
            z-index: 101;
            pointer-events: none;
        }
        #refresh-bar {
            height: 100%;
            width: 0;
            background: linear-gradient(90deg, #6af, #ac6);
            transition: width linear;
        }
        #refresh-status {
            position: fixed;
            top: 12px;
            right: 12px;
            background: #333;
            color: #ddd;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            z-index: 102;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            border: 1px solid #555;
        }
        #refresh-status.visible { opacity: 1; }
        #refresh-status.refreshing {
            background: #6af;
            color: #000;
            font-weight: 600;
        }
        .overlay-badge {
            position: absolute;
            top: -18px;
            left: -2px;
            background: rgba(102, 170, 255, 0.95);
            color: #000;
            font-weight: 700;
            font-size: 10px;
            padding: 1px 5px;
            border-radius: 2px;
            white-space: nowrap;
            pointer-events: none;
        }
        .overlay.text_input .overlay-badge { background: rgba(255, 170, 102, 0.95); }
        .overlay.text .overlay-badge { background: rgba(160, 160, 160, 0.95); }
        .overlay input {
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            color: #fff;
            border: none;
            padding: 2px 4px;
            font-size: 12px;
            box-sizing: border-box;
        }

        /* List pane */
        .elements-pane {
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            max-height: calc(100vh - 100px);
            overflow-y: auto;
            position: sticky;
            top: 20px;
            align-self: start;
        }
        .element {
            margin: 4px 0;
            padding: 6px 10px;
            background: #333;
            border-left: 3px solid #555;
            border-radius: 4px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .element.button, .element.icon { border-left-color: #6af; }
        .element.text_input { border-left-color: #fa6; }
        .element.text { border-left-color: #888; }
        .element-num {
            color: #888;
            font-family: monospace;
            font-size: 11px;
            min-width: 24px;
        }
        .element button, .element input {
            background: #444;
            color: #ddd;
            border: 1px solid #666;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 13px;
            cursor: pointer;
            flex: 1;
            text-align: left;
        }
        .element button:hover { background: #555; }
        .element input { cursor: text; min-width: 150px; }
        .element-text { color: #ccc; font-size: 13px; flex: 1; }
        .header {
            padding: 15px 20px;
            border-bottom: 1px solid #444;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
    </style>
</head>
<body>
    <div id="typing-banner"></div>
    <div id="refresh-indicator"><div id="refresh-bar"></div></div>
    <div id="refresh-status">Refresh in 3s</div>

    <div class="header">
        <h1>Screen Reader Anywhere</h1>
        <div style="font-size: 12px; color: #aaa;">
            {{ num_elements }} elements ·
            <span style="color: #6af;">click / Enter</span> = tap ·
            <span style="color: #fc0;">shift+click / shift+Enter</span> = tap and type ·
            <kbd>Esc</kbd> exit typing ·
            <kbd>R</kbd> refresh now
        </div>
    </div>

    <div class="container">
        <div class="screenshot-pane">
            <h2>Screen — click overlays to interact</h2>
            <div class="screenshot-wrap">
                <img src="/screenshot?t={{ timestamp }}" alt="Target device screen">
                {% for elem in elements %}
                {% set bx1, by1, bx2, by2 = elem.bbox %}
                {% set w_pct = (bx2 - bx1) / image_w * 100 %}
                {% set h_pct = (by2 - by1) / image_h * 100 %}
                {% set l_pct = bx1 / image_w * 100 %}
                {% set t_pct = by1 / image_h * 100 %}

                {% if elem.type == 'text_input' %}
                <div class="overlay {{ elem.type }}"
                     style="left: {{ l_pct }}%; top: {{ t_pct }}%;
                            width: {{ w_pct }}%; height: {{ h_pct }}%;"
                     role="group" aria-label="{{ elem.label }}">
                    <span class="overlay-badge">{{ loop.index }}. {{ elem.label }}</span>
                    <input type="text" placeholder="{{ elem.label }}"
                           aria-label="{{ elem.label }}"
                           data-device-x="{{ elem.x }}" data-device-y="{{ elem.y }}"
                           onfocus="focusInput(this)"
                           onkeydown="handleInputKey(event, this)">
                </div>
                {% elif elem.type == 'text' %}
                <div class="overlay {{ elem.type }}"
                     style="left: {{ l_pct }}%; top: {{ t_pct }}%;
                            width: {{ w_pct }}%; height: {{ h_pct }}%;"
                     role="text" aria-label="{{ elem.label }}"
                     tabindex="0">
                    <span class="overlay-badge">{{ loop.index }}. {{ elem.label }}</span>
                </div>
                {% else %}
                <button class="overlay {{ elem.type }}"
                        id="overlay-{{ elem.id }}"
                        style="left: {{ l_pct }}%; top: {{ t_pct }}%;
                               width: {{ w_pct }}%; height: {{ h_pct }}%;"
                        aria-label="{{ elem.label }}"
                        data-device-x="{{ elem.x }}" data-device-y="{{ elem.y }}"
                        onclick="handleOverlayClick(event, this, {{ elem.x }}, {{ elem.y }})"
                        onkeydown="handleOverlayKey(event, this, {{ elem.x }}, {{ elem.y }})">
                    <span class="overlay-badge">{{ loop.index }}. {{ elem.label }}</span>
                </button>
                {% endif %}
                {% endfor %}
            </div>
        </div>

        <div class="elements-pane">
            <h2>Element list (reading order)</h2>
            {% for elem in elements %}
            <div id="elem-{{ elem.id }}" class="element {{ elem.type }}"
                 role="{{ elem.role }}">
                <span class="element-num">{{ loop.index }}</span>
                {% if elem.type in ('button', 'icon') %}
                <button id="btn-{{ elem.id }}"
                        onclick="handleOverlayClick(event, this, {{ elem.x }}, {{ elem.y }})"
                        onkeydown="handleOverlayKey(event, this, {{ elem.x }}, {{ elem.y }})"
                        data-device-x="{{ elem.x }}" data-device-y="{{ elem.y }}"
                        aria-label="{{ elem.label }}">
                    {{ elem.label }}
                </button>
                {% elif elem.type == 'text_input' %}
                <input id="input-{{ elem.id }}" type="text"
                       placeholder="{{ elem.label }}"
                       data-device-x="{{ elem.x }}" data-device-y="{{ elem.y }}"
                       onfocus="focusInput(this)"
                       onkeydown="handleInputKey(event, this)"
                       aria-label="{{ elem.label }}">
                {% else %}
                <span class="element-text" aria-label="{{ elem.label }}">
                    {{ elem.label }}
                </span>
                {% endif %}
            </div>
            {% endfor %}
        </div>
    </div>

    <script>
    // After clicking any element, enter "typing mode" where keystrokes
    // are forwarded to the device. Omniparser can't reliably distinguish
    // text input fields from other clickable elements, so we let any
    // click start typing mode.
    let typingModeEl = null;

    // Refresh state management
    const REFRESH_DELAY_MS = 3000;
    let pendingRefreshTimer = null;
    let refreshBarAnimation = null;
    const refreshBar = document.getElementById('refresh-bar');
    const refreshStatus = document.getElementById('refresh-status');

    function cancelPendingRefresh() {
        if (pendingRefreshTimer !== null) {
            clearTimeout(pendingRefreshTimer);
            pendingRefreshTimer = null;
        }
        if (refreshBarAnimation) {
            refreshBarAnimation.cancel();
            refreshBarAnimation = null;
        }
        refreshBar.style.width = '0';
        refreshStatus.classList.remove('visible');
    }

    function schedulePendingRefresh(ms) {
        cancelPendingRefresh();
        const delay = ms || REFRESH_DELAY_MS;

        // Show the status pill and animate the progress bar
        refreshStatus.textContent = `Refresh in ${Math.round(delay / 1000)}s`;
        refreshStatus.classList.add('visible');
        refreshStatus.classList.remove('refreshing');

        refreshBar.style.width = '0';
        // Force layout so the transition actually runs
        refreshBar.offsetWidth;
        refreshBar.style.transitionDuration = delay + 'ms';
        refreshBar.style.width = '100%';

        // Update the countdown text
        let remaining = Math.round(delay / 1000);
        const countdownInterval = setInterval(() => {
            remaining--;
            if (remaining > 0) {
                refreshStatus.textContent = `Refresh in ${remaining}s`;
            }
        }, 1000);

        pendingRefreshTimer = setTimeout(() => {
            clearInterval(countdownInterval);
            pendingRefreshTimer = null;
            refreshStatus.textContent = 'Refreshing...';
            refreshStatus.classList.add('refreshing');
            location.reload();
        }, delay);
    }

    function refreshNow() {
        cancelPendingRefresh();
        refreshStatus.textContent = 'Refreshing...';
        refreshStatus.classList.add('visible', 'refreshing');
        refreshBar.style.width = '0';
        location.reload();
    }

    function doAction(action, x, y, text, opts) {
        const refresh = !(opts && opts.noRefresh);
        return fetch('/action', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({action: action, x: x, y: y, text: text || ''})
        }).then(() => {
            if (refresh) {
                schedulePendingRefresh();
            }
        });
    }

    // Flash the indicator briefly on page load to signal "refreshed"
    window.addEventListener('DOMContentLoaded', () => {
        refreshStatus.textContent = 'Refreshed';
        refreshStatus.classList.add('visible');
        refreshBar.style.transitionDuration = '0ms';
        refreshBar.style.width = '100%';
        refreshBar.offsetWidth;
        refreshBar.style.transitionDuration = '600ms';
        refreshBar.style.width = '0';
        setTimeout(() => {
            refreshStatus.classList.remove('visible');
        }, 1500);
    });

    // Keyboard shortcut: 'r' or F5 to refresh immediately
    document.addEventListener('keydown', (e) => {
        // Don't interfere with text input fields or typing mode
        const active = document.activeElement;
        if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA')) {
            return;
        }
        if (typingModeEl) return;

        if (e.key === 'r' || e.key === 'R') {
            if (!e.ctrlKey && !e.metaKey && !e.altKey) {
                e.preventDefault();
                refreshNow();
            }
        }
    });

    function handleOverlayClick(event, el, x, y) {
        event.preventDefault();
        if (event.shiftKey) {
            // Shift+click → click and enter typing mode (for text fields)
            clickAndType(el, x, y);
        } else {
            // Regular click → send click and refresh
            doAction('click', x, y);
        }
    }

    function handleOverlayKey(event, el, x, y) {
        // Shift+Enter (or Shift+Space) on a focused overlay = enter typing mode
        // Enter/Space alone activates the click via the browser's default behavior
        if ((event.key === 'Enter' || event.key === ' ') && event.shiftKey) {
            event.preventDefault();
            event.stopPropagation();
            clickAndType(el, x, y);
        }
    }

    function clickAndType(el, x, y) {
        // Cancel any pending refresh so typing mode isn't interrupted
        cancelPendingRefresh();

        // Send the click to the device
        fetch('/action', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({action: 'click', x: x, y: y})
        });

        // Enter typing mode on this element
        if (typingModeEl) typingModeEl.classList.remove('typing-active');
        typingModeEl = el;
        el.classList.add('typing-active');
        // Keep focus on this element so keyboard input works immediately
        try { el.focus(); } catch (e) {}
        updateTypingBanner();
    }

    function exitTypingMode() {
        if (typingModeEl) typingModeEl.classList.remove('typing-active');
        typingModeEl = null;
        updateTypingBanner();
    }

    function updateTypingBanner() {
        const banner = document.getElementById('typing-banner');
        if (!banner) return;
        if (typingModeEl) {
            const label = typingModeEl.getAttribute('aria-label') || '';
            banner.innerHTML = '⌨️ Typing mode on "<b>' + escapeHtml(label) +
                '</b>" — keystrokes sent to device. Press Escape to stop, or ' +
                '<a href="#" onclick="event.preventDefault(); location.reload();">refresh</a>.';
            banner.style.display = 'flex';
        } else {
            banner.style.display = 'none';
        }
    }

    function escapeHtml(s) {
        const d = document.createElement('div');
        d.textContent = s || '';
        return d.innerHTML;
    }

    // Global keydown — forward keystrokes to the device when typing mode is active.
    // Don't intercept when a native HTML form element has focus (e.g. the manual
    // input fields in the list pane) since those have their own handlers.
    document.addEventListener('keydown', (e) => {
        if (!typingModeEl) return;

        const active = document.activeElement;
        if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA')) {
            return;
        }

        const key = e.key;
        const specialKeys = {
            'Enter': 'enter',
            'Tab': 'tab',
            'Backspace': 'backspace',
            'Delete': 'delete',
            'ArrowUp': 'up',
            'ArrowDown': 'down',
            'ArrowLeft': 'left',
            'ArrowRight': 'right',
        };

        if (key === 'Escape') {
            e.preventDefault();
            exitTypingMode();
            return;
        }

        if (specialKeys[key]) {
            e.preventDefault();
            fetch('/action', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action: 'key', text: specialKeys[key]})
            });
            return;
        }

        if (key.length === 1 && !e.ctrlKey && !e.metaKey && !e.altKey) {
            e.preventDefault();
            fetch('/action', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action: 'type', text: key})
            });
        }
    });
    </script>
</body>
</html>
"""


def _bbox_pixels(item, image_size):
    """Return (x1, y1, x2, y2, cx, cy) in pixel coordinates."""
    width, height = image_size
    if "bbox" in item:
        bbox = item["bbox"]
        x1 = int(bbox[0] * width)
        y1 = int(bbox[1] * height)
        x2 = int(bbox[2] * width)
        y2 = int(bbox[3] * height)
    elif "location" in item:
        cx = int(item["location"]["x"] * width)
        cy = int(item["location"]["y"] * height)
        x1, y1 = cx - 25, cy - 25
        x2, y2 = cx + 25, cy + 25
    else:
        return None
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return (x1, y1, x2, y2, cx, cy)


def reading_order_sort(parsed_content_list, image_size, row_tolerance=0.05):
    """Sort elements in reading order (top-to-bottom, left-to-right).

    Elements whose vertical centers are within `row_tolerance * height` of
    each other are considered to be on the same row and sorted by X.
    """
    _, height = image_size
    row_px = height * row_tolerance

    items = []
    for item in parsed_content_list:
        box = _bbox_pixels(item, image_size)
        if box is None:
            continue
        items.append((item, box))

    if not items:
        return []

    # Group into rows by Y
    items.sort(key=lambda it: it[1][5])  # sort by cy first
    rows = []
    for it in items:
        cy = it[1][5]
        if rows and abs(cy - rows[-1][-1][1][5]) < row_px:
            rows[-1].append(it)
        else:
            rows.append([it])

    # Sort each row by X
    ordered = []
    for row in rows:
        row.sort(key=lambda it: it[1][4])
        ordered.extend(row)

    return ordered


def _bbox_iou(b1, b2):
    """Intersection over union for two [x1, y1, x2, y2] boxes."""
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection = iw * ih
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _label_similarity(a, b):
    """Simple label similarity: normalized overlap of lowercase characters."""
    a = (a or "").lower().strip()
    b = (b or "").lower().strip()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # Jaccard on character bigrams
    bg_a = {a[i:i + 2] for i in range(len(a) - 1)} or {a}
    bg_b = {b[i:i + 2] for i in range(len(b) - 1)} or {b}
    inter = len(bg_a & bg_b)
    union = len(bg_a | bg_b)
    return inter / union if union else 0.0


def stabilize_elements(new_elements):
    """Match new elements against _stable_elements and preserve identity.

    For each new element, find the best match in the cached list:
      - IoU >= 0.4 AND label similarity >= 0.4 → match
    Matched elements keep the cached element's `id` and `label`.
    Unmatched new elements get a fresh id.
    """
    global _stable_elements

    # Track which cached elements have been matched
    matched_cache = set()
    result = []

    for new_elem in new_elements:
        best_match_idx = None
        best_score = 0.0
        for i, old_elem in enumerate(_stable_elements):
            if i in matched_cache:
                continue
            iou = _bbox_iou(new_elem["bbox"], old_elem["bbox"])
            if iou < 0.4:
                continue
            label_sim = _label_similarity(
                new_elem["label"], old_elem["label"])
            if label_sim < 0.4:
                continue
            score = iou * 0.6 + label_sim * 0.4
            if score > best_score:
                best_score = score
                best_match_idx = i

        if best_match_idx is not None:
            old_elem = _stable_elements[best_match_idx]
            matched_cache.add(best_match_idx)
            # Preserve stable ID and label from the cache.
            # Use the new position (in case element moved slightly).
            stable = dict(new_elem)
            stable["id"] = old_elem["id"]
            stable["label"] = old_elem["label"]
            result.append(stable)
        else:
            # New element not seen before — assign a fresh id
            _stable_id_counter[0] += 1
            stable = dict(new_elem)
            stable["id"] = _stable_id_counter[0]
            result.append(stable)

    _stable_elements = result
    return result


def parse_elements_for_html(parsed_content_list, image_size):
    """Convert parsed UI elements to accessible HTML descriptors,
    sorted in reading order and stabilized across refreshes."""
    ordered = reading_order_sort(parsed_content_list, image_size)
    elements = []

    for item, box in ordered:
        x1, y1, x2, y2, cx, cy = box
        label = item.get("content") or item.get("caption") or "element"
        elem_type = item.get("type", "button")

        role = ("button" if elem_type in ("button", "icon")
                else "textbox" if elem_type == "text_input"
                else "text")

        elements.append({
            "type": elem_type,
            "label": str(label)[:100],
            "x": cx,
            "y": cy,
            "bbox": [x1, y1, x2, y2],
            "role": role,
        })

    # Preserve labels and IDs across refreshes so screen readers don't
    # flicker on minor OCR variations.
    return stabilize_elements(elements)


def make_annotated_image(img, elements):
    """Draw numbered bounding boxes on the screenshot in reading order."""
    annotated = img.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Helvetica.ttc", 18)
        small_font = ImageFont.truetype(
            "/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    colors = {
        "button": (102, 170, 255),
        "icon": (102, 170, 255),
        "text_input": (255, 170, 102),
        "text": (160, 160, 160),
    }

    for i, elem in enumerate(elements, start=1):
        x1, y1, x2, y2 = elem["bbox"]
        color = colors.get(elem["type"], (200, 200, 200))
        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        # Draw number badge
        badge_w = 30 if i < 100 else 40
        draw.rectangle([x1, y1 - 22, x1 + badge_w, y1], fill=color)
        draw.text((x1 + 4, y1 - 22), str(i), fill="black", font=font)

    return annotated


def refresh_screen():
    """Capture a new screenshot and parse it for UI elements."""
    global current_elements, current_screenshot_b64, current_image_size

    img = get_screenshot()
    current_image_size = img.size
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    current_screenshot_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    parsed_content = recognize_gui_elements(img)
    current_elements = parse_elements_for_html(parsed_content, img.size)


@app.route("/")
def index():
    try:
        refresh_screen()
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            f"<h1>Error refreshing screen</h1><pre>{traceback.format_exc()}</pre>",
            500)
    try:
        return render_template_string(
            HTML_TEMPLATE,
            elements=current_elements,
            num_elements=len(current_elements),
            image_w=current_image_size[0],
            image_h=current_image_size[1],
            timestamp=int(time.time()),
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            f"<h1>Error rendering page</h1><pre>{traceback.format_exc()}</pre>"
            f"<p>Elements: {len(current_elements)}</p>"
            f"<p>Image size: {current_image_size}</p>",
            500)


@app.route("/screenshot")
def screenshot():
    if not current_screenshot_b64:
        refresh_screen()
    img_bytes = base64.b64decode(current_screenshot_b64)
    return img_bytes, 200, {"Content-Type": "image/png"}




@app.route("/action", methods=["POST"])
def action():
    from hidagent.hidagent import keypress
    data = request.get_json()
    action_type = data.get("action")
    x = data.get("x", 0)
    y = data.get("y", 0)
    text = data.get("text", "")

    # Element coordinates are in screenshot pixels, as click_mouse expects.
    if action_type == "click":
        click_mouse(x, y)

    elif action_type == "type_at":
        click_mouse(x, y)
        time.sleep(0.5)
        type_text(text)

    elif action_type == "type":
        # Live typing — send text directly without clicking
        type_text(text)

    elif action_type == "key":
        # Special key press (enter, tab, backspace, arrows, etc.)
        keypress([text])

    return jsonify({"status": "ok"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Screen Reader Anywhere via HIDAgent")
    parser.add_argument("--port", type=int, default=8080, help="Web server port")
    args = parser.parse_args()

    print(f"Starting Screen Reader Anywhere on http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)
