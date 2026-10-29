"""
Extensible UI Agent (Claude variant)

Uses Claude in Computer Use mode to control any device connected via HIDAgent.
Claude receives screenshots, reasons about the UI, and issues click/type/key
actions that are forwarded to the target device over serial.

Adds a custom `runcommand` tool for launching applications via spotlight/search.

Usage:
    python claude_agent.py "Open Safari and search for the weather"
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
from io import BytesIO
import numpy as np
from anthropic import Anthropic
from hidagent.hidagent import (
    get_screenshot, click_mouse, move_mouse, type_text, keypress,
    run_application, drag_up, drag_down
)


# Claude sees the full-resolution screenshot, and click_mouse() & co. take
# screenshot pixels, so Claude's coordinates are used as they are.

# Counter for debug images
_click_counter = [0]


def _save_click_debug(x, y, label="click", before_img=None, after_img=None):
    """Save a debug image showing the click target AND where the cursor
    actually landed (detected by diffing before/after images).

    This lets you distinguish between Claude giving wrong coordinates vs
    the HID conversion producing wrong cursor positions.
    """
    import cv2
    import numpy as np

    _click_counter[0] += 1
    n = _click_counter[0]

    # Use after_img as the canvas if we have it, else take fresh
    if after_img is None:
        after_img = get_screenshot()

    arr = np.array(after_img)
    if arr.shape[-1] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    x, y = int(x), int(y)

    # Draw Claude's intended click location (green)
    cv2.circle(arr, (x, y), 20, (0, 255, 0), 2)
    cv2.line(arr, (x - 30, y), (x + 30, y), (0, 255, 0), 2)
    cv2.line(arr, (x, y - 30), (x, y + 30), (0, 255, 0), 2)
    cv2.putText(arr, f"Claude target: ({x},{y})", (x + 25, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Detect where the cursor actually landed by diffing before/after
    cursor_x, cursor_y = None, None
    if before_img is not None:
        before_arr = np.array(before_img)
        after_arr = np.array(after_img)
        # Crop both to matching dimensions (border cropping can vary by 1-2 px)
        min_h = min(before_arr.shape[0], after_arr.shape[0])
        min_w = min(before_arr.shape[1], after_arr.shape[1])
        before_arr = before_arr[:min_h, :min_w]
        after_arr = after_arr[:min_h, :min_w]
        gray_before = cv2.cvtColor(before_arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray_after = cv2.cvtColor(after_arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
        diff = np.abs(gray_after - gray_before).astype(np.uint8)
        _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, centroids = \
            cv2.connectedComponentsWithStats(thresh, connectivity=8)
        candidates = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if 30 < area < 3000:
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                aspect = max(w, h) / (min(w, h) + 1)
                if aspect < 3:
                    candidates.append((centroids[i][0], centroids[i][1], area))
        if candidates:
            # Pick the largest cursor-like blob
            best = max(candidates, key=lambda c: c[2])
            cursor_x, cursor_y = int(best[0]), int(best[1])

    # Draw actual cursor location (red) if detected
    if cursor_x is not None:
        cv2.circle(arr, (cursor_x, cursor_y), 20, (0, 0, 255), 2)
        cv2.line(arr, (cursor_x - 30, cursor_y),
                 (cursor_x + 30, cursor_y), (0, 0, 255), 2)
        cv2.line(arr, (cursor_x, cursor_y - 30),
                 (cursor_x, cursor_y + 30), (0, 0, 255), 2)
        cv2.putText(arr, f"Cursor landed: ({cursor_x},{cursor_y})",
                    (cursor_x + 25, cursor_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Compute and draw offset vector
        dx = cursor_x - x
        dy = cursor_y - y
        cv2.arrowedLine(arr, (x, y), (cursor_x, cursor_y),
                        (0, 165, 255), 2, tipLength=0.2)
        cv2.putText(arr, f"offset: ({dx:+d},{dy:+d})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        # Print correction suggestion
        print(f"  [debug] Claude target: ({x},{y}), cursor landed: "
              f"({cursor_x},{cursor_y}), offset: ({dx:+d},{dy:+d})")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imgs")
    os.makedirs(out_dir, exist_ok=True)
    safe_label = ''.join(c if c.isalnum() else '_' for c in label)[:20]
    out_path = os.path.join(out_dir, f"click_{n:02d}_{safe_label}.png")
    cv2.imwrite(out_path, arr)
    print(f"  [debug] Saved click target to imgs/click_{n:02d}_{safe_label}.png")


def capture_screenshot_b64():
    """Capture screenshot from target device at full resolution for Claude."""
    img = get_screenshot()
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8"), img.size


def probe_screenshot_size():
    """Take a screenshot to determine the actual image dimensions."""
    img = get_screenshot()
    return img.size  # (width, height)


def run_tool(tool_name, tool_input):
    """Execute a tool action on the target device via HIDAgent."""

    if tool_name == "computer" and tool_input.get("action") == "screenshot":
        img_b64, size = capture_screenshot_b64()
        return [{"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": img_b64
        }}]

    elif tool_name == "computer" and tool_input.get("action") == "left_click":
        x, y = tool_input["coordinate"]
        _save_click_debug(x, y, "left_click")
        click_mouse(x, y)
        return [{"type": "text", "text": f"Clicked at ({x}, {y})"}]

    elif tool_name == "computer" and tool_input.get("action") == "key":
        keyname = tool_input["text"]
        keypress(keyname.split("+"))
        return [{"type": "text", "text": f"Key pressed: {keyname}"}]

    elif tool_name == "computer" and tool_input.get("action") == "type":
        text = tool_input["text"]
        type_text(text)
        return [{"type": "text", "text": f"Typed: {text}"}]

    elif tool_name == "computer" and tool_input.get("action") == "mouse_move":
        x, y = tool_input["coordinate"]
        move_mouse(x, y)
        return [{"type": "text", "text": f"Moved to ({x}, {y})"}]

    elif tool_name == "computer" and tool_input.get("action") == "wait":
        time.sleep(tool_input.get("duration", 1))
        return [{"type": "text", "text": "Waited"}]

    elif tool_name == "computer" and tool_input.get("action") == "scroll":
        # Claude's scroll action: coordinate is where to scroll, direction
        # is "up"/"down"/"left"/"right", scroll_amount is in "clicks".
        # On a touchscreen, scrolling DOWN (to see content below) means
        # dragging UP; scrolling UP means dragging DOWN.
        coord = tool_input.get("coordinate", [200, 400])
        direction = tool_input.get("scroll_direction", "down")
        x, y = coord
        if direction == "down":
            drag_up(x, y)
        elif direction == "up":
            drag_down(x, y)
        else:
            return [{"type": "text",
                     "text": f"Scroll direction {direction} not supported "
                             f"on touchscreen"}]
        return [{"type": "text",
                 "text": f"Scrolled {direction} at ({x}, {y})"}]

    elif tool_name == "computer" and tool_input.get("action") == "zoom":
        # Zoom into a specific region of the most recent screenshot.
        # Claude provides region = [x1, y1, x2, y2] in display coordinates.
        region = tool_input.get("region", [0, 0, 0, 0])
        img = get_screenshot()
        x1, y1, x2, y2 = region
        # Clamp to image bounds
        x1 = max(0, min(img.width, int(x1)))
        y1 = max(0, min(img.height, int(y1)))
        x2 = max(0, min(img.width, int(x2)))
        y2 = max(0, min(img.height, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return [{"type": "text", "text": "Error: Invalid zoom region"}]
        cropped = img.crop((x1, y1, x2, y2))
        buffered = BytesIO()
        cropped.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return [{"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": img_b64
        }}]

    elif tool_name == "runcommand":
        app = tool_input["command"]
        run_application(app)
        return [{"type": "text", "text": f"Launched: {app}"}]

    return [{"type": "text", "text": f"Unknown tool: {tool_name}"}]


def sampling_loop(task, model="claude-sonnet-4-6", use_thinking=False,
                   messages=None):
    """Main agent loop: send task to Claude, execute tools, repeat.

    Args:
        model: "claude-sonnet-4-6" (faster, default) or "claude-opus-4-6"
            (more capable but slower). Both support the new computer use
            tool with zoom.
        use_thinking: Enable extended thinking. Improves reasoning but adds
            significant latency per request (especially on Opus).
        messages: Optional existing conversation history to continue from.
            If provided, `task` is appended as a new user message to this
            history, preserving prior context across calls.
    """
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Probe the actual screenshot size so Claude's coordinate system
    # matches the real image dimensions.
    display_w, display_h = probe_screenshot_size()
    print(f"Display dimensions: {display_w}x{display_h}")

    # Use the newer computer use tool version which supports zoom —
    # this lets Claude zoom into screen regions to see icons in detail,
    # improving click accuracy on small mobile UI elements.
    tools = [
        {
            "type": "computer_20251124",
            "name": "computer",
            "display_width_px": display_w,
            "display_height_px": display_h,
            "enable_zoom": True,
        },
        {
            "name": "runcommand",
            "description": "Launch an application on the target device "
                           "(e.g., safari, notes, settings, chrome)",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Name of the application to launch",
                    }
                },
                "required": ["command"],
            },
        },
    ]

    system_prompt = (
        "You are controlling an iPhone via a hardware bridge. "
        "When clicking on app icons, buttons, or UI elements, always click "
        "at the GEOMETRIC CENTER of the target element, not at a label, "
        "corner, or edge. Icons are typically ~60-80 pixels wide and you "
        "should aim for the middle of the icon graphic (not the label text "
        "below it). "
        "\n\nIMPORTANT: Before clicking small targets like app icons, use "
        "the 'zoom' action to inspect the area at full resolution. This gives "
        "you much more accurate coordinates for the click. Example: "
        "if an icon looks like it's around (100, 250), zoom into region "
        "[60, 210, 140, 290] to see it clearly before clicking."
        "\n\nTo scroll, use the 'scroll' action with a coordinate in the "
        "middle of the scrollable area. This is a touchscreen, so scrolling "
        "is implemented as a swipe gesture on the device (scroll down = "
        "swipe up, scroll up = swipe down). Only 'up' and 'down' directions "
        "are supported."
    )

    if messages is None:
        messages = [{"role": "user", "content": task}]
    else:
        # Continuing an existing conversation — a new device may now be
        # connected.  Capture a fresh screenshot so Claude can see the
        # current screen and explicitly tell it the device has changed.
        img_b64, _ = capture_screenshot_b64()
        messages.append({"role": "user", "content": [
            {"type": "text", "text": (
                "IMPORTANT: A new device is now connected and you are "
                "controlling it. The screenshot below shows the current "
                "screen of this new device. Treat this as a different "
                "device from before — the OS, screen layout, and "
                "available apps may all be different.\n\n" + task
            )},
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": img_b64,
            }},
        ]})

    iteration = 0
    while True:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")
        print(f"  Calling {model}... (this may take 15-60s)")
        import sys as _sys
        _sys.stdout.flush()

        t_start = time.time()
        kwargs = {
            "model": model,
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": messages,
            "tools": tools,
            "betas": ["computer-use-2025-11-24"],
        }
        if use_thinking:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": 1024}

        response = client.beta.messages.create(**kwargs)
        print(f"  ({time.time() - t_start:.1f}s)")

        messages.append({"role": "assistant", "content": response.content})

        # Print Claude's text responses so you can see what it's thinking
        for block in response.content:
            if block.type == "text" and block.text.strip():
                msg = block.text.strip()
                print(f"\nClaude: {msg}")
                try:
                    from hidagent.logger import log_message
                    log_message(msg, source="Claude")
                except Exception:
                    pass
            elif block.type == "thinking" and hasattr(block, "thinking"):
                # Only show first line of thinking to avoid clutter
                first_line = block.thinking.strip().split("\n")[0][:200]
                print(f"  (thinking: {first_line}...)")

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  → {block.name}({block.input})")
                result = run_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        if not tool_results:
            print("\nAgent completed task.")
            return messages

        messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Claude agent for HIDAgent")
    parser.add_argument("task", nargs="*", help="Task description")
    parser.add_argument("--model", default="claude-sonnet-4-6",
                        help="claude-sonnet-4-6 (faster) or claude-opus-4-6")
    parser.add_argument("--thinking", action="store_true",
                        help="Enable extended thinking (slower but better reasoning)")
    parser.add_argument("--debug-log", action="store_true",
                        help="Start web-based debug log on http://localhost:8080")
    parser.add_argument("--debug-port", type=int, default=8080)
    args = parser.parse_args()

    if args.debug_log:
        from hidagent.logger import start_logging_server
        start_logging_server(port=args.debug_port)

    task = " ".join(args.task) if args.task else \
        "Take a screenshot and describe what you see."
    sampling_loop(task, model=args.model, use_thinking=args.thinking)
