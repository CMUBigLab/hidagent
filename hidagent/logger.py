"""
HIDAgent Debug Logger — Web-based visual logging interface.

Provides a local web server that displays a history of screenshots and
commands sent to the target device. Also supports sending commands
interactively by clicking on the displayed screenshot.

Usage:
    from hidagent.hidagent import click_mouse, type_text
    from hidagent.logger import start_logging_server, log_action

    start_logging_server(port=8080)

    # Automatic logging via monkey-patched hidagent functions:
    click_mouse(100, 200)  # Automatically logged

    # Or manual logging:
    log_action("click", "Clicked the Safari button", x=100, y=200)

Then open http://localhost:8080 in a browser.
"""

import os
import io
import json
import time
import base64
import threading
import http.server
import socketserver
from datetime import datetime
from urllib.parse import urlparse, parse_qs

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None


# Shared state
_log_entries = []  # List of log entries
_log_lock = threading.Lock()
_server_thread = None
_server = None
_pending_commands = []  # Commands waiting to be sent to device
_pending_lock = threading.Lock()


def _now_ts():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def log_action(action_type, details="", x=None, y=None, metadata=None,
               coord_space="screen"):
    """Log an action with the current screenshot.

    Args:
        action_type: Short string like "click", "type", "key", "screenshot".
        details: Human-readable description.
        x, y: Coordinates for click/move actions.
        metadata: Optional dict of additional info to display.
        coord_space: One of:
            - "screen" (default): (x, y) are in full screenshot pixel space
            - "calibration": (x, y) are in the half-size calibration space
              (will be scaled up by 2 to match the screenshot)
    """
    from hidagent.hidagent import get_screenshot

    try:
        img = get_screenshot()
    except Exception as e:
        img = None
        details = f"{details} [screenshot failed: {e}]"

    # Store the draw coordinates (scaled if needed)
    draw_x, draw_y = x, y
    if x is not None and y is not None and coord_space == "calibration":
        draw_x = x * 2
        draw_y = y * 2

    # Draw action mark on the screenshot
    if img is not None and Image is not None:
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)
        if draw_x is not None and draw_y is not None:
            r = 25
            draw.ellipse([draw_x - r, draw_y - r, draw_x + r, draw_y + r],
                         outline="red", width=4)
            draw.line([draw_x - r - 15, draw_y, draw_x + r + 15, draw_y],
                      fill="red", width=4)
            draw.line([draw_x, draw_y - r - 15, draw_x, draw_y + r + 15],
                      fill="red", width=4)
        buf = io.BytesIO()
        img_copy.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    else:
        img_b64 = ""

    entry = {
        "id": len(_log_entries),
        "timestamp": _now_ts(),
        "action": action_type,
        "details": details,
        "x": draw_x,
        "y": draw_y,
        "img": img_b64,
        "metadata": metadata or {},
    }

    with _log_lock:
        _log_entries.append(entry)


def log_message(text, source="agent"):
    """Log a text message (e.g., from an LLM agent) to the debug console.

    The message is displayed prominently in a large banner at the top of
    the detail panel, and also added to the thumbnail history as a
    text-only entry.

    Args:
        text: The message text to display.
        source: Short label shown on the banner (e.g., "Claude", "agent").
    """
    entry = {
        "id": len(_log_entries),
        "timestamp": _now_ts(),
        "action": "message",
        "details": text,
        "source": source,
        "x": None,
        "y": None,
        "img": "",
        "metadata": {},
    }
    with _log_lock:
        _log_entries.append(entry)


def get_pending_commands():
    """Returns (and clears) pending commands from the web interface.

    The sampling loops in agents should call this periodically to pick up
    commands typed by the user through the web UI.
    """
    with _pending_lock:
        cmds = list(_pending_commands)
        _pending_commands.clear()
    return cmds


HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>HIDAgent Debug Log</title>
<style>
* { box-sizing: border-box; }
body {
    font-family: -apple-system, sans-serif;
    margin: 0;
    padding: 0;
    background: #1e1e1e;
    color: #ddd;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}
.header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 20px;
    border-bottom: 1px solid #444;
    flex-shrink: 0;
}
h1 { font-size: 16px; margin: 0; }
.controls { display: flex; gap: 8px; align-items: center; }
.controls input, .controls button, .controls select {
    padding: 6px 10px;
    background: #2d2d2d;
    color: #ddd;
    border: 1px solid #555;
    border-radius: 4px;
    font-size: 13px;
}
.controls button:hover { background: #3d3d3d; cursor: pointer; }
#status { font-size: 11px; color: #6a6; min-width: 150px; }

.main {
    display: flex;
    flex: 1;
    overflow: hidden;
}
.live-panel {
    width: 340px;
    background: #2a2a2a;
    padding: 15px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    border-right: 1px solid #444;
}
.live-panel h2 { font-size: 13px; margin: 0 0 10px; color: #888; text-transform: uppercase; }
.live-panel img {
    width: 100%;
    display: block;
    border: 1px solid #444;
    cursor: crosshair;
}
.hint { font-size: 11px; color: #888; margin-top: 8px; }

.detail-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}
.detail-view {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: auto;
    padding: 20px;
    position: relative;
}
.detail-view img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    border: 1px solid #444;
}
.detail-info {
    position: absolute;
    top: 20px;
    left: 20px;
    background: rgba(20, 20, 20, 0.9);
    padding: 10px 14px;
    border-radius: 6px;
    border: 1px solid #444;
    font-size: 12px;
    max-width: 400px;
}
.detail-info .action {
    font-weight: 600;
    color: #6af;
    text-transform: uppercase;
    font-size: 11px;
}
.detail-info .time { color: #888; font-size: 11px; margin-left: 8px; }
.detail-info .details { margin-top: 6px; color: #ccc; white-space: pre-wrap; word-break: break-word; }
.detail-info .coord { color: #faa; font-family: monospace; font-size: 11px; margin-top: 4px; }
.detail-info .meta { color: #888; font-size: 11px; margin-top: 5px; font-family: monospace; }
.detail-empty { color: #666; font-size: 14px; }

.agent-banner {
    background: linear-gradient(180deg, #1a2638 0%, #17202e 100%);
    border-bottom: 2px solid #3a5f9f;
    padding: 16px 24px;
    flex-shrink: 0;
    max-height: 30vh;
    overflow-y: auto;
}
.agent-banner.empty { display: none; }
.agent-banner .label {
    font-size: 11px;
    color: #6af;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.agent-banner .label .time {
    color: #678; font-weight: 400; margin-left: 10px;
    text-transform: none; letter-spacing: 0;
}
.agent-banner .text {
    font-size: 22px;
    line-height: 1.4;
    color: #e8eef8;
    white-space: pre-wrap;
    word-wrap: break-word;
}

.thumbnail-bar {
    height: 140px;
    background: #242424;
    border-top: 1px solid #444;
    display: flex;
    align-items: center;
    padding: 10px;
    gap: 8px;
    overflow-x: auto;
    overflow-y: hidden;
    flex-shrink: 0;
}
.thumbnail-bar::-webkit-scrollbar { height: 10px; }
.thumbnail-bar::-webkit-scrollbar-track { background: #1e1e1e; }
.thumbnail-bar::-webkit-scrollbar-thumb { background: #555; border-radius: 5px; }
.thumbnail-bar::-webkit-scrollbar-thumb:hover { background: #666; }

.thumb {
    flex-shrink: 0;
    width: 70px;
    height: 120px;
    background: #1a1a1a;
    border: 2px solid transparent;
    border-radius: 4px;
    cursor: pointer;
    overflow: hidden;
    position: relative;
    display: flex;
    flex-direction: column;
    transition: border-color 0.15s;
}
.thumb:hover { border-color: #555; }
.thumb.selected { border-color: #6af; }
.thumb img {
    width: 100%;
    flex: 1;
    object-fit: contain;
    object-position: center;
    min-height: 0;
    background: #000;
}
.thumb .thumb-label {
    font-size: 9px;
    padding: 3px 5px;
    background: rgba(0, 0, 0, 0.85);
    color: #ccc;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
}
.thumb .thumb-label .action { color: #6af; font-weight: 600; text-transform: uppercase; }
.empty-thumb { color: #666; padding: 20px; font-size: 13px; }

/* Fullscreen modal */
.modal {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.92);
    z-index: 1000;
    align-items: center;
    justify-content: center;
    padding: 40px;
}
.modal.open { display: flex; }
.modal img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
}
.modal-close {
    position: absolute;
    top: 20px;
    right: 20px;
    background: rgba(255, 255, 255, 0.1);
    color: #fff;
    border: none;
    width: 40px;
    height: 40px;
    border-radius: 50%;
    font-size: 20px;
    cursor: pointer;
}
.modal-close:hover { background: rgba(255, 255, 255, 0.2); }
</style>
</head>
<body>

<div class="header">
    <h1>HIDAgent Debug Log</h1>
    <div class="controls">
        <input type="text" id="text-input" placeholder="Type text..." style="width: 150px;">
        <button onclick="sendText()">Send Text</button>
        <select id="key-select">
            <option value="">— key —</option>
            <option>enter</option><option>tab</option><option>escape</option>
            <option>space</option><option>delete</option><option>backspace</option>
            <option>up</option><option>down</option><option>left</option><option>right</option>
        </select>
        <button onclick="sendKey()">Send Key</button>
        <input type="text" id="run-input" placeholder="App name (e.g. Safari)..." style="width: 160px;">
        <button onclick="runCommand()">Run App</button>
        <button onclick="clearLog()">Clear Log</button>
        <span id="status"></span>
    </div>
</div>

<div class="main">
    <div class="live-panel">
        <h2>Live View</h2>
        <img id="live-img" src="/live.png" alt="Live screenshot">
        <div class="hint">Click on the image to send a click to the target device.<br>Auto-refreshes every 3s.</div>
    </div>

    <div class="detail-panel">
        <div class="agent-banner empty" id="agent-banner">
            <div class="label"><span id="agent-banner-src">Agent</span><span class="time" id="agent-banner-time"></span></div>
            <div class="text" id="agent-banner-text"></div>
        </div>
        <div class="detail-view" id="detail-view">
            <div class="detail-empty">Click a thumbnail below to view the full screenshot.</div>
        </div>
        <div class="thumbnail-bar" id="thumbnail-bar">
            <div class="empty-thumb">No log entries yet. Waiting for actions...</div>
        </div>
    </div>
</div>

<div class="modal" id="modal">
    <button class="modal-close" onclick="closeModal()">×</button>
    <img id="modal-img" src="">
</div>

<script>
const liveImg = document.getElementById('live-img');
const statusEl = document.getElementById('status');
let entries = [];
let selectedId = null;

function refreshLive() {
    liveImg.src = '/live.png?t=' + Date.now();
}
setInterval(refreshLive, 3000);

liveImg.addEventListener('click', async (e) => {
    const rect = liveImg.getBoundingClientRect();
    const scaleX = liveImg.naturalWidth / rect.width;
    const scaleY = liveImg.naturalHeight / rect.height;
    const x = Math.round((e.clientX - rect.left) * scaleX);
    const y = Math.round((e.clientY - rect.top) * scaleY);
    setStatus(`Sending click (${x}, ${y})...`);
    const res = await fetch('/cmd', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({type: 'click', x: x, y: y}),
    });
    if (res.ok) {
        setStatus(`Click (${x}, ${y}) queued`);
        setTimeout(refreshLive, 500);
        setTimeout(refreshLog, 500);
    }
});

async function sendText() {
    const input = document.getElementById('text-input');
    const text = input.value;
    if (!text) return;
    setStatus(`Sending text...`);
    const res = await fetch('/cmd', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({type: 'type', text: text}),
    });
    if (res.ok) {
        setStatus(`Text sent`);
        input.value = '';
        setTimeout(refreshLive, 500);
        setTimeout(refreshLog, 500);
    }
}

async function sendKey() {
    const select = document.getElementById('key-select');
    const key = select.value;
    if (!key) return;
    setStatus(`Sending key...`);
    const res = await fetch('/cmd', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({type: 'key', text: key}),
    });
    if (res.ok) {
        setStatus(`Key sent: ${key}`);
        setTimeout(refreshLive, 500);
        setTimeout(refreshLog, 500);
    }
}

async function runCommand() {
    const input = document.getElementById('run-input');
    const app = input.value;
    if (!app) return;
    setStatus(`Launching ${app}...`);
    const res = await fetch('/cmd', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({type: 'run', text: app}),
    });
    if (res.ok) {
        setStatus(`Launched: ${app}`);
        input.value = '';
        setTimeout(refreshLive, 800);
        setTimeout(refreshLog, 500);
    }
}

async function clearLog() {
    await fetch('/clear', { method: 'POST' });
    selectedId = null;
    refreshLog();
}

document.getElementById('text-input').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        sendText();
    }
});

document.getElementById('run-input').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        runCommand();
    }
});

function setStatus(s) {
    statusEl.textContent = s;
    setTimeout(() => { statusEl.textContent = ''; }, 3000);
}

function selectEntry(id) {
    selectedId = id;
    const entry = entries.find(e => e.id === id);
    if (!entry) return;
    const detail = document.getElementById('detail-view');
    const coord = (entry.x !== null && entry.y !== null)
        ? `<div class="coord">at (${entry.x}, ${entry.y})</div>` : '';
    const meta = Object.keys(entry.metadata || {}).length > 0
        ? `<div class="meta">${escapeHtml(JSON.stringify(entry.metadata))}</div>` : '';
    const imgTag = entry.img
        ? `<img src="data:image/png;base64,${entry.img}" onclick="openModal(this.src)">`
        : `<div style="font-size:18px;line-height:1.5;color:#e8eef8;max-width:900px;padding:40px;white-space:pre-wrap">${escapeHtml(entry.details)}</div>`;
    detail.innerHTML = `
        <div class="detail-info">
            <span class="action">${entry.action}</span>
            <span class="time">${entry.timestamp}</span>
            <div class="details">${escapeHtml(entry.details)}</div>
            ${coord}
            ${meta}
        </div>
        ${imgTag}
    `;
    // Update selected thumbnail highlighting
    document.querySelectorAll('.thumb').forEach(t => {
        t.classList.toggle('selected', parseInt(t.dataset.id) === id);
    });
}

function openModal(src) {
    document.getElementById('modal-img').src = src;
    document.getElementById('modal').classList.add('open');
}

function closeModal() {
    document.getElementById('modal').classList.remove('open');
}

document.getElementById('modal').addEventListener('click', (e) => {
    if (e.target.id === 'modal') closeModal();
});

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModal();
});

async function refreshLog() {
    const res = await fetch('/entries');
    const newEntries = await res.json();
    const hadNew = newEntries.length !== entries.length;
    entries = newEntries;

    // Update the agent message banner with the most recent message entry.
    const banner = document.getElementById('agent-banner');
    let latestMsg = null;
    for (let i = entries.length - 1; i >= 0; i--) {
        if (entries[i].action === 'message') { latestMsg = entries[i]; break; }
    }
    if (latestMsg) {
        document.getElementById('agent-banner-src').textContent =
            latestMsg.source || 'agent';
        document.getElementById('agent-banner-time').textContent =
            latestMsg.timestamp;
        document.getElementById('agent-banner-text').textContent =
            latestMsg.details;
        banner.classList.remove('empty');
    } else {
        banner.classList.add('empty');
    }

    const bar = document.getElementById('thumbnail-bar');
    if (entries.length === 0) {
        bar.innerHTML = '<div class="empty-thumb">No log entries yet. Waiting for actions...</div>';
        return;
    }
    bar.innerHTML = entries.map(e => {
        const imgSrc = e.img
            ? `data:image/png;base64,${e.img}`
            : '';
        const sel = e.id === selectedId ? ' selected' : '';
        const body = imgSrc
            ? `<img src="${imgSrc}">`
            : (e.action === 'message'
                ? `<div style="flex:1;padding:4px;font-size:9px;color:#cde;background:#1a2638;overflow:hidden">${escapeHtml((e.details || '').slice(0, 80))}</div>`
                : '<div style="flex:1;background:#333"></div>');
        return `<div class="thumb${sel}" data-id="${e.id}" onclick="selectEntry(${e.id})">
            ${body}
            <div class="thumb-label"><span class="action">${e.action}</span> ${e.timestamp}</div>
        </div>`;
    }).join('');

    // Auto-select the newest entry if we got new ones
    if (hadNew && entries.length > 0) {
        const newest = entries[entries.length - 1];
        selectEntry(newest.id);
        // Scroll thumbnail bar to the right to show newest
        setTimeout(() => { bar.scrollLeft = bar.scrollWidth; }, 50);
    }
}

function escapeHtml(s) {
    const div = document.createElement('div');
    div.textContent = s || '';
    return div.innerHTML;
}

setInterval(refreshLog, 2000);
refreshLog();
</script>
</body>
</html>
"""


class _RequestHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress default logging noise
        pass

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self._respond_html(HTML_PAGE)

        elif parsed.path == "/entries":
            with _log_lock:
                entries = list(_log_entries)
            self._respond_json(entries)

        elif parsed.path == "/live.png":
            self._respond_live_screenshot()

        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/cmd":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            try:
                cmd = json.loads(body)
            except Exception:
                self.send_error(400, "Invalid JSON")
                return

            with _pending_lock:
                _pending_commands.append(cmd)

            # Execute the command immediately via hidagent functions
            try:
                self._execute_command(cmd)
            except Exception as e:
                self.send_error(500, f"Command failed: {e}")
                return

            self._respond_json({"status": "ok"})

        elif parsed.path == "/clear":
            with _log_lock:
                _log_entries.clear()
            self._respond_json({"status": "ok"})

        else:
            self.send_error(404)

    def _execute_command(self, cmd):
        """Execute a command from the web interface.

        The browser sends coordinates in full-screenshot-pixel space
        (the live view is at native resolution). We scale down by the
        calibration resize factor before calling click_mouse.
        """
        from hidagent.hidagent import (
            click_mouse, type_text, keypress, run_application)
        cmd_type = cmd.get("type", "")
        if cmd_type == "click":
            x = cmd.get("x", 0)
            y = cmd.get("y", 0)
            # click_mouse expects calibration-space coords (half-size)
            click_mouse(x // 2, y // 2)
            if _log_entries:
                _log_entries[-1]["action"] = "web_click"
                _log_entries[-1]["details"] = f"Web click at ({x}, {y})"
        elif cmd_type == "type":
            text = cmd.get("text", "")
            type_text(text)
            if _log_entries:
                _log_entries[-1]["action"] = "web_type"
                _log_entries[-1]["details"] = f"Web typed: {text!r}"
        elif cmd_type == "key":
            key = cmd.get("text", "")
            keypress([key])
            if _log_entries:
                _log_entries[-1]["action"] = "web_key"
                _log_entries[-1]["details"] = f"Web pressed: {key}"
        elif cmd_type == "run":
            app = cmd.get("text", "")
            run_application(app)
            if _log_entries:
                _log_entries[-1]["action"] = "web_run"
                _log_entries[-1]["details"] = f"Web launched: {app}"

    def _respond_html(self, html):
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _respond_json(self, obj):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _respond_live_screenshot(self):
        try:
            from hidagent.hidagent import get_screenshot
            img = get_screenshot()
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            data = buf.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_error(500, f"Screenshot failed: {e}")


class _ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_logging_server(port=8080, auto_log_hidagent=True):
    """Start the web logging server in a background thread.

    Args:
        port: Port to listen on (default 8080).
        auto_log_hidagent: If True, monkey-patches hidagent's click_mouse,
            type_text, keypress, and run_application to automatically log
            each call.

    Returns the server instance (already running).
    """
    global _server, _server_thread

    if _server is not None:
        print(f"Logging server already running on port {_server.server_address[1]}")
        return _server

    _server = _ThreadedServer(("127.0.0.1", port), _RequestHandler)
    _server_thread = threading.Thread(target=_server.serve_forever, daemon=True)
    _server_thread.start()

    print(f"HIDAgent debug log: http://localhost:{port}")

    if auto_log_hidagent:
        _install_auto_logging()

    log_action("server_start", f"Logging server started on port {port}")
    return _server


def stop_logging_server():
    """Stop the logging server."""
    global _server
    if _server is not None:
        _server.shutdown()
        _server = None


def _install_auto_logging():
    """Monkey-patch hidagent functions to automatically log calls."""
    from hidagent import hidagent as ha

    orig_click = ha.click_mouse
    orig_move = ha.move_mouse
    orig_type = ha.type_text
    orig_key = ha.keypress
    orig_run = ha.run_application

    def wrapped_click(x, y, button="left"):
        # click_mouse receives coordinates in the calibration's half-size space.
        # Scale them up to full screenshot space for display.
        log_action("click", f"{button} click", x=int(x), y=int(y),
                   coord_space="calibration")
        return orig_click(x, y, button)

    def wrapped_move(x, y):
        log_action("move", f"move to ({x}, {y})", x=int(x), y=int(y),
                   coord_space="calibration")
        return orig_move(x, y)

    def wrapped_type(text):
        log_action("type", f"typed: {text!r}")
        return orig_type(text)

    def wrapped_key(keys):
        log_action("key", f"pressed: {keys}")
        return orig_key(keys)

    def wrapped_run(app_name):
        log_action("run", f"launched: {app_name}")
        return orig_run(app_name)

    ha.click_mouse = wrapped_click
    ha.move_mouse = wrapped_move
    ha.type_text = wrapped_type
    ha.keypress = wrapped_key
    ha.run_application = wrapped_run
