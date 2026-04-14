"""
Simulator Backend for HIDAgent

Replaces the physical HID bridge + USB camera with the macOS Simulator app.
Provides the same interface (click, type, key, screenshot, run) so all
HIDAgent examples work without hardware.

Uses:
- xcrun simctl: screenshots, app launching, device management
- cliclick: mouse/keyboard control (install via: brew install cliclick)
- AppleScript: window management

Usage:
    from hidagent.simulator import SimulatorBackend
    sim = SimulatorBackend()
    sim.boot()

    # Same interface as hardware HIDAgent
    screenshot = sim.screenshot()
    sim.click(200, 300)
    sim.type_text("hello")
    sim.key("enter")
    sim.run_application("Safari")
"""

import subprocess
import json
import time
import os
import tempfile

from PIL import Image
from Quartz import (
    CGWindowListCopyWindowInfo,
    kCGWindowListOptionOnScreenOnly,
    kCGNullWindowID,
)


def _cliclick(args):
    """Run a cliclick command. Install via: brew install cliclick"""
    result = subprocess.run(["cliclick"] + args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"cliclick failed: {result.stderr}")


class SimulatorBackend:
    """Drop-in replacement for the HID hardware bridge, targeting macOS Simulator."""

    def __init__(self, device_name=None):
        """Initialize the simulator backend.

        Args:
            device_name: Simulator device name (e.g., "iPhone 16").
                If None, uses whatever device is currently booted,
                or boots the first available iPhone.
        """
        self.device_udid = None
        self.device_name = device_name
        self._window_bounds = None
        self._screen_region = None  # (x, y, w, h) of actual screen within window
        self._device_size = None    # (width, height) of device screen in pixels

    def boot(self):
        """Boot the simulator and bring it to the foreground."""
        if self.device_name:
            # Find the UDID for the requested device
            self.device_udid = self._find_device_udid(self.device_name)
            if not self.device_udid:
                raise RuntimeError(f"Simulator device '{self.device_name}' not found. "
                                   f"Available: {', '.join(self._list_device_names())}")
            subprocess.run(["xcrun", "simctl", "boot", self.device_udid],
                           capture_output=True)
        else:
            # Check if anything is already booted
            booted = self._get_booted_udid()
            if booted:
                self.device_udid = booted
            else:
                # Boot the first available iPhone
                for name in self._list_device_names():
                    if "iPhone" in name:
                        self.device_udid = self._find_device_udid(name)
                        subprocess.run(["xcrun", "simctl", "boot", self.device_udid],
                                       capture_output=True)
                        self.device_name = name
                        break
                if not self.device_udid:
                    raise RuntimeError("No iPhone simulator device available.")

        # Open the Simulator app and bring to front
        subprocess.run(["open", "-a", "Simulator"], capture_output=True)
        time.sleep(2)
        self.bring_to_foreground()
        time.sleep(1)

        # Cache window bounds and detect actual screen region
        self._window_bounds = self._get_window_bounds()
        if self._window_bounds:
            print(f"Simulator booted: {self.device_name or self.device_udid}")
            print(f"  Window bounds: {self._window_bounds}")
            self._detect_screen_region()
        else:
            print("Warning: Could not find Simulator window bounds.")

    def shutdown(self):
        """Shutdown the simulator."""
        if self.device_udid:
            subprocess.run(["xcrun", "simctl", "shutdown", self.device_udid],
                           capture_output=True)

    def screenshot(self):
        """Capture a screenshot of the simulated device. Returns PIL Image."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name

        try:
            result = subprocess.run(
                ["xcrun", "simctl", "io", "booted", "screenshot", tmp_path],
                capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Screenshot failed: {result.stderr}")
            img = Image.open(tmp_path).copy()
        finally:
            os.unlink(tmp_path)

        return img

    def click(self, x, y):
        """Click at device coordinates (x, y) in the Simulator.

        Coordinates are in the device's screen space (e.g., 0-393 x 0-852
        for iPhone 16). They get mapped to the Simulator window's position
        on the Mac screen.
        """
        screen_x, screen_y = self._device_to_screen(x, y)
        self.bring_to_foreground()
        time.sleep(0.1)
        _cliclick([f"c:{screen_x},{screen_y}"])

    def type_text(self, text):
        """Type text into the Simulator."""
        self.bring_to_foreground()
        time.sleep(0.1)
        # cliclick t: types a string
        _cliclick([f"t:{text}"])

    def key(self, key_name):
        """Press a key or key combination.

        Supports: enter, tab, escape, space, delete, up, down, left, right,
        and combinations like "command+space", "shift+a".
        """
        self.bring_to_foreground()
        time.sleep(0.1)

        # Map HIDAgent key names to cliclick key-press (kp:) codes
        # cliclick uses: return, tab, esc, space, delete, arrow-up, etc.
        key_map = {
            "enter": "return",
            "return": "return",
            "tab": "tab",
            "escape": "esc",
            "esc": "esc",
            "space": "space",
            "spacebar": "space",
            "delete": "delete",
            "backspace": "delete",
            "up": "arrow-up",
            "down": "arrow-down",
            "left": "arrow-left",
            "right": "arrow-right",
            "home": "home",
            "end": "end",
            "pageup": "page-up",
            "pagedown": "page-down",
        }

        # Handle key combinations (e.g., "command+space")
        if "+" in key_name:
            parts = [p.strip().lower() for p in key_name.split("+")]
            # cliclick uses kd/ku (key down/up) for modifiers
            modifiers = []
            main_key = None
            for p in parts:
                if p in ("cmd", "command", "super", "win"):
                    modifiers.append("cmd")
                elif p in ("ctrl", "control"):
                    modifiers.append("ctrl")
                elif p in ("alt", "option"):
                    modifiers.append("alt")
                elif p in ("shift",):
                    modifiers.append("shift")
                elif p in ("fn",):
                    modifiers.append("fn")
                else:
                    main_key = key_map.get(p, p)

            # Press modifiers down, press key, release modifiers
            cmds = [f"kd:{m}" for m in modifiers]
            if main_key:
                cmds.append(f"kp:{main_key}")
            cmds.extend(f"ku:{m}" for m in reversed(modifiers))
            _cliclick(cmds)
        else:
            mapped = key_map.get(key_name.lower(), key_name.lower())
            _cliclick([f"kp:{mapped}"])

    def run_application(self, app_name):
        """Launch an application in the Simulator.

        Uses swipe-down for Spotlight search, then types the app name.
        """
        self.bring_to_foreground()
        time.sleep(0.2)

        # Swipe up to go to home screen first
        self._swipe_up_home()
        time.sleep(0.5)

        # Swipe down from middle of home screen to open Spotlight search
        bounds = self._get_window_bounds()
        if bounds:
            mid_x = int(bounds["X"] + bounds["Width"] // 2)
            mid_y = int(bounds["Y"] + bounds["Height"] // 2)
            _cliclick([f"c:{mid_x},{mid_y}"])
            time.sleep(0.2)
            # Drag down to open search
            top_y = mid_y - 50
            bot_y = mid_y + 50
            _cliclick([f"dd:{mid_x},{top_y}", f"du:{mid_x},{bot_y}"])
            time.sleep(0.5)

        # Type the app name and press enter
        _cliclick([f"t:{app_name}"])
        time.sleep(0.5)
        _cliclick(["kp:return"])
        time.sleep(1)

    def move(self, x, y):
        """Move the mouse to device coordinates without clicking."""
        screen_x, screen_y = self._device_to_screen(x, y)
        self.bring_to_foreground()
        _cliclick([f"m:{screen_x},{screen_y}"])

    def drag(self, x1, y1, x2, y2, duration=0.5):
        """Drag from one device coordinate to another (for scrolling)."""
        sx1, sy1 = self._device_to_screen(x1, y1)
        sx2, sy2 = self._device_to_screen(x2, y2)
        self.bring_to_foreground()
        time.sleep(0.1)
        # cliclick dd: = drag down (mouse down), du: = drag up (mouse up)
        _cliclick([f"dd:{sx1},{sy1}", f"du:{sx2},{sy2}"])

    def scroll(self, x, y, amount):
        """Scroll at device coordinates. Negative amount = scroll down."""
        screen_x, screen_y = self._device_to_screen(x, y)
        self.bring_to_foreground()
        time.sleep(0.1)
        # Simulate scroll by dragging
        drag_amount = amount * 5  # scale scroll amount to pixels
        _cliclick([f"dd:{screen_x},{screen_y}",
                   f"du:{screen_x},{screen_y + drag_amount}"])

    def bring_to_foreground(self):
        """Bring the Simulator window to the front."""
        subprocess.run(
            ["osascript", "-e", 'tell app "Simulator" to activate'],
            capture_output=True)

    # --- Private helpers ---

    def _device_to_screen(self, device_x, device_y):
        """Convert device coordinates to Mac screen coordinates.

        The Simulator window has a bezel (black border) around the actual
        iPhone screen. We detect the screen region within the window and
        map device coordinates to that region.
        """
        if not self._screen_region:
            self._detect_screen_region()

        if not self._screen_region:
            raise RuntimeError("Cannot detect screen region in Simulator window.")

        region_x, region_y, region_w, region_h = self._screen_region
        device_w, device_h = self._device_size

        scale_x = region_w / device_w
        scale_y = region_h / device_h

        screen_x = int(region_x + device_x * scale_x)
        screen_y = int(region_y + device_y * scale_y)

        return screen_x, screen_y

    def _detect_screen_region(self):
        """Detect the actual iPhone screen area within the Simulator window.

        The Simulator window has a title bar and a bezel around the device
        screen. We compute the screen region by fitting the device's aspect
        ratio into the window, centered with equal margins.
        """
        bounds = self._get_window_bounds()
        if not bounds:
            return

        win_x = int(bounds["X"])
        win_y = int(bounds["Y"])
        win_w = int(bounds["Width"])
        win_h = int(bounds["Height"])

        # Get device dimensions from simctl screenshot
        device_img = self.screenshot()
        self._device_size = device_img.size
        device_w, device_h = self._device_size
        device_aspect = device_w / device_h

        # Auto-detect title bar + bezel by finding the title bar height
        # that produces a side bezel consistent with the Simulator's device
        # frame rendering. The device screen is centered in the window below
        # the title bar, with equal horizontal padding (bezel).
        #
        # We search for a title bar height where the computed layout produces
        # a reasonable bezel (> 0 and < 10% of window width).
        best_tb = 28  # fallback
        for tb in range(15, 60):
            avail_h = win_h - tb
            fit_w = avail_h * device_aspect
            side_bezel = (win_w - fit_w) / 2
            if 5 <= side_bezel <= 20:
                best_tb = tb
                break

        avail_h = win_h - best_tb
        screen_w = int(avail_h * device_aspect)
        screen_h = avail_h
        pad_x = (win_w - screen_w) // 2

        region_x = win_x + pad_x
        region_y = win_y + best_tb

        self._screen_region = (region_x, region_y, screen_w, screen_h)

        print(f"  Title bar: {best_tb}px, side bezel: {pad_x}px")
        print(f"  Screen region: ({region_x},{region_y}) {screen_w}x{screen_h}")
        print(f"  Device resolution: {device_w}x{device_h}")
        print(f"  Scale: {screen_w/device_w:.4f}x, {screen_h/device_h:.4f}x")

    def _get_window_bounds(self):
        """Get the Simulator window bounds on screen."""
        windows = CGWindowListCopyWindowInfo(
            kCGWindowListOptionOnScreenOnly, kCGNullWindowID)

        for window in windows:
            owner = window.get("kCGWindowOwnerName", "")
            name = window.get("kCGWindowName", "")
            # Look for the main Simulator device window (not menus/panels)
            if owner == "Simulator" and name and "Simulator" not in name:
                return dict(window.get("kCGWindowBounds", {}))

        # Fallback: any Simulator window
        for window in windows:
            owner = window.get("kCGWindowOwnerName", "")
            if owner == "Simulator" and window.get("kCGWindowBounds"):
                bounds = dict(window.get("kCGWindowBounds", {}))
                if bounds.get("Width", 0) > 100 and bounds.get("Height", 0) > 100:
                    return bounds

        return None

    def _swipe_up_home(self):
        """Swipe up from bottom to go home (iPhone gesture)."""
        bounds = self._get_window_bounds()
        if bounds:
            mid_x = int(bounds["X"] + bounds["Width"] // 2)
            bottom_y = int(bounds["Y"] + bounds["Height"] - 20)
            top_y = bottom_y - 200
            _cliclick([f"dd:{mid_x},{bottom_y}", f"du:{mid_x},{top_y}"])

    def _find_device_udid(self, name):
        """Find the UDID for a device by name."""
        result = subprocess.run(
            ["xcrun", "simctl", "list", "devices", "--json"],
            capture_output=True, text=True)
        devices = json.loads(result.stdout)
        for runtime, device_list in devices.get("devices", {}).items():
            for device in device_list:
                if device["name"] == name and device.get("isAvailable", True):
                    return device["udid"]
        return None

    def _get_booted_udid(self):
        """Get the UDID of the currently booted device, if any."""
        result = subprocess.run(
            ["xcrun", "simctl", "list", "devices", "--json"],
            capture_output=True, text=True)
        devices = json.loads(result.stdout)
        for runtime, device_list in devices.get("devices", {}).items():
            for device in device_list:
                if device["state"] == "Booted":
                    self.device_name = device["name"]
                    return device["udid"]
        return None

    def _list_device_names(self):
        """List available simulator device names."""
        result = subprocess.run(
            ["xcrun", "simctl", "list", "devices", "--json"],
            capture_output=True, text=True)
        devices = json.loads(result.stdout)
        names = []
        for runtime, device_list in devices.get("devices", {}).items():
            for device in device_list:
                if device.get("isAvailable", True):
                    names.append(device["name"])
        return names


def patch_hidagent(sim):
    """Monkey-patch hidagent module to use the Simulator backend instead of hardware.

    After calling this, all hidagent functions (get_image_from_usb,
    send_serial_message, etc.) route through the Simulator.

    Usage:
        from hidagent.simulator import SimulatorBackend, patch_hidagent
        sim = SimulatorBackend()
        sim.boot()
        patch_hidagent(sim)

        # Now run any example as normal
        from examples.claude_agent import sampling_loop
        sampling_loop("Open Safari")
    """
    import hidagent.hidagent as ha
    import numpy as np

    def sim_get_image_from_usb():
        """Replacement for get_image_from_usb using Simulator screenshots."""
        return sim.screenshot()

    def sim_send_serial_message(message):
        """Replacement for send_serial_message that routes to Simulator."""
        msg = message.strip()
        try:
            cmd = json.loads(msg)
        except json.JSONDecodeError:
            print(f"  [Simulator] Invalid JSON: {msg}")
            return

        cmd_type = cmd.get("type", "")

        if cmd_type == "click":
            # Coordinates are already in HID space — convert back to device pixels
            # by dividing by the ratio
            x = cmd.get("x", 0)
            y = cmd.get("y", 0)
            device_x = int(x / ha.hid_x_ratio) if ha.hid_x_ratio else x
            device_y = int(y / ha.hid_y_ratio) if ha.hid_y_ratio else y
            sim.click(device_x, device_y)

        elif cmd_type == "move":
            x = cmd.get("x", 0)
            y = cmd.get("y", 0)
            device_x = int(x / ha.hid_x_ratio) if ha.hid_x_ratio else x
            device_y = int(y / ha.hid_y_ratio) if ha.hid_y_ratio else y
            sim.move(device_x, device_y)

        elif cmd_type == "type":
            sim.type_text(cmd.get("text", ""))

        elif cmd_type == "key":
            sim.key(cmd.get("text", ""))

        elif cmd_type == "run":
            sim.run_application(cmd.get("text", ""))

        elif cmd_type == "dragdown":
            # Scroll down from the given position
            x = cmd.get("x", 0)
            y = cmd.get("y", 0)
            device_x = int(x / ha.hid_x_ratio) if ha.hid_x_ratio else x
            device_y = int(y / ha.hid_y_ratio) if ha.hid_y_ratio else y
            sim.drag(device_x, device_y,
                     device_x, device_y + 125, duration=0.3)

        else:
            print(f"  [Simulator] Unknown command type: {cmd_type}")

    # Apply patches
    ha.get_image_from_usb = sim_get_image_from_usb
    ha.send_serial_message = sim_send_serial_message

    # Set ratios to 1:1 since Simulator screenshots are in device pixels
    # and the patch converts back from HID coords to device coords
    ha.hid_x_ratio = 1.0
    ha.hid_y_ratio = 1.0

    print("HIDAgent patched to use Simulator backend.")


# Also update pyautogui references in the demo



if __name__ == "__main__":
    # Quick test: boot simulator, take screenshot, click center
    sim = SimulatorBackend()
    sim.boot()
    time.sleep(2)

    img = sim.screenshot()
    print(f"Screenshot: {img.size}")
    img.save("simulator_test.png")

    # Click center of screen
    w, h = img.size
    sim.click(w // 2, h // 2)
    print("Clicked center of screen.")
