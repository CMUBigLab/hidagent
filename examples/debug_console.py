"""
HIDAgent Debug Console — Manual interactive control via web interface.

Starts a web-based debug log on http://localhost:8080 where you can:
  - See a live view of the target device (auto-refreshes every 3s)
  - Click on the live view to send clicks to the device
  - Type text and send it with the "Send Text" button
  - Send special keys (enter, tab, escape, arrows, etc.)
  - Review a scrollable history of every action sent to the device

No AI agent is involved — this is purely for manual debugging and for
exploring the target device through the browser.

Usage:
    python debug_console.py [--port 8080]
"""

import os
import sys
# Allow running from examples/ directory by adding parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time

from hidagent.logger import start_logging_server, stop_logging_server


def main():
    parser = argparse.ArgumentParser(
        description="Manual HIDAgent debug console")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port for the web interface (default 8080)")
    args = parser.parse_args()

    server = start_logging_server(port=args.port, auto_log_hidagent=True)

    print()
    print("=" * 60)
    print(f"  HIDAgent Debug Console")
    print(f"  Open http://localhost:{args.port} in your browser")
    print("=" * 60)
    print()
    print("  - Click on the live view to send clicks to the target device")
    print("  - Type text in the input box to send to the target")
    print("  - Use the key dropdown for special keys")
    print("  - All actions are logged in the history below the live view")
    print()
    print("  Press Ctrl+C here to exit.")
    print()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        stop_logging_server()


if __name__ == "__main__":
    main()
