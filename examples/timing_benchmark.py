"""
Timing benchmark for HIDAgent components.

Runs each component 1000 times and reports timing statistics:
  1. get_screenshot()        — capture via HDMI
  2. send_serial_message()   — send a "run" command over serial
  3. hid_move(100, 100)      — raw firmware move to a fixed position
  4. gui_diff(img1, img2)    — diff two different screenshots

Usage:
    python timing_benchmark.py [--iterations 1000]
"""

from dotenv import load_dotenv
load_dotenv()

import os
import sys
import time
import argparse
import statistics
import base64
from io import BytesIO

# Allow running from examples/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hidagent.hidagent import (
    get_screenshot, send_serial_message, gui_diff,
)
# The raw firmware move (move_mouse() now also watches the pointer).
from hidagent.cursor import hid_move
from anthropic import Anthropic
import json


def benchmark(func, n, label, subtract_s=0.0):
    """Run func() n times, collect per-call timings, print stats.

    Args:
        subtract_s: Fixed overhead (in seconds) to subtract from each
            measurement (e.g. known time.sleep delays in the function).
    """
    times = []
    print(f"\n{'='*60}")
    print(f"Benchmarking: {label}  ({n} iterations)")
    if subtract_s > 0:
        print(f"  (subtracting {subtract_s*1000:.0f} ms known sleep per call)")
    print(f"{'='*60}")

    for i in range(n + 1):  # +1 for warmup iteration 0
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        if i == 0:
            print(f"  warmup: {(t1 - t0)*1000:.1f} ms (excluded)")
            continue
        elapsed = max(0.0, (t1 - t0) - subtract_s)
        times.append(elapsed)

        # Progress indicator every 10%
        if i % max(1, n // 10) == 0:
            print(f"  {i}/{n} done  (last: {elapsed*1000:.1f} ms)")

    total = sum(times)
    avg = statistics.mean(times)
    med = statistics.median(times)
    mn = min(times)
    mx = max(times)
    sd = statistics.stdev(times) if len(times) > 1 else 0.0

    print(f"\nResults for: {label}")
    print(f"  Total:   {total:.3f} s")
    print(f"  Mean:    {avg*1000:.2f} ms")
    print(f"  Median:  {med*1000:.2f} ms")
    print(f"  Min:     {mn*1000:.2f} ms")
    print(f"  Max:     {mx*1000:.2f} ms")
    print(f"  Stdev:   {sd*1000:.2f} ms")

    return {
        "label": label,
        "iterations": n,
        "total_s": total,
        "mean_ms": avg * 1000,
        "median_ms": med * 1000,
        "min_ms": mn * 1000,
        "max_ms": mx * 1000,
        "stdev_ms": sd * 1000,
    }


def main():
    parser = argparse.ArgumentParser(description="HIDAgent timing benchmark")
    parser.add_argument("--iterations", "-n", type=int, default=1000,
                        help="Number of iterations per benchmark (default: 1000)")
    args = parser.parse_args()
    n = args.iterations

    results = []

    # ----------------------------------------------------------------
    # 1. Screenshot capture
    # ----------------------------------------------------------------
    results.append(benchmark(get_screenshot, n, "get_screenshot()"))

    # ----------------------------------------------------------------
    # 2. Serial command (key press — no device-side delays)
    # ----------------------------------------------------------------
    key_cmd = json.dumps({"type": "key", "text": "escape"}) + "\n"

    def serial_cmd():
        send_serial_message(key_cmd)

    results.append(benchmark(serial_cmd, n, 'send_serial_message (key press, round trip)'))

    # ----------------------------------------------------------------
    # 3. Move mouse to (50, 50)
    # ----------------------------------------------------------------
    def mouse_move():
        hid_move(100, 100)

    results.append(benchmark(mouse_move, n, "hid_move(100, 100)"))

    # ----------------------------------------------------------------
    # 4. gui_diff with varying image pairs
    #    Capture a pool of screenshots first, then diff consecutive pairs
    #    so we aren't timing screenshot capture inside the diff benchmark.
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Pre-capturing {min(n + 1, 50)} screenshots for gui_diff pool...")
    print(f"{'='*60}")

    pool_size = min(n + 1, 50)  # cap pool size; we'll cycle through them
    screenshot_pool = []
    for i in range(pool_size):
        img = get_screenshot()
        screenshot_pool.append(img)
        if (i + 1) % max(1, pool_size // 5) == 0:
            print(f"  Captured {i+1}/{pool_size}")
        # Small mouse jitter between captures to ensure images differ
        hid_move(100 + (i % 5) * 40, 100 + (i % 3) * 40)
        time.sleep(0.05)

    pair_idx = [0]

    def diff_images():
        idx = pair_idx[0] % (len(screenshot_pool) - 1)
        pair_idx[0] += 1
        gui_diff(screenshot_pool[idx], screenshot_pool[idx + 1])

    results.append(benchmark(diff_images, n, "gui_diff (varying image pairs)"))

    # ----------------------------------------------------------------
    # 5. VLM round-trip (screenshot → Claude → response)
    # ----------------------------------------------------------------
    vlm_n = min(n, 10)  # cap VLM iterations — API calls are expensive/slow
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Pre-encode a screenshot for reuse
    vlm_img = get_screenshot()
    buf = BytesIO()
    vlm_img.save(buf, format="PNG")
    vlm_img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    display_w, display_h = vlm_img.size
    vlm_tools = [
        {
            "type": "computer_20251124",
            "name": "computer",
            "display_width_px": display_w,
            "display_height_px": display_h,
        },
    ]

    def vlm_call():
        client.beta.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            betas=["computer-use-2025-11-24"],
            tools=vlm_tools,
            system="You are controlling a device. Perform the requested action.",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png",
                        "data": vlm_img_b64,
                    }},
                    {"type": "text",
                     "text": "Click on the Settings app icon."},
                ],
            }],
        )

    print(f"\n(VLM benchmark capped at {vlm_n} iterations)")
    results.append(benchmark(vlm_call, vlm_n, "VLM (Claude Sonnet)"))

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Component':<40} {'Mean (ms)':>10} {'Median (ms)':>12} {'Stdev (ms)':>11}")
    print("-" * 75)
    for r in results:
        print(f"{r['label']:<40} {r['mean_ms']:>10.2f} {r['median_ms']:>12.2f} {r['stdev_ms']:>11.2f}")
    print()


if __name__ == "__main__":
    main()
