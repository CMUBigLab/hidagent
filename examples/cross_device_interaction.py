"""
Cross-Device Interaction

Demonstrates using HIDAgent to complete a task that spans multiple devices.
The agent's context (conversation history) lives on the control computer,
so it persists when HIDAgent is physically moved between target devices.

Example workflow:
  1. Connect to laptop → agent researches how to enable high contrast mode
  2. Disconnect, connect to iPhone → agent applies the steps it found

Usage:
    python cross_device_interaction.py
"""

from dotenv import load_dotenv
load_dotenv()

import os
import sys
# Allow running from examples/ directory by adding parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import time
from claude_agent import sampling_loop, get_screenshot, run_tool
from anthropic import Anthropic


def cross_device_session():
    """Guide the user through a cross-device task."""
    print("=" * 60)
    print("Cross-Device Interaction via HIDAgent")
    print("=" * 60)
    print()
    print("This demo completes a task across two different devices.")
    print("The agent's memory persists on this computer when you")
    print("move HIDAgent between devices.")
    print()

    # Phase 1: Research on first device
    print("PHASE 1: Research")
    print("-" * 40)
    research_task = input("Enter the research task (e.g., 'Find how to enable "
                          "high contrast mode on an iPhone'): ").strip()
    if not research_task:
        research_task = "Find how to enable high contrast mode on an iPhone"

    print(f"\nRunning agent on current device: {research_task}")
    messages = sampling_loop(research_task)

    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)

    # Wait for device switch
    print("\n" + "=" * 60)
    print("PHASE 2: Switch devices")
    print("-" * 40)
    input("Disconnect HIDAgent from this device and connect it to the\n"
          "target device. Press Enter when ready...")

    # Phase 2: Apply on second device, continuing the same conversation
    # so the agent remembers everything from Phase 1.
    apply_task = input("\nEnter the task for the target device (or press Enter "
                       "to apply the findings): ").strip()
    if not apply_task:
        apply_task = ("I've now switched to the target device. "
                      "Please apply the steps you found earlier on this device.")

    print(f"\nRunning agent on target device...")
    messages = sampling_loop(apply_task, messages=messages)

    print("\n" + "=" * 60)
    print("Cross-device task complete!")
    print("=" * 60)


if __name__ == "__main__":
    cross_device_session()
