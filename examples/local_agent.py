"""
Extensible UI Agent (Local/On-Device variant)

Uses Omniparser for screenshot parsing and Gemma (via Ollama) for reasoning.
Runs entirely locally — no cloud API keys required.

Usage:
    python local_agent.py "Open the notes app and write a reminder"
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
import ollama
from hidagent.hidagent import (
    get_screenshot, click_mouse, type_text, keypress, run_application,
    recognize_gui_elements
)


OMNIPARSER_CONFIG = {
    "som_model_path": "weights/icon_detect/model.pt",
    "caption_model_name": "florence2",
    "caption_model_path": "weights/icon_caption_florence",
    "BOX_TRESHOLD": 0.05,
}

OLLAMA_MODEL = "gemma:27b"


def get_screenshot_base64():
    """Capture screenshot and return as base64 string."""
    img = get_screenshot()
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8"), img


def query_local_llm(parsed_elements, image_base64, task_description):
    """Ask Gemma to suggest the next action given parsed UI elements."""
    prompt = (
        "Given the parsed list of elements from a UI screenshot, suggest a "
        "specific action to take. Return your response as valid JSON only.\n\n"
        f"Task: {task_description}\n\n"
        f"UI Elements: {json.dumps(parsed_elements, indent=2)}\n\n"
        "Return one of:\n"
        '  {"action": "click", "location": {"x": 0.5, "y": 0.3}}\n'
        '  {"action": "type", "text": "hello"}\n'
        '  {"action": "key", "text": "enter"}\n'
        '  {"action": "run", "text": "app_name"}\n'
        '  {"action": "done", "summary": "Task completed"}\n'
    )

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def execute_action(action_json, image_size):
    """Execute an action on the target device."""
    width, height = image_size

    if action_json["action"] == "click":
        px = int(action_json["location"]["x"] * width)
        py = int(action_json["location"]["y"] * height)
        click_mouse(px, py)
        print(f"  Clicked at ({px}, {py})")

    elif action_json["action"] == "type":
        type_text(action_json["text"])
        print(f"  Typed: {action_json['text']}")

    elif action_json["action"] == "key":
        keypress(action_json["text"].split("+"))
        print(f"  Key: {action_json['text']}")

    elif action_json["action"] == "run":
        run_application(action_json["text"])
        print(f"  Launched: {action_json['text']}")

    elif action_json["action"] == "done":
        print(f"  Done: {action_json.get('summary', 'Task completed')}")
        return False

    return True


def agent_loop(task, max_iterations=10):
    """Main agent loop using local models only."""
    print(f"Task: {task}")

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")

        image_base64, image = get_screenshot_base64()
        parsed_elements = recognize_gui_elements(image)
        print(f"  Found {len(parsed_elements)} UI elements")

        response_text = query_local_llm(parsed_elements, image_base64, task)
        print(f"  LLM response: {response_text[:200]}")

        try:
            action = json.loads(response_text)
        except json.JSONDecodeError:
            print("  Failed to parse LLM response as JSON")
            continue

        should_continue = execute_action(action, image.size)
        if not should_continue:
            return

        time.sleep(1)

    print("Reached max iterations.")


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "Take a screenshot and describe what you see."
    agent_loop(task)
