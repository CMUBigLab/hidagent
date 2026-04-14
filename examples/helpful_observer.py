"""
The Helpful Observer

An advisory assistant that captures the current screen and answers user
questions about what they see. Does not take control of the device — just
observes and provides contextual help.

Supports Claude (cloud) or Gemma (local) as the reasoning model.

Usage:
    python helpful_observer.py
    python helpful_observer.py --model gemma
"""

from dotenv import load_dotenv
load_dotenv()

import os
import sys
# Allow running from examples/ directory by adding parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import base64
import argparse
from io import BytesIO
from hidagent.hidagent import get_screenshot, llm_screenshot_query


def get_screenshot_base64():
    """Capture screenshot and return as base64."""
    img = get_screenshot()
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8"), img


def query_claude(screenshot_b64, question):
    """Ask Claude about the screenshot."""
    from anthropic import Anthropic

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-opus-4-1-20250805",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": screenshot_b64,
                }},
            ],
        }],
    )
    return response.content[0].text


def query_gemma(screenshot_b64, question):
    """Ask Gemma (local via Ollama) about the screenshot."""
    import ollama

    response = ollama.chat(
        model="gemma:27b",
        messages=[{
            "role": "user",
            "content": f"{question}\n\n[Screenshot attached as base64:{screenshot_b64[:100]}...]",
        }],
    )
    return response["message"]["content"]


def observer_loop(model="claude"):
    """Interactive loop: user asks questions, observer captures and answers."""
    query_fn = query_claude if model == "claude" else query_gemma
    print(f"Helpful Observer (using {model})")
    print("Type your question about the current screen, or 'quit' to exit.\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        print("  Capturing screenshot...")
        screenshot_b64, screenshot_img = get_screenshot_base64()

        print("  Thinking...")
        answer = query_fn(screenshot_b64, question)

        print(f"\n  {answer}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Helpful Observer via HIDAgent")
    parser.add_argument("--model", choices=["claude", "gemma"], default="claude",
                        help="Which model to use for reasoning")
    args = parser.parse_args()
    observer_loop(args.model)
