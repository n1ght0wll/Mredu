"""
main.py — MrEdu v1.0 (Gemini + Streaming)
The orchestration layer. This is the brain of the agent.

STREAMING EXPLAINED:
Without streaming: wait for full response → print everything at once
With streaming:    print each chunk as it arrives → feels instant

The API sends the response in small chunks (tokens).
We print each chunk immediately as it arrives using
client.models.generate_content_stream() instead of
client.chats.create() — streaming requires a slightly
different call pattern but the concept is identical.

We manually maintain the messages array so the model
has full conversation history on every call.
"""

from google import genai
from google.genai import types
import os
import sys

from config import MODEL, API_KEY, MAX_TOKENS, TEMPERATURE


# ──────────────────────────────────────────────────────────
# STEP 1: Load the system prompt
# ──────────────────────────────────────────────────────────

def load_system_prompt(filepath="system_prompt.txt"):
    if not os.path.exists(filepath):
        print(f"[ERROR] system_prompt.txt not found at: {filepath}")
        print("MrEdu cannot run without a system prompt.")
        sys.exit(1)
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


# ──────────────────────────────────────────────────────────
# STEP 2: Initialise the Gemini client
# ──────────────────────────────────────────────────────────

def create_client():
    if not API_KEY:
        print("[ERROR] GEMINI_API_KEY environment variable is not set.")
        print("Run: set GEMINI_API_KEY=your-key-here")
        sys.exit(1)
    return genai.Client(api_key=API_KEY)


# ──────────────────────────────────────────────────────────
# STEP 3: Stream a response
#
# generate_content_stream() returns an iterator.
# Each iteration yields a chunk object with a .text field.
# We print each chunk immediately with end="" and flush=True
# so it appears word by word without waiting for the full response.
#
# We also build the full response string as we go so we can
# append it to the messages array after streaming completes.
#
# messages format for Gemini:
# [
#   {"role": "user",  "parts": [{"text": "..."}]},
#   {"role": "model", "parts": [{"text": "..."}]},
# ]
# Note: Gemini uses "model" where Anthropic uses "assistant"
# ──────────────────────────────────────────────────────────

def stream_response(client, system_prompt, messages):
    """
    Streams MrEdu's response chunk by chunk.
    Prints each chunk as it arrives.
    Returns the full response text when done.
    """
    full_response = ""

    response_stream = client.models.generate_content_stream(
        model=MODEL,
        contents=messages,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
    )

    # Iterate over chunks as they arrive from the API
    for chunk in response_stream:
        if chunk.text:
            print(chunk.text, end="", flush=True)
            full_response += chunk.text

    print()  # Newline after streaming completes
    return full_response


# ──────────────────────────────────────────────────────────
# STEP 4: The conversation loop
# ──────────────────────────────────────────────────────────

def run_conversation(client, system_prompt):
    # We manually maintain messages here because streaming
    # requires direct access to the messages array.
    # Same concept as before — full history every call.
    messages = []

    print("\n" + "="*60)
    print("  MrEdu v1.0 — Learn how I was built by talking to me")
    print("  Type 'exit' or 'quit' to end the session")
    print("="*60 + "\n")

    while True:

        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nSession ended.")
            break

        if user_input.lower() in ["exit", "quit", "q"]:
            print("\nMrEdu: Session ended. Come back when you want to learn more.")
            break

        if not user_input:
            continue

        # Append user message in Gemini's expected format
        messages.append({
            "role": "user",
            "parts": [{"text": user_input}]
        })

        try:
            print("\nMrEdu: ", end="", flush=True)
            response_text = stream_response(client, system_prompt, messages)

        except Exception as e:
            error = str(e)
            if "api_key" in error.lower() or "authentication" in error.lower():
                print("[ERROR] Invalid API key. Check your GEMINI_API_KEY.")
                sys.exit(1)
            elif "quota" in error.lower() or "rate" in error.lower() or "429" in error:
                print("[ERROR] Rate limit hit. Wait a moment and try again.")
                messages.pop()
                continue
            elif "connect" in error.lower() or "network" in error.lower():
                print("[ERROR] Connection failed. Check your internet.")
                messages.pop()
                continue
            else:
                print(f"[ERROR] {error}")
                messages.pop()
                continue

        # Append model response to history
        messages.append({
            "role": "model",
            "parts": [{"text": response_text}]
        })

        print()


# ──────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    system_prompt = load_system_prompt()
    client = create_client()
    run_conversation(client, system_prompt)
