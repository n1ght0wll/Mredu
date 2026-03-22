"""
main.py — MrEdu v1.0 (google-genai SDK)
The orchestration layer. This is the brain of the agent.

What this file does:
1. Loads the system prompt from system_prompt.txt
2. Initialises the Gemini client using your API key
3. Runs a conversation loop:
   - Takes user input
   - Sends full conversation history to the API
   - Receives the response
   - Appends both to history
   - Prints the response
   - Repeats

NOTE: We use the new google-genai SDK (not google-generativeai).
google-generativeai is deprecated — google-genai is the replacement.
The concepts are identical, only the syntax changed slightly.
"""

from google import genai                        # New SDK
from google.genai import types                  # For config objects
import os
import sys

# ── Import your config values ──────────────────────────────
from config import MODEL, API_KEY, MAX_TOKENS, TEMPERATURE


# ──────────────────────────────────────────────────────────
# STEP 1: Load the system prompt
# ──────────────────────────────────────────────────────────

def load_system_prompt(filepath="system_prompt.txt"):
    """
    Reads system_prompt.txt and returns its contents as a string.
    If the file is missing, the agent cannot run.
    """
    if not os.path.exists(filepath):
        print(f"[ERROR] system_prompt.txt not found at: {filepath}")
        print("MrEdu cannot run without a system prompt.")
        sys.exit(1)
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


# ──────────────────────────────────────────────────────────
# STEP 2: Initialise the Gemini client
#
# genai.Client() creates the client using your API key.
# The system prompt and generation config are passed
# at the time of each API call, not at client creation.
# ──────────────────────────────────────────────────────────

def create_client():
    """
    Creates and returns a Gemini client instance.
    Reads GEMINI_API_KEY from your environment variable.
    """
    if not API_KEY:
        print("[ERROR] GEMINI_API_KEY environment variable is not set.")
        print("Run: set GEMINI_API_KEY=your-key-here")
        sys.exit(1)

    client = genai.Client(api_key=API_KEY)
    return client


# ──────────────────────────────────────────────────────────
# STEP 3: Send a message and get a response
#
# In the new SDK, client.chats.create() starts a chat session.
# The system prompt is passed as a config object.
# chat.send_message() handles history automatically.
#
# response.text = the model's reply as a plain string.
# ──────────────────────────────────────────────────────────

def create_chat(client, system_prompt):
    """
    Creates a chat session with the system prompt baked in.
    Returns the chat object which manages conversation history.
    """
    chat = client.chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,   # MrEdu's identity + knowledge
            max_output_tokens=MAX_TOKENS,        # Max tokens per response
            temperature=TEMPERATURE,             # Randomness control
        )
    )
    return chat


def get_response(chat, user_input):
    """
    Sends user input to the chat session.
    The session internally manages the full conversation history.
    Returns MrEdu's reply as a plain string.
    """
    response = chat.send_message(user_input)
    return response.text


# ──────────────────────────────────────────────────────────
# STEP 4: The conversation loop
#
# Runs forever until the user types exit or quit.
# The chat session internally maintains the messages array —
# same concept as manually managing it with Anthropic's SDK,
# just abstracted by the SDK here.
# ──────────────────────────────────────────────────────────

def run_conversation(client, system_prompt):
    """
    Runs the main conversation loop.
    """
    chat = create_chat(client, system_prompt)

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

        try:
            print("\nMrEdu: ", end="", flush=True)
            response_text = get_response(chat, user_input)
            print(response_text)

        except Exception as e:
            error = str(e)
            if "api_key" in error.lower() or "authentication" in error.lower():
                print("[ERROR] Invalid API key. Check your GEMINI_API_KEY.")
                sys.exit(1)
            elif "quota" in error.lower() or "rate" in error.lower():
                print("[ERROR] Rate limit hit. Wait a moment and try again.")
                continue
            elif "connect" in error.lower() or "network" in error.lower():
                print("[ERROR] Connection failed. Check your internet.")
                continue
            else:
                print(f"[ERROR] {error}")
                continue

        print()


# ──────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    system_prompt = load_system_prompt()
    client = create_client()
    run_conversation(client, system_prompt)
