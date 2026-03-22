from google import genai
from google.genai import types
import os

api_key = os.environ.get('GEMINI_API_KEY')
client = genai.Client(api_key=api_key)

models_to_try = [
    'gemini-2.0-flash-lite',
    'gemini-2.0-flash-lite-001',
    'gemini-flash-latest',
    'gemini-flash-lite-latest',
    'gemma-3-4b-it',
]

for model in models_to_try:
    try:
        print(f"Trying {model}...")
        chat = client.chats.create(
            model=model,
            config=types.GenerateContentConfig(
                system_instruction='You are a helpful assistant.',
                max_output_tokens=50,
                temperature=0.7,
            )
        )
        response = chat.send_message('say hello in one word')
        print(f"SUCCESS with {model}: {response.text}")
        break
    except Exception as e:
        print(f"FAILED: {str(e)[:80]}\n")