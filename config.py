"""
config.py — MrEdu v1.0 (Gemini)
All configuration lives here. No logic, just values.
Change model, token limits, or temperature from this one file.
"""

import os

# Gemini model — Flash is free tier
MODEL = "gemini-flash-latest"

# Read API key from environment — never hardcode this
API_KEY = os.environ.get("GEMINI_API_KEY")

# Maximum tokens MrEdu can generate per response
# 1 token ≈ 0.75 words. 1024 handles detailed explanations well.
MAX_TOKENS = 1024

# Controls randomness. 0.0 = deterministic, 1.0 = creative.
# 0.7 is standard for conversational agents.
TEMPERATURE = 0.7
