"""Hello-world check for the LLM setup (1 request).

    uv run python scripts/check_llm.py                      # model from .env
    uv run python scripts/check_llm.py gemini-3.8-flash     # try another model
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

if len(sys.argv) > 1:
    os.environ["LLM_MODEL"] = sys.argv[1]  # set before src.graph reads it

from src.graph import get_llm, text_of  # noqa: E402

print("model:", os.getenv("LLM_MODEL", "gemini-4-hydrogen"))
print("reply:", text_of(get_llm(0).invoke("Say hello world")).strip())
