"""Quick check that .env loads regardless of the working directory."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app.config as c

print("Running from cwd:", os.getcwd())
print("  APP_MODE             :", c.APP_MODE)
print("  PINECONE_API_KEY set :", bool(c.PINECONE_API_KEY))
print("  LLM_API_KEY set      :", bool(c.LLM_API_KEY))
print("  LLM_MODEL            :", c.LLM_MODEL)
print("  ANTHROPIC_WORKSPACE_ID set:", bool(c.ANTHROPIC_WORKSPACE_ID))
