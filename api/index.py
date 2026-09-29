import os
import sys
from pathlib import Path

# Add project root and backend to sys.path so modules and models can be found
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "backend"))

# Import Flask app
try:
    from backend.app import app
except ImportError:
    from app import app

# Vercel serverless function entrypoint
# The WSGI callable MUST be named 'app'
if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 5001)))
