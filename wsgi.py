# ============================================================
# PythonAnywhere WSGI configuration file
# ============================================================
# Place this file at the path shown in your PythonAnywhere
# Web tab → WSGI configuration file (e.g. /var/www/yourusername_pythonanywhere_com_wsgi.py)
#
# Then in the Web tab:
#   Source code:   /home/yourusername/mysite
#   Virtualenv:    /home/yourusername/.virtualenvs/framevault
# ============================================================

import sys
import os

# Add your project folder to the path
project_home = '/home/yourusername/mysite'  # ← change "yourusername"
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(project_home, '.env'))
except ImportError:
    pass

# Import the FastAPI app and wrap it for WSGI
from main import app

# PythonAnywhere uses WSGI — wrap FastAPI (ASGI) with a2wsgi
try:
    from a2wsgi import ASGIMiddleware
    application = ASGIMiddleware(app)
except ImportError:
    # Fallback: install a2wsgi → pip install a2wsgi
    raise ImportError("Run: pip install a2wsgi")
