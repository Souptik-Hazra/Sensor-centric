"""
EquiTraffic-GPT Master Single-Server Launcher (backend.py)

Single-command entry point to launch both FastAPI REST APIs and built React Web GIS:
  $ python backend.py
"""
import os
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from EquiTrafficAI.backend.backend import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
