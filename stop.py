#!/usr/bin/env python3
"""
CardioSense Stopper Entry Point
Allows typing:
    python stop
    python stop.py
    python3 stop
    ./stop
"""
import sys
import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "start.py"
    # Pass --stop argument
    sys.argv = [str(target), "--stop"]
    runpy.run_path(str(target), run_name="__main__")
