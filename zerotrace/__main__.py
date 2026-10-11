#!/usr/bin/env python3
"""
ZeroTrace AI - Universal Command Line Entrypoint.
Allows running ZeroTrace on any operating system (Windows, Linux, macOS) via:
    python -m zerotrace [chat | live | info | compress]
or simply:
    python -m zerotrace
"""

import sys
import os

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import cli

if __name__ == "__main__":
    cli.main()
