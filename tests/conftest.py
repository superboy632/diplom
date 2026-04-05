"""
conftest.py — Make the project root importable from the tests/ directory.
"""
import sys
from pathlib import Path

# Insert the repo root (parent of tests/) at the front of the module search path
# so that `import geo_utils` resolves to /home/romeo/diplom/geo_utils.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
