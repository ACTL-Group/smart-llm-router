import sys
from pathlib import Path

# Add src/ directory to sys.path for test runner
src_dir = str(Path(__file__).resolve().parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
