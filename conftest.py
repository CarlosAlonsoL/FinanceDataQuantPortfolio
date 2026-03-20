# conftest.py
import sys
from pathlib import Path

# Make scripts/ importable for tests
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
