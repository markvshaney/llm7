# conftest.py
import os
import sys
from pathlib import Path

def pytest_configure():
    # Get the project root directory
    project_root = Path(__file__).parent

    # Add both the root and src directories to Python path
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))