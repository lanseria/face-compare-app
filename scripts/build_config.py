"""Configuration for building executable."""
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files
# Build Paths
ROOT_DIR = Path(__file__).parent.parent