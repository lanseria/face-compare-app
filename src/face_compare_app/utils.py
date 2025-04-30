# src/face_compare_app/utils.py
"""Utility functions."""
import json
import logging
from typing import Any, Optional, Dict
from pathlib import Path
import cv2 # Import OpenCV
import numpy as np # Import Numpy

# Import custom exceptions relative to the package root
from .exceptions import InvalidInputError, ImageLoadError

logger = logging.getLogger(__name__)

def load_image(image_path: Path) -> np.ndarray:
    """
    Loads an image from the specified path using OpenCV.

    Args:
        image_path: Path object pointing to the image file.

    Returns:
        A NumPy array representing the image in BGR format.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ImageLoadError: If the image file cannot be loaded or read by OpenCV.
    """
    absolute_path_str = str(image_path.resolve())
    logger.debug(f"Attempting to load image: {absolute_path_str}")

    if not image_path.is_file():
        logger.error(f"Image file not found at: {absolute_path_str}")
        # Raise FileNotFoundError which is often handled specifically in CLI
        raise FileNotFoundError(f"No such file or directory: '{absolute_path_str}'")

    try:
        # Read the image using OpenCV
        img = cv2.imread(absolute_path_str)

        if img is None:
            # This can happen for various reasons (corrupt file, unsupported format)
            logger.error(f"Failed to load image using OpenCV (cv2.imread returned None): {absolute_path_str}")
            raise ImageLoadError(f"Could not load image file: {absolute_path_str}")

        logger.debug(f"Successfully loaded image {absolute_path_str} with shape {img.shape}")
        return img
    except Exception as e:
        # Catch any other unexpected errors during loading
        logger.error(f"An unexpected error occurred while loading image {absolute_path_str}: {e}", exc_info=True)
        # Wrap the original exception for context
        raise ImageLoadError(f"Failed to load image {absolute_path_str}: {e}") from e


def parse_metadata(meta_str: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parses a JSON string into a dictionary."""
    if meta_str is None:
        return None
    try:
        metadata = json.loads(meta_str)
        if not isinstance(metadata, dict):
            raise InvalidInputError("Metadata must be a valid JSON object (dictionary).")
        logger.debug(f"Parsed metadata: {metadata}")
        return metadata
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format for metadata: {meta_str}")
        raise InvalidInputError(f"Invalid JSON metadata: {e}") from e