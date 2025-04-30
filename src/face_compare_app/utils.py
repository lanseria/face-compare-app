# src/face_compare_app/utils.py
"""Utility functions."""
import json
import logging
from typing import Any, Optional, Dict

from .exceptions import InvalidInputError

logger = logging.getLogger(__name__)

def load_image(image_path: str) -> Any:
    """
    Placeholder for loading an image.
    In a real implementation, this would use libraries like OpenCV or Pillow.
    Returns a representation of the image (e.g., numpy array) or raises ImageLoadError.
    """
    logger.debug(f"Attempting to load image: {image_path}")
    # Simulate loading
    try:
        # In real code: Check if file exists and is readable
        # img = cv2.imread(image_path) or Image.open(image_path)
        # if img is None: raise ImageLoadError(...)
        print(f"Placeholder: Loaded image from {image_path}")
        return f"image_data_for_{image_path}" # Return a dummy object
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {e}")
        # raise ImageLoadError(f"Could not load image: {image_path}") from e # In real code
        # For now, just signal failure if needed by returning None or raising generic error
        raise # Re-raise generic exception for now

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