# src/face_compare_app/exceptions.py
from typing import Optional
"""Custom exceptions for the application."""

class FaceCompareError(Exception):
    """Base exception for face comparison errors."""
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}") # Make the default message informative

class ImageLoadError(FaceCompareError):
    """Error loading or processing an image."""
    # You might want specific codes for loading errors too
    def __init__(self, message: str):
        super().__init__(1000, f"Image Loading Error: {message}")

class NoFaceFoundError(FaceCompareError):
    """Error when no face is detected in an image."""
    def __init__(self, message: str = "No face detected in the image."):
        super().__init__(1001, message)

class MultipleFacesFoundError(FaceCompareError):
    """Error when multiple faces are detected but only one was expected."""
    def __init__(self, count: int, message: Optional[str] = None):
        msg = message or f"Multiple faces ({count}) detected; expected only one."
        super().__init__(1002, msg)

class ModelError(FaceCompareError):
    """Errors related to model loading or execution."""
    def __init__(self, message: str):
        super().__init__(1003, f"Model Error: {message}")

class EmbeddingError(FaceCompareError):
    """Errors related to feature embedding generation."""
    def __init__(self, message: str = "Failed to generate embedding for the detected face."):
        super().__init__(1004, message)


class DatabaseError(FaceCompareError):
    """Error related to database operations."""
    # Define specific codes if needed
    def __init__(self, message: str):
        super().__init__(2000, f"Database Error: {message}")


class InvalidInputError(FaceCompareError):
    """Error for invalid user input (e.g., bad JSON)."""
    # Define specific codes if needed
    def __init__(self, message: str):
        super().__init__(3000, f"Invalid Input: {message}")