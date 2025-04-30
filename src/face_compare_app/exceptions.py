# src/face_compare_app/exceptions.py
"""Custom exceptions for the application."""

class FaceCompareError(Exception):
    """Base exception for face comparison errors."""
    pass

class ImageLoadError(FaceCompareError):
    """Error loading or processing an image."""
    pass

class NoFaceFoundError(FaceCompareError):
    """Error when no face is detected in an image."""
    pass

class DatabaseError(FaceCompareError):
    """Error related to database operations."""
    pass

class InvalidInputError(FaceCompareError):
    """Error for invalid user input (e.g., bad JSON)."""
    pass