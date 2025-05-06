# src/face_compare_app/server/routes/compare.py
import logging
import time
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends # Added Depends

# Assuming models are in server.models
from ..models import CompareResponse
from ..dependencies import get_initialized_processor_http # <-- CHANGE THIS IMPORT
# Import necessary functions and exceptions from your core module
from ... import core as core_func
from ...exceptions import (
    FaceCompareError, ImageLoadError, NoFaceFoundError,
    MultipleFacesFoundError, ModelError, EmbeddingError
)
# Import the FaceProcessor class for type hinting dependency
from ...core import FaceProcessor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Compare"])

@router.post("/compare", response_model=CompareResponse)
async def api_compare_faces(
    image1: UploadFile = File(..., description="First image file for comparison."),
    image2: UploadFile = File(..., description="Second image file for comparison."),
    threshold: Optional[float] = Form(None, description="Similarity threshold (0.0 to 1.0). If provided, 'is_match' is calculated."),
    # processor: FaceProcessor = Depends(get_face_processor) # Use Dependency Injection
    # OR access global instance (simpler for now, but less robust):
    # Note: We don't directly use the processor instance here because compare_faces uses the global one.
    # We just call ..() to ensure it's loaded/initialized before proceeding.
    _processor_check: FaceProcessor = Depends(get_initialized_processor_http) # Ensures processor is ready
):
    """
    Compares faces in two uploaded images.
    Requires exactly one face per image.
    """
    start_time = time.time()
    logger.info(f"Received compare request. Image1: {image1.filename}, Image2: {image2.filename}, Threshold: {threshold}")

    # Validate threshold if provided
    if threshold is not None and not (0.0 <= threshold <= 1.0):
        logger.error(f"Invalid threshold received: {threshold}")
        raise HTTPException(status_code=422, detail="Invalid threshold value. Must be between 0.0 and 1.0.")

    # Create temporary files to store uploaded images
    # Using NamedTemporaryFile ensures they have paths accessible by core_func
    # Suffix helps OpenCV identify file type sometimes, though not strictly necessary if core loads correctly
    suffix1 = Path(image1.filename or ".tmp").suffix
    suffix2 = Path(image2.filename or ".tmp").suffix

    temp_file1_path: Optional[str] = None
    temp_file2_path: Optional[str] = None
    similarity: Optional[float] = None
    error_detail: Optional[str] = None
    status_code: int = 200

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix1) as temp_file1, \
            tempfile.NamedTemporaryFile(delete=False, suffix=suffix2) as temp_file2:

            temp_file1_path = temp_file1.name
            temp_file2_path = temp_file2.name
            logger.debug(f"Saving uploaded files to temporary paths: {temp_file1_path}, {temp_file2_path}")

            # Efficiently copy file contents using shutil
            try:
                # Reset file pointers of UploadFile before copying
                await image1.seek(0)
                await image2.seek(0)
                with open(temp_file1.name, 'wb') as f_dst1:
                    shutil.copyfileobj(image1.file, f_dst1)
                with open(temp_file2.name, 'wb') as f_dst2:
                    shutil.copyfileobj(image2.file, f_dst2)
            finally:
                # Ensure UploadFile internal pointers are closed
                # (FastAPI might handle this, but doesn't hurt to be explicit)
                image1.file.close()
                image2.file.close()

            # --- Call the core comparison function ---
            # This function now re-raises specific exceptions
            similarity = core_func.compare_faces(temp_file1_path, temp_file2_path)
            # If we reach here, comparison was successful

    # --- Handle Specific Application Errors ---
    except FileNotFoundError as e: # Should not happen with temp files unless deleted prematurely
        logger.error(f"Internal Error: Temporary file likely deleted prematurely: {e}", exc_info=True)
        status_code = 500
        error_detail = "Internal server error: Could not process temporary image file."
    except ImageLoadError as e:
        logger.warning(f"Image loading failed: {e.message} (Code: {e.code})")
        status_code = 422 # Unprocessable Entity - image format/content issue
        # Be careful not to expose too much internal path info in error messages
        error_detail = f"Failed to load one or both images. Ensure they are valid image files. ({e.code})"
    except NoFaceFoundError as e:
        logger.info(f"Comparison failed: No face found. Details: {e.message} (Code: {e.code})")
        status_code = 400 # Bad Request - input image lacks required feature
        error_detail = f"No face detected in one or both images. ({e.code})"
    except MultipleFacesFoundError as e:
        logger.info(f"Comparison failed: Multiple faces found. Details: {e.message} (Code: {e.code})")
        status_code = 400 # Bad Request - input image doesn't meet criteria
        error_detail = f"Multiple faces detected in one or both images; expected exactly one. ({e.code})"
    except ModelError as e:
        logger.error(f"Model error during comparison: {e.message} (Code: {e.code})", exc_info=True)
        status_code = 503 # Service Unavailable - underlying model issue
        error_detail = f"Face processing model error occurred. Please try again later. ({e.code})"
    except EmbeddingError as e:
        logger.error(f"Embedding error during comparison: {e.message} (Code: {e.code})", exc_info=True)
        status_code = 500 # Internal Server Error - failure during processing step
        error_detail = f"Failed to generate face features for comparison. ({e.code})"
    except FaceCompareError as e: # Catch other specific comparison errors
        logger.error(f"Face comparison error: {e.message} (Code: {e.code})", exc_info=True)
        status_code = 500
        error_detail = f"An error occurred during face comparison. ({e.code})"
    except TypeError as e: # Catch if processor wasn't available
        logger.error(f"Type error (likely processor unavailable): {e}", exc_info=True)
        status_code = 503
        error_detail = "Face processing service is unavailable or not initialized."
    except HTTPException:
        # Re-raise HTTPExceptions raised by dependencies (like ..)
        raise
    except Exception as e:
        # Catch any other unexpected errors during file handling or core call
        logger.error(f"Unexpected error during comparison API call: {e}", exc_info=True)
        status_code = 500
        error_detail = "An unexpected internal server error occurred."
    finally:
        # --- Clean up temporary files ---
        for temp_path in [temp_file1_path, temp_file2_path]:
            if temp_path and Path(temp_path).exists():
                try:
                    Path(temp_path).unlink()
                    logger.debug(f"Deleted temporary file: {temp_path}")
                except OSError as unlink_err:
                    # Log error but don't prevent response
                    logger.error(f"Error deleting temporary file {temp_path}: {unlink_err}")

    # --- Prepare and return response ---
    if status_code != 200 or similarity is None:
        # If an error occurred, raise the corresponding HTTPException
        raise HTTPException(status_code=status_code, detail=error_detail)
    else:
        # Success case
        is_match = None
        if threshold is not None:
            is_match = similarity >= threshold

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Compare request successful. Similarity: {similarity:.4f}, Match: {is_match}, Elapsed: {elapsed_ms}ms")

        return CompareResponse(
            similarity=similarity,
            is_match=is_match,
            elapsed_ms=elapsed_ms
        )