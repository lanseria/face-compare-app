# src/face_compare_app/server/routes/faces.py
import logging
import time
import json
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends

# Assuming models are in server.models
from ..models import FaceInsertResponse
# Import necessary functions and exceptions
from ... import core as core_func
from ... import database as db_func
from ...exceptions import (
    FaceCompareError, ImageLoadError, NoFaceFoundError,
    MultipleFacesFoundError, ModelError, EmbeddingError, DatabaseError,
    InvalidInputError # Added for metadata parsing errors in utils
)
# Import the FaceProcessor class for type hinting dependency
from ...core import FaceProcessor
from ..dependencies import get_database_path, get_initialized_processor_http # Import the database path dependency
# Import the utility for metadata parsing (though we do it inline here too)
from ... import utils

# Import the dependency function to ensure processor is ready

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Faces"])


@router.post("/faces", response_model=FaceInsertResponse)
async def api_insert_face(
    image: UploadFile = File(..., description="Image file containing the face to insert."),
    id: str = Form(..., description="Unique ID for this face/person."),
    name: Optional[str] = Form(None, description="Optional name associated with the face."),
    meta: Optional[str] = Form(None, description="Optional metadata as a JSON string (e.g., '{\"dept\": \"IT\"}')."),
    # Ensure processor is ready before proceeding
    # _processor_check: FaceProcessor = Depends(get_initialized_processor),
    # Get the database path (using the simple function above for now)
    db_path: Path = Depends(get_database_path)
):
    """
    Extracts features from an uploaded image and stores face information.
    Expects exactly one face in the image.
    """
    start_time = time.time()
    logger.info(f"Received insert face request. ID: {id}, Name: {name}, Image: {image.filename}, Meta: {meta}, DB: {db_path}")

    # 1. Parse metadata JSON string (optional)
    metadata_dict = None
    if meta:
        try:
            metadata_dict = json.loads(meta)
            if not isinstance(metadata_dict, dict):
                raise ValueError("Metadata must be a valid JSON object (dictionary).")
            logger.debug(f"Parsed metadata for ID {id}: {metadata_dict}")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Invalid JSON metadata provided for ID {id}: {meta} - Error: {e}")
            # Use 422 for validation errors
            raise HTTPException(status_code=422, detail=f"Invalid JSON metadata: {e}")

    # 2. Save image temporarily and extract features
    temp_file_path: Optional[str] = None
    features_bytes: Optional[bytes] = None
    error_detail: Optional[str] = None
    status_code: int = 200

    try:
        # Create a temporary file
        suffix = Path(image.filename or ".tmp").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file_path = temp_file.name
            logger.debug(f"Saving uploaded image for ID '{id}' to temporary path: {temp_file_path}")

            # Copy content
            try:
                await image.seek(0)
                shutil.copyfileobj(image.file, temp_file)
            finally:
                image.file.close()

            # --- Extract Features using Core function ---
            # This function handles loading, single face check, and embedding
            logger.info(f"Extracting features from temporary file: {temp_file_path}")
            features_bytes = core_func.extract_features(temp_file_path) # Expects str path
            logger.info(f"Successfully extracted features ({len(features_bytes)} bytes) for ID '{id}'")

        # If feature extraction successful, proceed to database insertion
        if features_bytes:
            logger.info(f"Adding face ID '{id}' to database: {db_path}")
            # --- Add to Database ---
            # This function handles DB connection, init, and insert/replace
            db_func.add_face_to_db(
                db_path=db_path, # Pass Path object
                user_id=id,
                name=name or f"Unnamed_{id}", # Provide a default name if None
                features=features_bytes,
                metadata=metadata_dict # Pass parsed dict
            )
            logger.info(f"Successfully added/updated face ID '{id}' in database.")

    # --- Handle Specific Application Errors ---
    except FileNotFoundError as e: # Should only happen if temp file fails creation/access badly
        logger.error(f"Internal Error: Failed to access temporary file: {e}", exc_info=True)
        status_code = 500
        error_detail = "Internal server error: Could not process temporary image file."
    except ImageLoadError as e:
        logger.warning(f"Image loading failed for ID '{id}': {e.message} (Code: {e.code})")
        status_code = 422 # Image format/content issue
        error_detail = f"Failed to load image for ID '{id}'. Ensure it's a valid image file. ({e.code})"
    except NoFaceFoundError as e:
        logger.info(f"Insertion failed for ID '{id}': No face found. Details: {e.message} (Code: {e.code})")
        status_code = 400 # Input image lacks required feature
        error_detail = f"No face detected in the provided image for ID '{id}'. ({e.code})"
    except MultipleFacesFoundError as e:
        logger.info(f"Insertion failed for ID '{id}': Multiple faces found. Details: {e.message} (Code: {e.code})")
        status_code = 400 # Input image doesn't meet criteria
        error_detail = f"Multiple faces detected in the image for ID '{id}'; expected exactly one. ({e.code})"
    except ModelError as e:
        logger.error(f"Model error during feature extraction for ID '{id}': {e.message} (Code: {e.code})", exc_info=True)
        status_code = 503 # Service Unavailable - underlying model issue
        error_detail = f"Face processing model error occurred. Please try again later. ({e.code})"
    except EmbeddingError as e:
        logger.error(f"Embedding error during feature extraction for ID '{id}': {e.message} (Code: {e.code})", exc_info=True)
        status_code = 500 # Internal Server Error - failure during processing step
        error_detail = f"Failed to generate face features for ID '{id}'. ({e.code})"
    except DatabaseError as e:
        logger.error(f"Database error during insertion for ID '{id}': {e.message} (Code: {e.code})", exc_info=True)
        status_code = 500 # Treat DB errors as internal server errors
        error_detail = f"Database operation failed for ID '{id}'. ({e.code})"
    except InvalidInputError as e: # Catch metadata validation errors if utils.parse_metadata was used
        logger.error(f"Invalid input error (likely metadata) for ID '{id}': {e.message}", exc_info=False)
        status_code = 422
        error_detail = f"Invalid input: {e.message} ({e.code})"
    except FaceCompareError as e: # Catch other specific app errors
        logger.error(f"Face processing error for ID '{id}': {e.message} (Code: {e.code})", exc_info=True)
        status_code = 500
        error_detail = f"An error occurred during face processing for ID '{id}'. ({e.code})"
    except TypeError as e: # Catch if processor wasn't available
        logger.error(f"Type error (likely processor unavailable) for ID '{id}': {e}", exc_info=True)
        status_code = 503
        error_detail = "Face processing service is unavailable or not initialized."
    except HTTPException:
        # Re-raise HTTPExceptions raised by dependencies
        raise
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error during face insertion for ID '{id}': {e}", exc_info=True)
        status_code = 500
        error_detail = "An unexpected internal server error occurred."
    finally:
        # --- Clean up temporary file ---
        if temp_file_path and Path(temp_file_path).exists():
            try:
                Path(temp_file_path).unlink()
                logger.debug(f"Deleted temporary file: {temp_file_path}")
            except OSError as unlink_err:
                logger.error(f"Error deleting temporary file {temp_file_path}: {unlink_err}")

    # --- Prepare and return response ---
    if status_code != 200 or features_bytes is None:
        # If an error occurred or features couldn't be extracted
        raise HTTPException(status_code=status_code, detail=error_detail or "An unknown error occurred during processing.")
    else:
        # Success case
        elapsed_ms = int((time.time() - start_time) * 1000)

        # Calculate feature size (assuming float32 embeddings)
        # TODO: Get this more robustly from the model/processor if possible
        feature_size = len(features_bytes) // 4 if len(features_bytes) % 4 == 0 else None
        if feature_size is None:
            logger.warning(f"Could not determine feature size from byte length ({len(features_bytes)}) for ID '{id}'.")


        logger.info(f"Insert face request successful for ID: {id}. Elapsed: {elapsed_ms}ms")
        return FaceInsertResponse(
            face_id=id,
            feature_size=feature_size, # Can be None if calculation failed
            message=f"Successfully inserted/updated face data for ID '{id}'."
        )