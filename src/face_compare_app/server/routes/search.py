# src/face_compare_app/server/routes/search.py
import logging
import time
import json
import tempfile
import shutil
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends

# Assuming models are in server.models
from ..models import SearchResponse, SearchResultItem
# Import necessary functions and exceptions
from ... import core as core_func
from ... import database as db_func
from ...exceptions import (
    FaceCompareError, ImageLoadError, NoFaceFoundError,
    MultipleFacesFoundError, ModelError, EmbeddingError, DatabaseError,
    InvalidInputError # For feature size errors etc.
)
# Import the FaceProcessor class and its static method
from ...core import FaceProcessor
# Import the dependency function to ensure processor is ready
# from ..dependencies import get_initialized_processor # Re-use from compare route
# Import the dependency function for DB path
from ..dependencies import get_database_path # Import the database path dependency

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Search"])

# --- Configuration / Constants ---
# TODO: Get embedding dimension from the model itself if possible
EXPECTED_EMBEDDING_DIM = 512

@router.post("/search", response_model=SearchResponse)
async def api_search_faces(
    image: UploadFile = File(..., description="Image file containing the query face."),
    top_k: int = Form(3, description="Maximum number of similar faces to return.", gt=0),
    threshold: Optional[float] = Form(None, description="Optional similarity threshold (0.0 to 1.0). Only results above this threshold are returned."),
    # Ensure processor is ready
    # processor: FaceProcessor = Depends(get_initialized_processor),
    # Get the database path
    db_path: Path = Depends(get_database_path)
):
    """
    Searches the database for faces similar to the face in the uploaded query image.
    Expects exactly one face in the query image.
    """
    start_time = time.time()
    logger.info(f"Received search request. Image: {image.filename}, Top_K: {top_k}, Threshold: {threshold}, DB: {db_path}")

    # Validate threshold
    if threshold is not None and not (0.0 <= threshold <= 1.0):
        logger.error(f"Invalid threshold received: {threshold}")
        raise HTTPException(status_code=422, detail="Invalid threshold value. Must be between 0.0 and 1.0.")

    temp_file_path: Optional[str] = None
    query_features_bytes: Optional[bytes] = None
    db_records: List[Tuple[str, str, bytes, Optional[str]]] = []
    results: List[SearchResultItem] = []
    error_detail: Optional[str] = None
    status_code: int = 200

    try:
        # 1. Save query image temporarily and extract features
        suffix = Path(image.filename or ".tmp").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file_path = temp_file.name
            logger.debug(f"Saving query image to temporary path: {temp_file_path}")
            try:
                await image.seek(0)
                shutil.copyfileobj(image.file, temp_file)
            finally:
                image.file.close()

            # Extract features (handles single face check)
            query_features_bytes = core_func.extract_features(temp_file_path) # Raises specific errors on failure
            logger.info(f"Successfully extracted features from query image ({len(query_features_bytes)} bytes).")

        # 2. Load face data from the database
        logger.info(f"Loading face database from: {db_path}")
        db_records = db_func.get_all_faces_from_db(db_path)
        if not db_records:
            logger.warning(f"Database '{db_path}' is empty. Search will return no results.")
            # Not an error, just return empty results
            elapsed_ms = int((time.time() - start_time) * 1000)
            return SearchResponse(results=[], search_time_ms=elapsed_ms)

        # 3. Perform the search if features extracted and DB has records
        if query_features_bytes and db_records:
            # Deserialize query features
            query_embedding = np.frombuffer(query_features_bytes, dtype=np.float32)
            if query_embedding.size != EXPECTED_EMBEDDING_DIM:
                logger.error(f"Query feature size mismatch. Expected {EXPECTED_EMBEDDING_DIM}, got {query_embedding.size}.")
                raise InvalidInputError(f"Invalid query feature size ({query_embedding.size}), expected {EXPECTED_EMBEDDING_DIM}.")

            all_matches: List[Tuple[str, str, float, Optional[str]]] = [] # id, name, similarity, meta_str

            logger.info(f"Comparing query face against {len(db_records)} entries in the database...")
            search_loop_start = time.time()
            for db_id, db_name, db_features_bytes, db_metadata_str in db_records:
                try:
                    db_embedding = np.frombuffer(db_features_bytes, dtype=np.float32)
                    if db_embedding.size != EXPECTED_EMBEDDING_DIM:
                        logger.warning(f"Skipping DB entry ID '{db_id}': Feature size mismatch (got {db_embedding.size}, expected {EXPECTED_EMBEDDING_DIM}).")
                        continue

                    # Calculate similarity using the static method from FaceProcessor
                    similarity = FaceProcessor._cosine_similarity(query_embedding, db_embedding)

                    # Apply threshold if provided
                    if threshold is None or similarity >= threshold:
                        all_matches.append((db_id, db_name, float(similarity), db_metadata_str))

                except Exception as entry_err:
                    logger.error(f"Error processing database entry ID '{db_id}' during search: {entry_err}", exc_info=False)
                    continue # Skip problematic entry

            logger.info(f"Similarity calculations finished in {time.time() - search_loop_start:.3f}s. Found {len(all_matches)} potential matches.")

            # Sort matches by similarity (descending)
            all_matches.sort(key=lambda x: x[2], reverse=True)

            # Take top K results
            top_matches = all_matches[:top_k]
            logger.info(f"Selected top {len(top_matches)} matches (max K={top_k}).")

            # 4. Format results (including metadata parsing)
            for match_id, match_name, match_sim, match_meta_str in top_matches:
                meta_dict: Optional[Dict[str, Any]] = None
                if match_meta_str:
                    try:
                        meta_dict = json.loads(match_meta_str)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse metadata JSON for matched ID '{match_id}': {match_meta_str}")
                        # Optionally include the raw string or an error marker in meta? For now, just None.
                        meta_dict = {"_parse_error": "Invalid JSON in database"}


                results.append(SearchResultItem(
                    face_id=match_id,
                    name=match_name,
                    similarity=match_sim,
                    meta=meta_dict
                ))

    # --- Handle Specific Application Errors ---
    except FileNotFoundError as e:
        logger.error(f"Internal Error: Failed to access temporary file: {e}", exc_info=True)
        status_code = 500
        error_detail = "Internal server error: Could not process temporary image file."
    except (ImageLoadError, NoFaceFoundError, MultipleFacesFoundError, ModelError, EmbeddingError, InvalidInputError) as e:
        # Errors specifically related to processing the *query* image or its features
        error_prefix = "Error processing query image"
        if isinstance(e, NoFaceFoundError):
            status_code = 400
            error_prefix = "No face found in query image"
        elif isinstance(e, MultipleFacesFoundError):
            status_code = 400
            error_prefix = "Multiple faces found in query image"
        elif isinstance(e, ImageLoadError):
            status_code = 422
            error_prefix = "Failed to load query image"
        elif isinstance(e, InvalidInputError): # e.g., feature size mismatch
            status_code = 422 # or 500 if it's unexpected internal state
            error_prefix = "Invalid input features from query image"
        elif isinstance(e, ModelError):
            status_code = 503
            error_prefix = "Model error processing query image"
        else: # EmbeddingError, other FaceCompareError
            status_code = 500
            error_prefix = "Error generating features for query image"

        logger.warning(f"{error_prefix}: {e.message} (Code: {e.code})")
        error_detail = f"{error_prefix}. ({e.code})"
    except DatabaseError as e:
        # Error loading the database itself
        logger.error(f"Database error during search prep: {e.message} (Code: {e.code})", exc_info=True)
        status_code = 500 # Treat DB errors as internal server errors
        error_detail = f"Database operation failed during search. ({e.code})"
    except FaceCompareError as e: # Catch other specific app errors during query processing
        logger.error(f"Face processing error during query: {e.message} (Code: {e.code})", exc_info=True)
        status_code = 500
        error_detail = f"An error occurred during query face processing. ({e.code})"
    except TypeError as e: # Catch if processor wasn't available
        logger.error(f"Type error (likely processor unavailable): {e}", exc_info=True)
        status_code = 503
        error_detail = "Face processing service is unavailable or not initialized."
    except HTTPException:
        # Re-raise HTTPExceptions raised by dependencies
        raise
    except Exception as e:
        # Catch any other unexpected errors (e.g., during search loop)
        logger.error(f"Unexpected error during search API call: {e}", exc_info=True)
        status_code = 500
        error_detail = "An unexpected internal server error occurred."
    finally:
        # --- Clean up temporary query image file ---
        if temp_file_path and Path(temp_file_path).exists():
            try:
                Path(temp_file_path).unlink()
                logger.debug(f"Deleted temporary query file: {temp_file_path}")
            except OSError as unlink_err:
                logger.error(f"Error deleting temporary query file {temp_file_path}: {unlink_err}")

    # --- Prepare and return response ---
    if status_code != 200:
        # If a critical error occurred before results could be generated
        raise HTTPException(status_code=status_code, detail=error_detail)
    else:
        # Success case (even if results list is empty)
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Search request successful. Found {len(results)} matches meeting criteria. Elapsed: {elapsed_ms}ms")
        return SearchResponse(
            results=results,
            search_time_ms=elapsed_ms
        )