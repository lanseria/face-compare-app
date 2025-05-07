# src/face_compare_app/server/routes/live_utils.py
import logging
import time
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from ...core import FaceProcessor # For type hint
from ...exceptions import ModelError # For catching specific error
from ... import database as db_func # For LiveSearchDBData.load()

logger = logging.getLogger(__name__)

EXPECTED_EMBEDDING_DIM = 512 # Keep consistent

async def process_frame_common(
    frame_bytes: bytes,
    processor: FaceProcessor
) -> Tuple[Optional[np.ndarray], Optional[List[Any]], Optional[str]]:
    """
    Common frame decoding and face detection logic.
    Returns insightface face objects.
    """
    try:
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return None, None, "Failed to decode image frame."

        # --- Time the critical call ---
        inference_start_time = time.perf_counter()
        # Call processor.get_faces() only ONCE
        detected_faces_insight = processor.get_faces(frame)
        inference_duration = (time.perf_counter() - inference_start_time) * 1000
        logger.debug(f"Face detection/embedding for frame took: {inference_duration:.2f} ms, Found: {len(detected_faces_insight)} faces")
        
        return frame, detected_faces_insight, None # Return insightface's face objects

    except ModelError as e:
        logger.error(f"Model error during live frame processing: {e}", exc_info=False)
        return None, None, f"Model processing error: {e.message}"
    except Exception as e:
        logger.error(f"Unexpected error decoding/processing frame: {e}", exc_info=True)
        return None, None, "Internal server error processing frame."


class LiveSearchDBData:
    def __init__(self):
        self.embeddings: List[Tuple[str, str, np.ndarray, Optional[Dict[str, Any]]]] = []
        self.is_loaded = False
        self.load_time = 0.0

    def load(self, db_path: Path):
        start_time = time.time()
        logger.info(f"LiveSearchDBData: Loading database from {db_path} into memory...")
        self.embeddings = [] # Reset before loading
        try:
            db_records_raw = db_func.get_all_faces_from_db(db_path)
            if not db_records_raw:
                logger.warning(f"LiveSearchDBData: Database '{db_path}' is empty.")
                self.is_loaded = True
                self.load_time = time.time() - start_time
                return

            for rec_id, rec_name, rec_features_bytes, rec_meta_str in db_records_raw:
                try:
                    embedding_np = np.frombuffer(rec_features_bytes, dtype=np.float32)
                    if embedding_np.size != EXPECTED_EMBEDDING_DIM:
                        logger.warning(f"LiveSearchDBData: Skipping DB entry ID '{rec_id}': Feature size mismatch (got {embedding_np.size}, expected {EXPECTED_EMBEDDING_DIM}).")
                        continue
                    meta_dict: Optional[Dict[str, Any]] = None
                    if rec_meta_str:
                        try:
                            meta_dict = json.loads(rec_meta_str)
                        except json.JSONDecodeError:
                            logger.warning(f"LiveSearchDBData: Failed to parse metadata JSON for DB ID '{rec_id}'.")
                            meta_dict = {"_parse_error": "Invalid JSON in database"}
                    self.embeddings.append((rec_id, rec_name, embedding_np, meta_dict))
                except Exception as e:
                    logger.error(f"LiveSearchDBData: Error processing DB entry ID '{rec_id}': {e}", exc_info=False)
            
            self.is_loaded = True
            self.load_time = time.time() - start_time
            logger.info(f"LiveSearchDBData: Loaded {len(self.embeddings)} valid face records in {self.load_time:.3f}s.")
        
        except Exception as e: # Catch DatabaseError, sqlite3.Error, etc.
            logger.error(f"LiveSearchDBData: Failed to load database '{db_path}': {e}", exc_info=True)
            self.is_loaded = False # Ensure state reflects failure
            raise # Re-raise the caught exception to be handled by the caller

# --- Create the shared instance HERE in live_utils.py ---
live_search_db_instance = LiveSearchDBData()