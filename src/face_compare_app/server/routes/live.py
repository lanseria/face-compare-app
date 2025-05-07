# src/face_compare_app/server/routes/live.py
import logging
import asyncio
import time
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple # Added Tuple

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends # Added Depends

# Assuming models are in server.models
from ..models import LiveCompareWSResponse, LiveSearchWSResponse, LiveSearchMatchDetail
# Import core/db functions and exceptions
from ... import core as core_func
from ... import database as db_func
from ...exceptions import (
    FaceCompareError, ImageLoadError, NoFaceFoundError,
    MultipleFacesFoundError, ModelError, EmbeddingError, DatabaseError,
    InvalidInputError
)
from ...core import FaceProcessor # Import class for type hints and static method
# Import dependency functions
from ..dependencies import get_initialized_processor_ws, get_database_path

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Live"])

# --- Constants ---
# TODO: Make threshold configurable via query param or settings
LIVE_COMPARE_THRESHOLD = 0.55
LIVE_SEARCH_THRESHOLD = 0.55
EXPECTED_EMBEDDING_DIM = 512 # TODO: Get from model if possible
SEND_NO_FACE_UPDATES = False # Control if "no_face" messages are sent
SEND_NO_MATCH_UPDATES = False # Control if "no_match" messages are sent (for live search)
FRAME_PROCESSING_INTERVAL = 1 # Minimum seconds between processing frames (controls rate)


async def process_frame_common(
    frame_bytes: bytes,
    processor: FaceProcessor
) -> Tuple[Optional[np.ndarray], Optional[List[Any]], Optional[str]]:
    """
    Common frame decoding and face detection logic.

    Returns:
        Tuple: (decoded_frame, list_of_face_objects, error_message)
        Returns (None, None, error_message) if decoding fails.
        Returns (frame, faces, None) if successful.
    """
    try:
        # Decode the image bytes received from the client
        nparr = np.frombuffer(frame_bytes, np.uint8)
        # flag cv2.IMREAD_COLOR ensures it's read as BGR
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return None, None, "Failed to decode image frame."

        # Get faces using the processor
        # This can raise ModelError
        faces = processor.get_faces(frame)

        # --- Time the critical call ---
        inference_start_time = time.perf_counter()
        faces = processor.get_faces(frame)
        inference_duration = (time.perf_counter() - inference_start_time) * 1000 # Duration in ms
        logger.debug(f"Face detection/embedding took: {inference_duration:.2f} ms")
        # --- End Timing ---
        
        return frame, faces, None

    except ModelError as e:
        logger.error(f"Model error during live frame processing: {e}", exc_info=False)
        return None, None, f"Model processing error: {e.message}"
    except Exception as e:
        logger.error(f"Unexpected error decoding/processing frame: {e}", exc_info=True)
        return None, None, "Internal server error processing frame."

# --- Live Comparison WebSocket ---
@router.websocket("/live/ws")
async def websocket_live_compare(
    websocket: WebSocket,
    reference_id: str = Query(..., description="The ID of the reference face in the database to compare against."),
    processor: FaceProcessor = Depends(get_initialized_processor_ws),
    db_path: Path = Depends(get_database_path)
):
    """
    WebSocket endpoint for live face comparison against a single reference face.
    Client sends video frames (bytes), server sends comparison results.
    """
    await websocket.accept()
    logger.info(f"WS connected: Live Compare started for reference_id='{reference_id}', DB='{db_path}'")

    reference_embedding_np: Optional[np.ndarray] = None
    try:
        # 1. Fetch reference features for 'reference_id' from DB.
        reference_features_bytes = db_func.get_face_features_by_id(db_path, reference_id)
        if reference_features_bytes is None:
            logger.warning(f"Reference ID '{reference_id}' not found in DB '{db_path}'. Closing WebSocket.")
            await websocket.close(code=1008, reason=f"Reference ID '{reference_id}' not found.")
            return

        # Deserialize reference features
        reference_embedding_np = np.frombuffer(reference_features_bytes, dtype=np.float32)
        if reference_embedding_np.size != EXPECTED_EMBEDDING_DIM:
            logger.error(f"Reference ID '{reference_id}' has invalid feature size ({reference_embedding_np.size}). Closing.")
            await websocket.close(code=1011, reason=f"Invalid feature data for reference ID '{reference_id}'.")
            return
        logger.info(f"Reference embedding loaded successfully for ID '{reference_id}'.")

    except (DatabaseError, Exception) as e:
        logger.error(f"Failed to load reference features for ID '{reference_id}': {e}", exc_info=True)
        await websocket.close(code=1011, reason=f"Error loading reference data: {str(e)}")
        return

    last_process_time = time.time()
    try:
        while True:
            # Throttle frame processing
            current_time = time.time()
            if current_time - last_process_time < FRAME_PROCESSING_INTERVAL:
                await asyncio.sleep(FRAME_PROCESSING_INTERVAL - (current_time - last_process_time))
            last_process_time = time.time() # Update time *after* potential sleep

            # Receive frame data
            frame_bytes = await websocket.receive_bytes()
            logger.debug(f"LiveCompare WS: Received frame ({len(frame_bytes)} bytes) for '{reference_id}'.")

            # Process frame
            _frame, faces, error_msg = await process_frame_common(frame_bytes, processor)

            response_data = LiveCompareWSResponse(status="error", message=error_msg, reference_id=reference_id) # Default to error
            # Initialize response_data with common fields
            response_data_dict = {
                "status": "error", # Default
                "message": error_msg,
                "reference_id": reference_id,
                "detection_box": None,
                "all_detection_boxes": None
            }
            if error_msg:
                # Error during decoding/processing
                pass # response_data already set
            elif not faces:
                # No face detected
                if SEND_NO_FACE_UPDATES:
                    response_data_dict["status"] = "no_face"
                    response_data_dict["message"] = None # Clear error message
                else:
                    continue # Skip sending update
            elif len(faces) > 1:
                # Multiple faces detected
                response_data_dict["status"] = "multiple_faces"
                response_data_dict["message"] = None # Clear error message
                response_data_dict["all_detection_boxes"] = [f.bbox.astype(int).tolist() for f in faces if f.bbox is not None]
            else:
                # Exactly one face detected
                live_face = faces[0]
                live_embedding = live_face.normed_embedding
                bbox_list = live_face.bbox.astype(int).tolist() if live_face.bbox is not None else None
                response_data_dict["detection_box"] = bbox_list # Set single detection box

                if live_embedding is None:
                    logger.warning(f"LiveCompare WS: Detected face has no embedding for '{reference_id}'.")
                    response_data_dict["status"] = "error"
                    response_data_dict["message"] = "Failed to get embedding for detected face."
                else:
                    # Calculate similarity
                    similarity = FaceProcessor._cosine_similarity(reference_embedding_np, live_embedding)
                    is_match = similarity >= LIVE_COMPARE_THRESHOLD

                    logger.debug(f"LiveCompare WS: Comparison result for '{reference_id}': Sim={similarity:.4f}, Match={is_match}")
                    response_data_dict["status"] = "match_found" if is_match else "no_match"
                    response_data_dict["similarity"] = float(similarity)
                    response_data_dict["is_match"] = is_match
                    response_data_dict["message"] = None # Clear error message

            # Send the response (unless skipped)
            # Convert dict to Pydantic model then to dict for sending
            final_response = LiveCompareWSResponse(**response_data_dict)

            # Only send if it's an error, or if it's a status update we're configured to send
            should_send = False
            if final_response.status == "error" and final_response.message:
                should_send = True
            elif final_response.status == "no_face" and SEND_NO_FACE_UPDATES:
                should_send = True
            elif final_response.status in ["match_found", "no_match", "multiple_faces"]: # Always send these results
                should_send = True

            if should_send:
                await websocket.send_json(final_response.model_dump())


    except WebSocketDisconnect:
        logger.info(f"WS disconnected: Live Compare ended for reference_id='{reference_id}'.")
    except Exception as e:
        logger.error(f"Error in Live Compare WebSocket (Reference ID: {reference_id}): {e}", exc_info=True)
        # Attempt to close gracefully
        try:
            await websocket.close(code=1011, reason=f"Internal server error: {str(e)}")
        except RuntimeError: pass


# --- Live Search WebSocket ---
# Structure to hold loaded DB data in memory
class LiveSearchDBData:
    def __init__(self):
        self.embeddings: List[Tuple[str, str, np.ndarray, Optional[Dict[str, Any]]]] = [] # id, name, embedding_np, meta_dict
        self.is_loaded = False
        self.load_time = 0.0

    def load(self, db_path: Path):
        start_time = time.time()
        logger.info(f"LiveSearch WS: Loading database from {db_path} into memory...")
        self.embeddings = []
        try:
            db_records_raw = db_func.get_all_faces_from_db(db_path)
            if not db_records_raw:
                logger.warning(f"LiveSearch WS: Database '{db_path}' is empty.")
                self.is_loaded = True # Mark as loaded even if empty
                self.load_time = time.time() - start_time
                return

            for rec_id, rec_name, rec_features_bytes, rec_meta_str in db_records_raw:
                try:
                    embedding_np = np.frombuffer(rec_features_bytes, dtype=np.float32)
                    if embedding_np.size != EXPECTED_EMBEDDING_DIM:
                        logger.warning(f"LiveSearch WS: Skipping DB entry ID '{rec_id}': Feature size mismatch (got {embedding_np.size}).")
                        continue

                    meta_dict: Optional[Dict[str, Any]] = None
                    if rec_meta_str:
                        try:
                            meta_dict = json.loads(rec_meta_str)
                        except json.JSONDecodeError:
                            logger.warning(f"LiveSearch WS: Failed to parse metadata JSON for DB ID '{rec_id}'.")
                            meta_dict = {"_parse_error": "Invalid JSON in database"}

                    self.embeddings.append((rec_id, rec_name, embedding_np, meta_dict))

                except Exception as e:
                    logger.error(f"LiveSearch WS: Error processing DB entry ID '{rec_id}': {e}", exc_info=False)

            self.is_loaded = True
            self.load_time = time.time() - start_time
            logger.info(f"LiveSearch WS: Loaded {len(self.embeddings)} valid face records in {self.load_time:.3f}s.")

        except (DatabaseError, Exception) as e:
            logger.error(f"LiveSearch WS: Failed to load database '{db_path}': {e}", exc_info=True)
            self.is_loaded = False # Failed to load
            raise # Re-raise to be caught by the websocket handler


# Create a shared instance (consider locking if updates are possible)
live_search_db = LiveSearchDBData()


@router.websocket("/live-search/ws")
async def websocket_live_search(
    websocket: WebSocket,
    processor: FaceProcessor = Depends(get_initialized_processor_ws),
    db_path: Path = Depends(get_database_path) # Get DB path dependency
):
    """
    WebSocket endpoint for live face search against the database.
    Client sends video frames (bytes), server sends search results if a match is found.
    """
    await websocket.accept()
    logger.info(f"WS connected: Live Search started using DB='{db_path}'.")

    # Load/Reload DB data if not already loaded (or if using a more complex refresh mechanism)
    # Simple approach: Load once per server start or first connection.
    # For production, might need locking and a refresh strategy if DB changes.
    if not live_search_db.is_loaded:
        try:
            # This blocks the first connection until loaded. Consider background loading.
            live_search_db.load(db_path)
        except Exception as e:
            logger.error(f"LiveSearch WS: Critical - database load failed. Closing connection.", exc_info=True)
            await websocket.close(code=1011, reason=f"Failed to load face database: {e}")
            return

    if not live_search_db.embeddings:
        logger.warning("LiveSearch WS: Proceeding with empty database.")
        # Optionally send an initial status message to client?

    last_process_time = time.time()
    try:
        while True:
            # Throttle frame processing
            current_time = time.time()
            if current_time - last_process_time < FRAME_PROCESSING_INTERVAL:
                await asyncio.sleep(FRAME_PROCESSING_INTERVAL - (current_time - last_process_time))
            last_process_time = time.time()

            # Receive frame data
            frame_bytes = await websocket.receive_bytes()
            logger.debug(f"LiveSearch WS: Received frame ({len(frame_bytes)} bytes).")

            # Process frame
            _frame, faces, error_msg = await process_frame_common(frame_bytes, processor)

            if error_msg:
                # Send error message
                await websocket.send_json(LiveSearchWSResponse(status="error", message=error_msg).model_dump())
                continue
            elif not faces:
                # No face detected
                if SEND_NO_FACE_UPDATES:
                    await websocket.send_json(LiveSearchWSResponse(status="no_face_detected").model_dump())
                continue
            elif not live_search_db.embeddings:
                # Faces detected, but DB is empty
                if SEND_NO_MATCH_UPDATES: # Reuse this flag? Or a specific DB empty flag?
                    # Send 'no_match' for each detected face if desired
                    for face in faces:
                        bbox = face.bbox.astype(int).tolist() if face.bbox is not None else None
                        await websocket.send_json(LiveSearchWSResponse(status="no_match_found", detection_box=bbox).model_dump())
                continue


            any_match_found_in_this_frame = False # New flag for the whole frame
            processed_faces_count = 0

            for face in faces:
                processed_faces_count += 1
                live_embedding = face.normed_embedding
                bbox = face.bbox.astype(int).tolist() if face.bbox is not None else None

                if live_embedding is None:
                    logger.warning("LiveSearch WS: Detected face has no embedding.")
                    # Optionally send an error status for this specific face here if desired
                    # e.g., await websocket.send_json(LiveSearchWSResponse(status="error", message="Face has no embedding", detection_box=bbox).model_dump())
                    continue

                best_match_sim = -1.0
                best_match_info: Optional[Tuple[str, str, Optional[Dict[str, Any]]]] = None

                for db_id, db_name, db_embedding_np, db_meta_dict in live_search_db.embeddings:
                    similarity = FaceProcessor._cosine_similarity(live_embedding, db_embedding_np)
                    if similarity > best_match_sim:
                        best_match_sim = similarity
                        best_match_info = (db_id, db_name, db_meta_dict)

                if best_match_sim >= LIVE_SEARCH_THRESHOLD and best_match_info:
                    any_match_found_in_this_frame = True # Set flag
                    match_id, match_name, match_meta = best_match_info
                    match_detail = LiveSearchMatchDetail(
                        face_id=match_id,
                        name=match_name,
                        similarity=float(best_match_sim),
                        meta=match_meta
                    )
                    logger.debug(f"LiveSearch WS: Match found - ID={match_id}, Sim={best_match_sim:.4f}")
                    await websocket.send_json(
                        LiveSearchWSResponse(
                            status="match_found",
                            match=match_detail,
                            detection_box=bbox,
                            processed_frame_timestamp_ms=int(time.time() * 1000)
                        ).model_dump()
                    )
                else:
                    # Face detected, but no match above threshold for THIS face
                    if SEND_NO_MATCH_UPDATES:
                        await websocket.send_json(
                            LiveSearchWSResponse(status="no_match_found", detection_box=bbox).model_dump()
                        )
            # ---- AFTER processing ALL faces in the current video frame ----
            if processed_faces_count > 0 and not any_match_found_in_this_frame and not SEND_NO_MATCH_UPDATES:
                # This condition means:
                # 1. At least one face was processed.
                # 2. None of them resulted in a "match_found" response being sent.
                # 3. We are not configured to send individual "no_match_found" for each non-matching face.
                # So, send a general "no_match_found" for the frame, perhaps without a specific box,
                # or with the box of the first detected face if that makes sense for your UI.
                logger.debug("LiveSearch WS: Faces detected in frame, but no matches found above threshold.")
                await websocket.send_json(
                    LiveSearchWSResponse(status="no_match_found", message="No matches found for detected faces in this frame.").model_dump()
                )

    except WebSocketDisconnect:
        logger.info("WS disconnected: Live Search ended.")
    except Exception as e:
        logger.error(f"Error in Live Search WebSocket: {e}", exc_info=True)
        try:
            await websocket.close(code=1011, reason=f"Internal server error: {str(e)}")
        except RuntimeError: pass