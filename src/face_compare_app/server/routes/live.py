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
from ..models import LiveSearchSingleFaceResult, MultiLiveSearchWSResponse, LiveSearchMatchDetail
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
SEND_NO_FACE_UPDATES = True # Control if "no_face" messages are sent
SEND_NO_MATCH_UPDATES = True # Control if "no_match" messages are sent (for live search)
FRAME_PROCESSING_INTERVAL = 0 # Minimum seconds between processing frames (controls rate)


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


# --- LiveSearchDBData class remains the same ---
class LiveSearchDBData:
    def __init__(self):
        self.embeddings: List[Tuple[str, str, np.ndarray, Optional[Dict[str, Any]]]] = []
        self.is_loaded = False
        self.load_time = 0.0
    def load(self, db_path: Path): # Implementation as before
        start_time = time.time()
        logger.info(f"LiveSearch WS: Loading database from {db_path} into memory...")
        self.embeddings = []
        try:
            db_records_raw = db_func.get_all_faces_from_db(db_path)
            if not db_records_raw:
                logger.warning(f"LiveSearch WS: Database '{db_path}' is empty.")
                self.is_loaded = True
                self.load_time = time.time() - start_time
                return

            for rec_id, rec_name, rec_features_bytes, rec_meta_str in db_records_raw:
                try:
                    embedding_np = np.frombuffer(rec_features_bytes, dtype=np.float32)
                    if embedding_np.size != EXPECTED_EMBEDDING_DIM:
                        logger.warning(f"LiveSearch WS: Skipping DB entry ID '{rec_id}': Feature size mismatch.")
                        continue
                    meta_dict: Optional[Dict[str, Any]] = None
                    if rec_meta_str:
                        try: meta_dict = json.loads(rec_meta_str)
                        except json.JSONDecodeError:
                            logger.warning(f"LiveSearch WS: Invalid metadata JSON for DB ID '{rec_id}'.")
                            meta_dict = {"_parse_error": "Invalid JSON"}
                    self.embeddings.append((rec_id, rec_name, embedding_np, meta_dict))
                except Exception as e:
                    logger.error(f"LiveSearch WS: Error processing DB entry ID '{rec_id}': {e}", exc_info=False)
            self.is_loaded = True
            self.load_time = time.time() - start_time
            logger.info(f"LiveSearch WS: Loaded {len(self.embeddings)} valid records in {self.load_time:.3f}s.")
        except Exception as e: # Catch DatabaseError, etc.
            logger.error(f"LiveSearch WS: Failed to load database '{db_path}': {e}", exc_info=True)
            self.is_loaded = False; raise

live_search_db = LiveSearchDBData() # Global instance


@router.websocket("/live-search/ws")
async def websocket_live_search(
    websocket: WebSocket,
    processor: FaceProcessor = Depends(get_initialized_processor_ws),
    db_path: Path = Depends(get_database_path)
):
    await websocket.accept()
    logger.info(f"WS connected: Multi-Face Live Search started using DB='{db_path}'.")

    if not live_search_db.is_loaded:
        try:
            live_search_db.load(db_path)
        except Exception as e:
            logger.error(f"LiveSearch WS: Critical DB load failed. Closing. Error: {e}", exc_info=True)
            # Try to send an error before closing
            error_response = MultiLiveSearchWSResponse(
                faces_results=[],
                processed_frame_timestamp_ms=int(time.time() * 1000),
                processing_time_ms=0,
                frame_error_message=f"Failed to load face database: {e}"
            )
            try: await websocket.send_json(error_response.model_dump())
            except: pass # Ignore if send fails
            await websocket.close(code=1011, reason="Database load failure.")
            return

    last_throttle_time = time.time()
    try:
        while True:
            current_loop_time = time.time()
            if current_loop_time - last_throttle_time < FRAME_PROCESSING_INTERVAL:
                await asyncio.sleep(FRAME_PROCESSING_INTERVAL - (current_loop_time - last_throttle_time))
            last_throttle_time = time.time()

            frame_processing_start_time = time.perf_counter()
            frame_bytes = await websocket.receive_bytes()
            logger.debug(f"LiveSearch WS: Received frame ({len(frame_bytes)} bytes).")

            _frame, detected_faces_insight, frame_proc_error_msg = await process_frame_common(frame_bytes, processor)

            all_faces_results: List[LiveSearchSingleFaceResult] = []
            global_frame_error: Optional[str] = frame_proc_error_msg

            if not global_frame_error and detected_faces_insight:
                if not live_search_db.embeddings:
                    logger.warning("LiveSearch WS: Faces detected, but database is empty.")
                    # Create a "no_match_found" result for each detected face due to empty DB
                    for insight_face in detected_faces_insight:
                        if insight_face.bbox is not None:
                            all_faces_results.append(LiveSearchSingleFaceResult(
                                status="no_match_found",
                                detection_box=insight_face.bbox.astype(int).tolist(),
                                message="Database is empty."
                            ))
                else:
                    # Process each detected face
                    for insight_face in detected_faces_insight:
                        bbox_list = insight_face.bbox.astype(int).tolist() if insight_face.bbox is not None else [-1,-1,-1,-1] # Default box if None
                        live_embedding = insight_face.normed_embedding

                        if live_embedding is None:
                            logger.warning("LiveSearch WS: Detected face has no embedding.")
                            all_faces_results.append(LiveSearchSingleFaceResult(
                                status="error_embedding",
                                detection_box=bbox_list,
                                message="Failed to get embedding for this face."
                            ))
                            continue # Next face

                        best_match_sim = -1.0
                        best_match_detail_info: Optional[LiveSearchMatchDetail] = None

                        for db_id, db_name, db_embedding_np, db_meta_dict in live_search_db.embeddings:
                            similarity = FaceProcessor._cosine_similarity(live_embedding, db_embedding_np)
                            if similarity > best_match_sim:
                                best_match_sim = similarity
                                if similarity >= LIVE_SEARCH_THRESHOLD: # Only store details if it's a potential match
                                    best_match_detail_info = LiveSearchMatchDetail(
                                        face_id=db_id,
                                        name=db_name,
                                        similarity=float(similarity), # Store the actual similarity
                                        meta=db_meta_dict
                                    )
                        
                        if best_match_detail_info: # A match above threshold was found
                            all_faces_results.append(LiveSearchSingleFaceResult(
                                status="match_found",
                                match_detail=best_match_detail_info,
                                detection_box=bbox_list
                            ))
                            logger.debug(f"LiveSearch WS: Match for a face - ID={best_match_detail_info.face_id}, Sim={best_match_detail_info.similarity:.4f}")
                        else: # No match above threshold for this face
                            all_faces_results.append(LiveSearchSingleFaceResult(
                                status="no_match_found",
                                detection_box=bbox_list,
                                message=f"No match (best sim: {best_match_sim:.3f})" # Optionally send best sim
                            ))
            elif not global_frame_error and not detected_faces_insight:
                # No faces detected in the frame, but no error during processing
                logger.debug("LiveSearch WS: No faces detected in the frame.")
                # `all_faces_results` will be empty, which is fine. Client can interpret.
                pass


            processing_time_ms = int((time.perf_counter() - frame_processing_start_time) * 1000)
            response_to_send = MultiLiveSearchWSResponse(
                faces_results=all_faces_results,
                processed_frame_timestamp_ms=int(time.time() * 1000),
                processing_time_ms=processing_time_ms,
                frame_error_message=global_frame_error
            )
            
            logger.debug(f"LiveSearch WS: Sending {len(all_faces_results)} face results. FrameErr: {global_frame_error}. ProcTime: {processing_time_ms}ms")
            await websocket.send_json(response_to_send.model_dump())

    except WebSocketDisconnect:
        logger.info("WS disconnected: Multi-Face Live Search ended.")
    except Exception as e:
        logger.error(f"Error in Multi-Face Live Search WebSocket: {e}", exc_info=True)
        try:
            proc_time = int((time.perf_counter() - frame_processing_start_time) * 1000) if 'frame_processing_start_time' in locals() else 0
            await websocket.send_json(MultiLiveSearchWSResponse(
                faces_results=[],
                processed_frame_timestamp_ms=int(time.time()*1000),
                processing_time_ms=proc_time,
                frame_error_message=f"Internal server error: {str(e)}"
            ).model_dump())
            await websocket.close(code=1011)
        except RuntimeError: pass
        except Exception as e2: logger.error(f"Error sending error report: {e2}")