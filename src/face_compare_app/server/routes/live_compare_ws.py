# src/face_compare_app/server/routes/live_compare_ws.py
import logging
import asyncio
import time
# import json # Not directly used here but good practice if models were more complex
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends

# Models specific to live compare
from ..models import LiveCompareWSResponse # Assuming this is still the correct model
# Import core/db functions and exceptions
from ... import core as core_func
from ... import database as db_func
from ...exceptions import DatabaseError, ModelError # Specific exceptions
from ...core import FaceProcessor
# Import dependency functions
from ..dependencies import get_initialized_processor_ws, get_database_path

# Common processing function (can be kept here or moved to a utils if shared more widely)
from .live_utils import process_frame_common # We'll create live_utils.py

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/live", tags=["Live Comparison WS"]) # Updated tag

# --- Constants ---
LIVE_COMPARE_THRESHOLD = 0.55 # Can be defined per route file
EXPECTED_EMBEDDING_DIM = 512
SEND_NO_FACE_UPDATES = True # Default for this route
FRAME_PROCESSING_INTERVAL = 0 # Default for this route

@router.websocket("/compare/ws") # Path relative to the router's prefix
async def websocket_live_compare(
    websocket: WebSocket,
    reference_id: str = Query(..., description="The ID of the reference face in the database to compare against."),
    processor: FaceProcessor = Depends(get_initialized_processor_ws),
    db_path: Path = Depends(get_database_path)
):
    await websocket.accept()
    logger.info(f"WS connected: Live Compare started for reference_id='{reference_id}', DB='{db_path}'")

    reference_embedding_np: Optional[np.ndarray] = None
    try:
        reference_features_bytes = db_func.get_face_features_by_id(db_path, reference_id)
        if reference_features_bytes is None:
            logger.warning(f"Ref ID '{reference_id}' not found in DB '{db_path}'. Closing.")
            await websocket.close(code=1008, reason=f"Reference ID '{reference_id}' not found.")
            return
        reference_embedding_np = np.frombuffer(reference_features_bytes, dtype=np.float32)
        if reference_embedding_np.size != EXPECTED_EMBEDDING_DIM:
            logger.error(f"Ref ID '{reference_id}' invalid feature size ({reference_embedding_np.size}). Closing.")
            await websocket.close(code=1011, reason="Invalid feature data for reference ID.")
            return
        logger.info(f"Ref embedding loaded for ID '{reference_id}'.")
    except (DatabaseError, Exception) as e:
        logger.error(f"Failed to load ref features for ID '{reference_id}': {e}", exc_info=True)
        await websocket.close(code=1011, reason=f"Error loading reference data: {str(e)}")
        return

    last_process_time = time.time()
    try:
        while True:
            current_time = time.time()
            if FRAME_PROCESSING_INTERVAL > 0 and (current_time - last_process_time < FRAME_PROCESSING_INTERVAL):
                await asyncio.sleep(FRAME_PROCESSING_INTERVAL - (current_time - last_process_time))
            last_process_time = time.time()

            frame_bytes = await websocket.receive_bytes()
            logger.debug(f"LiveCompare WS: Rx frame ({len(frame_bytes)} bytes) for '{reference_id}'.")

            _frame, faces, error_msg = await process_frame_common(frame_bytes, processor)

            response_data_dict = {
                "status": "error", "message": error_msg, "reference_id": reference_id,
                "detection_box": None, "all_detection_boxes": None,
                "similarity": None, "is_match": None # Ensure all fields are present
            }

            if error_msg:
                pass
            elif not faces:
                if SEND_NO_FACE_UPDATES:
                    response_data_dict["status"] = "no_face"
                    response_data_dict["message"] = None
                else: continue
            elif len(faces) > 1:
                response_data_dict["status"] = "multiple_faces"
                response_data_dict["message"] = None
                response_data_dict["all_detection_boxes"] = [f.bbox.astype(int).tolist() for f in faces if f.bbox is not None]
            else: # Exactly one face
                live_face = faces[0]
                live_embedding = live_face.normed_embedding
                bbox_list = live_face.bbox.astype(int).tolist() if live_face.bbox is not None else None
                response_data_dict["detection_box"] = bbox_list

                if live_embedding is None:
                    response_data_dict["status"] = "error"
                    response_data_dict["message"] = "Failed to get embedding for detected face."
                else:
                    similarity = FaceProcessor._cosine_similarity(reference_embedding_np, live_embedding)
                    is_match = similarity >= LIVE_COMPARE_THRESHOLD
                    response_data_dict["status"] = "match_found" if is_match else "no_match"
                    response_data_dict["similarity"] = float(similarity)
                    response_data_dict["is_match"] = is_match
                    response_data_dict["message"] = None
                    logger.debug(f"LiveCompare WS: Result for '{reference_id}': Sim={similarity:.4f}, Match={is_match}")
            
            final_response = LiveCompareWSResponse(**response_data_dict) # Ensure all fields present
            should_send = (final_response.status == "error" and final_response.message) or \
                          (final_response.status == "no_face" and SEND_NO_FACE_UPDATES) or \
                          (final_response.status in ["match_found", "no_match", "multiple_faces"])

            if should_send:
                await websocket.send_json(final_response.model_dump())

    except WebSocketDisconnect:
        logger.info(f"WS disconnected: Live Compare for ref_id='{reference_id}'.")
    except Exception as e:
        logger.error(f"Error in Live Compare WS (Ref ID: {reference_id}): {e}", exc_info=True)
        try: await websocket.close(code=1011, reason=f"Internal server error: {str(e)}")
        except RuntimeError: pass