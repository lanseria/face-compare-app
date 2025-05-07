# src/face_compare_app/server/routes/live_search_ws.py
import logging
import asyncio
import time
import json
import cv2 # Keep cv2 for process_frame_common if it stays here
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends # Query not needed here

# Models specific to live search
from ..models import LiveSearchSingleFaceResult, MultiLiveSearchWSResponse, LiveSearchMatchDetail
# Import core/db functions and exceptions
from ... import core as core_func
from ... import database as db_func
from ...exceptions import ModelError # Specific exceptions
from ...core import FaceProcessor
# Import dependency functions
from ..dependencies import get_initialized_processor_ws, get_database_path

# Common processing function (can be kept here or moved to a utils if shared more widely)
from .live_utils import process_frame_common, live_search_db_instance # We'll create live_utils.py

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/live", tags=["Live Search WS"]) # Updated tag

# --- Constants ---
LIVE_SEARCH_THRESHOLD = 0.55
FRAME_PROCESSING_INTERVAL = 0 # Default for this route


@router.websocket("/search/ws") # Path relative to the router's prefix
async def websocket_live_search(
    websocket: WebSocket,
    processor: FaceProcessor = Depends(get_initialized_processor_ws),
    db_path: Path = Depends(get_database_path)
):
    await websocket.accept()
    logger.info(f"WS connected: Multi-Face Live Search started using DB='{db_path}'.")

    if not live_search_db_instance.is_loaded: # Use the instance
        try:
            live_search_db_instance.load(db_path)
        except Exception as e:
            logger.error(f"LiveSearch WS: Critical DB load failed. Closing. Error: {e}", exc_info=True)
            error_response = MultiLiveSearchWSResponse(
                faces_results=[],
                processed_frame_timestamp_ms=int(time.time() * 1000),
                processing_time_ms=0,
                frame_error_message=f"Failed to load face database: {e}"
            )
            try: await websocket.send_json(error_response.model_dump())
            except: pass
            await websocket.close(code=1011, reason="Database load failure.")
            return

    last_throttle_time = time.time()
    try:
        while True:
            current_loop_time = time.time()
            if FRAME_PROCESSING_INTERVAL > 0 and (current_loop_time - last_throttle_time < FRAME_PROCESSING_INTERVAL):
                await asyncio.sleep(FRAME_PROCESSING_INTERVAL - (current_loop_time - last_throttle_time))
            last_throttle_time = time.time()

            frame_processing_start_time = time.perf_counter()
            frame_bytes = await websocket.receive_bytes()
            logger.debug(f"LiveSearch WS: Rx frame ({len(frame_bytes)} bytes).")

            _frame, detected_faces_insight, frame_proc_error_msg = await process_frame_common(frame_bytes, processor)

            all_faces_results: List[LiveSearchSingleFaceResult] = []
            global_frame_error: Optional[str] = frame_proc_error_msg

            if not global_frame_error and detected_faces_insight:
                if not live_search_db_instance.embeddings: # Use the instance
                    logger.warning("LiveSearch WS: Faces detected, but database is empty.")
                    for insight_face in detected_faces_insight:
                        if insight_face.bbox is not None:
                            all_faces_results.append(LiveSearchSingleFaceResult(
                                status="no_match_found",
                                detection_box=insight_face.bbox.astype(int).tolist(),
                                message="Database is empty."
                            ))
                else:
                    for insight_face in detected_faces_insight:
                        bbox_list = insight_face.bbox.astype(int).tolist() if insight_face.bbox is not None else [-1,-1,-1,-1]
                        live_embedding = insight_face.normed_embedding
                        if live_embedding is None:
                            all_faces_results.append(LiveSearchSingleFaceResult(
                                status="error_embedding", detection_box=bbox_list,
                                message="Failed to get embedding for this face."
                            )); continue

                        best_match_sim = -1.0
                        best_match_detail_info: Optional[LiveSearchMatchDetail] = None
                        for db_id, db_name, db_embedding_np, db_meta_dict in live_search_db_instance.embeddings: # Use the instance
                            similarity = FaceProcessor._cosine_similarity(live_embedding, db_embedding_np)
                            if similarity > best_match_sim:
                                best_match_sim = similarity
                                if similarity >= LIVE_SEARCH_THRESHOLD:
                                    best_match_detail_info = LiveSearchMatchDetail(
                                        face_id=db_id, name=db_name,
                                        similarity=float(similarity), meta=db_meta_dict
                                    )
                        if best_match_detail_info:
                            all_faces_results.append(LiveSearchSingleFaceResult(
                                status="match_found", match_detail=best_match_detail_info,
                                detection_box=bbox_list
                            )); logger.debug(f"LiveSearch WS: Match ID={best_match_detail_info.face_id}, Sim={best_match_detail_info.similarity:.4f}")
                        else:
                            all_faces_results.append(LiveSearchSingleFaceResult(
                                status="no_match_found", detection_box=bbox_list,
                                message=f"No match (best sim: {best_match_sim:.3f})"
                            ))
            elif not global_frame_error and not detected_faces_insight:
                logger.debug("LiveSearch WS: No faces detected in frame.")
            
            processing_time_ms = int((time.perf_counter() - frame_processing_start_time) * 1000)
            response_to_send = MultiLiveSearchWSResponse(
                faces_results=all_faces_results,
                processed_frame_timestamp_ms=int(time.time() * 1000),
                processing_time_ms=processing_time_ms,
                frame_error_message=global_frame_error
            )
            logger.debug(f"LiveSearch WS: Sending {len(all_faces_results)} results. FrameErr: {global_frame_error}. ProcTime: {processing_time_ms}ms")
            await websocket.send_json(response_to_send.model_dump())

    except WebSocketDisconnect:
        logger.info("WS disconnected: Multi-Face Live Search.")
    except Exception as e:
        logger.error(f"Error in Multi-Face Live Search WS: {e}", exc_info=True)
        try:
            proc_time = int((time.perf_counter() - frame_processing_start_time) * 1000) if 'frame_processing_start_time' in locals() else 0
            await websocket.send_json(MultiLiveSearchWSResponse(
                faces_results=[], processed_frame_timestamp_ms=int(time.time()*1000),
                processing_time_ms=proc_time, frame_error_message=f"Internal server error: {str(e)}"
            ).model_dump())
            await websocket.close(code=1011)
        except RuntimeError: pass
        except Exception as e2: logger.error(f"Error sending error report: {e2}")