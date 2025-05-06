# src/face_compare_app/live.py
"""Live camera face comparison logic."""
import logging
import time
from typing import Optional
from pathlib import Path

import cv2 # Import OpenCV
import numpy as np # Import Numpy

# Relative imports
from .core import FaceProcessor # Need the class for type hinting
from .exceptions import (
    ImageLoadError, NoFaceFoundError,
    MultipleFacesFoundError, EmbeddingError, ModelError
)
from .utils import load_image # Might need if we load reference inside, but better outside

logger = logging.getLogger(__name__)

# --- Constants ---
# Similarity threshold - determines if faces match. Tune based on model/use case.
# buffalo_l often uses thresholds around 0.6 for LFW benchmark, but real-world might need lower/higher.
SIMILARITY_THRESHOLD = 0.55 # Example: Adjust as needed
# Display configuration
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
BOX_THICKNESS = 2
MATCH_COLOR = (0, 255, 0) # Green
NO_MATCH_COLOR = (0, 0, 255) # Red
INFO_COLOR = (255, 255, 255) # White
MULTI_FACE_COLOR = (255, 165, 0) # Orange
NO_FACE_TEXT = "Searching for face..."
MULTI_FACE_TEXT = "Multiple faces detected"
PROCESSING_ERROR_TEXT = "Processing Error"


def run_live_comparison(
    processor: FaceProcessor, # Pass the initialized processor
    camera_id: int,
    reference_image_path: str,
    similarity_threshold: float = SIMILARITY_THRESHOLD # Allow override
):
    """
    Runs live face comparison using a camera feed against a reference image.

    Args:
        processor: The initialized FaceProcessor instance.
        camera_id: The ID of the camera device to use.
        reference_image_path: Path to the reference face image.
        similarity_threshold: The threshold for considering faces a match.

    Raises:
        FileNotFoundError: If the reference image is not found.
        ImageLoadError: If the reference image cannot be loaded.
        NoFaceFoundError: If no face is found in the reference image.
        MultipleFacesFoundError: If multiple faces are found in the reference image.
        ModelError: If there's an issue with the model during reference processing or live feed.
        EmbeddingError: If embedding fails for reference or live face.
        RuntimeError: If the camera cannot be opened.
        FaceCompareError: For other comparison-related errors.
    """
    logger.info(f"Starting live comparison using camera ID: {camera_id}")
    logger.info(f"Reference image: {reference_image_path}")
    logger.info(f"Similarity threshold: {similarity_threshold}")

    # --- 1. Load reference image and extract its features ---
    try:
        logger.info("Loading reference image and extracting features...")
        ref_path = Path(reference_image_path)
        # Use the processor's method which handles loading, detection, single face check, embedding
        reference_embedding = processor.get_single_face_embedding(ref_path)
        logger.info(f"Reference features extracted successfully from {ref_path.name}.")
    except (FileNotFoundError, ImageLoadError, NoFaceFoundError, MultipleFacesFoundError, ModelError, EmbeddingError) as e:
        # Log the specific error and re-raise it for the CLI to handle
        logger.error(f"Failed to process reference image '{reference_image_path}': {e}")
        raise e # Propagate the specific error

    # --- 2. Initialize Camera ---
    logger.info(f"Initializing camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error(f"Cannot open camera ID: {camera_id}")
        raise RuntimeError(f"Cannot open camera ID: {camera_id}")

    # <<<--- Add this small delay ---<<<
    logger.debug("Adding short delay after camera init...")
    time.sleep(0.5) # Wait half a second

    logger.info("Camera initialized. Starting live feed loop (press 'q' to stop)...")
    frame_count = 0
    start_time = time.time()
    first_frame_read = False # Keep track if we ever got a frame

    try:
        # --- 3. Live Feed Loop ---
        while True:
            frame_count += 1
            ret, frame = cap.read()
            if not ret:
                # <<<--- Modify error handling for immediate failure ---<<<
                if not first_frame_read and frame_count == 1:
                    logger.error(f"Failed to grab the *first* frame from camera {camera_id}. Check permissions or if camera is in use.")
                    # Raise an error here so the CLI knows it failed critically
                    raise RuntimeError(f"Failed to capture initial frame from camera {camera_id}. Check permissions/availability.")
                else:
                    # If it fails later, just warn and break as before
                    logger.warning(f"Failed to grab frame from camera (frame {frame_count}). End of stream or temporary issue?")
                    break # Exit the loop
            
            # If we reach here, a frame was successfully read
            if not first_frame_read:
                first_frame_read = True
                logger.debug("Successfully read the first frame.")

            display_frame = frame.copy() # Work on a copy for drawing
            status_text = NO_FACE_TEXT
            box_color = INFO_COLOR

            try:
                # --- 3a. Detect Faces in Current Frame ---
                # Use the processor's get_faces method
                # Note: get_faces can raise ModelError if inference fails
                live_faces = processor.get_faces(frame)

                if len(live_faces) == 0:
                    # No face detected, status_text remains NO_FACE_TEXT
                    pass # Handled by default status_text

                elif len(live_faces) > 1:
                    # Multiple faces detected
                    status_text = MULTI_FACE_TEXT
                    box_color = MULTI_FACE_COLOR
                    logger.debug(f"Frame {frame_count}: Multiple ({len(live_faces)}) faces detected.")
                    # Draw boxes for all detected faces
                    for face in live_faces:
                        bbox = face.bbox.astype(int)
                        cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_color, BOX_THICKNESS)

                else: # Exactly one face detected
                    live_face = live_faces[0]
                    bbox = live_face.bbox.astype(int)
                    live_embedding = live_face.normed_embedding

                    if live_embedding is None:
                        logger.warning(f"Frame {frame_count}: Detected face has no embedding.")
                        status_text = PROCESSING_ERROR_TEXT # Or some other indicator
                        box_color = NO_MATCH_COLOR # Use red for error state?
                        cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_color, BOX_THICKNESS)
                    else:
                        # --- 3b. Compare with Reference ---
                        similarity = processor._cosine_similarity(reference_embedding, live_embedding)
                        logger.debug(f"Frame {frame_count}: Single face detected. Similarity: {similarity:.4f}")

                        # --- 3c. Determine Match & Draw ---
                        if similarity >= similarity_threshold:
                            status_text = f"MATCH: {similarity:.2f}"
                            box_color = MATCH_COLOR
                            logger.info(f"Frame {frame_count}: Match found! Similarity: {similarity:.2f}")
                        else:
                            status_text = f"NO MATCH: {similarity:.2f}"
                            box_color = NO_MATCH_COLOR

                        # Draw bounding box around the single detected face
                        cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_color, BOX_THICKNESS)
                        # Put status text near the box
                        text_y = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[3] + 20
                        cv2.putText(display_frame, status_text, (bbox[0], text_y), FONT, FONT_SCALE, box_color, FONT_THICKNESS)

            except ModelError as me: # Catch errors during processor.get_faces
                logger.error(f"Frame {frame_count}: Model processing error: {me}", exc_info=False) # Avoid overly verbose logs in loop
                status_text = PROCESSING_ERROR_TEXT
                box_color = NO_MATCH_COLOR # Indicate error
                # Optionally display the error on frame? Maybe too verbose.
            except Exception as frame_err: # Catch unexpected errors during frame processing
                logger.error(f"Frame {frame_count}: Unexpected error processing frame: {frame_err}", exc_info=True)
                status_text = PROCESSING_ERROR_TEXT
                box_color = NO_MATCH_COLOR
                # Don't crash the loop if possible, just mark the frame

            # --- 3d. Display Frame ---
            # Add general status text if no face-specific text was set
            if len(live_faces) == 0 or (len(live_faces) > 1 and status_text == MULTI_FACE_TEXT):
                cv2.putText(display_frame, status_text, (10, 30), FONT, FONT_SCALE, box_color, FONT_THICKNESS)

            cv2.imshow("Live Face Comparison (Press 'q' to quit)", display_frame)

            # --- 3e. Check for Exit Key ---
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Exit key 'q' pressed. Stopping live feed.")
                break

    except KeyboardInterrupt:
        logger.info("Live comparison interrupted by user (Ctrl+C).")
    finally:
        # --- 4. Cleanup ---
        end_time = time.time()
        logger.info(f"Live comparison loop finished. Processed {frame_count} frames in {end_time - start_time:.2f} seconds.")
        if cap.isOpened():
            cap.release()
            logger.debug("Camera released.")
        cv2.destroyAllWindows()
        logger.debug("Display windows closed.")