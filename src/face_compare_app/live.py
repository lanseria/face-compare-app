# src/face_compare_app/live.py
"""Live camera face comparison logic."""
import logging
import time
from typing import Optional, List, Tuple
from pathlib import Path

import cv2 # Import OpenCV
import numpy as np # Import Numpy
# --- Pillow Imports --- <<< ADDED
from PIL import Image, ImageDraw, ImageFont

# Relative imports
from .core import FaceProcessor # Need the class for type hinting
from .exceptions import (
    FaceCompareError, ImageLoadError, NoFaceFoundError,
    MultipleFacesFoundError, EmbeddingError, ModelError,
    DatabaseError, InvalidInputError # Added DB/Input Error
)
from .utils import load_image # Might need if we load reference inside, but better outside
from .database import get_all_faces_from_db # Import DB function

logger = logging.getLogger(__name__)

# --- Constants ---
# Similarity threshold - determines if faces match. Tune based on model/use case.
# buffalo_l often uses thresholds around 0.6 for LFW benchmark, but real-world might need lower/higher.
# --- Constants (can be shared or redefined) ---
SIMILARITY_THRESHOLD = 0.55 # Example: Adjust as needed
FONT = cv2.FONT_HERSHEY_SIMPLEX
# --- Font Settings --- <<< ADDED
# !!! IMPORTANT: SET THIS PATH TO A VALID TTF/OTF FONT FILE ON YOUR SYSTEM !!!
# Example paths (choose ONE and make sure it exists):
# FONT_PATH_TTF = "C:/Windows/Fonts/simhei.ttf" # Windows SimHei Example
FONT_PATH_TTF = "/System/Library/Fonts/PingFang.ttc" # macOS PingFang Example (index 0 usually works for ttc)
# FONT_PATH_TTF = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc" # Linux WenQuanYi Example
FONT_SIZE = 18 # Adjust font size as needed

FONT_SCALE = 0.6 # Slightly smaller for potentially more text
FONT_THICKNESS = 2
BOX_THICKNESS = 2
MATCH_COLOR = (0, 255, 0) # Green
NO_MATCH_COLOR = (0, 0, 255) # Red (Unknown)
INFO_COLOR = (255, 255, 255) # White
MULTI_FACE_COLOR = (255, 165, 0) # Orange (can be used if needed)
ERROR_COLOR = (255, 0, 255) # Magenta for processing errors
NO_FACE_TEXT = "Searching for faces..."
MULTI_FACE_TEXT = "Multiple faces detected!"
NO_MATCH_TEXT = "No match found."
DB_EMPTY_TEXT = "Database is empty. Cannot search."
PROCESSING_ERROR_TEXT = "Proc. Error"
NO_EMBEDDING_TEXT = "No Embedding"

# --- Helper Function to Draw Text using Pillow --- <<< ADDED
def draw_text_pil(img_bgr, text, position, font_object, color_rgb):
    """Draws text on a BGR NumPy array using Pillow."""
    try:
        # Convert BGR NumPy array to Pillow RGB Image
        img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        # Draw text using the loaded font
        draw.text(position, text, font=font_object, fill=color_rgb)
        # Convert back to BGR NumPy array
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.error(f"Error drawing text with Pillow: {e}", exc_info=False)
        # Return original image if drawing fails
        return img_bgr

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


# --- New Function for Live Search ---
def run_live_search(
    processor: FaceProcessor,
    camera_id: int,
    db_path: Path, # Use Path object
    similarity_threshold: float = SIMILARITY_THRESHOLD
):
    """
    Runs live face detection and searches recognized faces against a database.

    Args:
        processor: The initialized FaceProcessor instance.
        camera_id: The ID of the camera device to use.
        db_path: Path object pointing to the SQLite database file.
        similarity_threshold: The threshold for considering faces a match.

    Raises:
        DatabaseError: If the database cannot be loaded or accessed.
        RuntimeError: If the camera cannot be opened.
        ModelError: If there's an issue with the model during live feed processing.
        # Other errors might be logged but attempt to continue if possible
    """
    logger.info(f"Starting live search using camera ID: {camera_id}")
    logger.info(f"Database path: {db_path}")
    logger.info(f"Similarity threshold: {similarity_threshold}")

    # --- 1. Load Database Features into Memory ---
    try:
        logger.info(f"Loading face database from: {db_path}")
        # Returns List[Tuple[str, str, bytes, Optional[str]]]
        db_records_raw = get_all_faces_from_db(db_path)
        if not db_records_raw:
            logger.warning(f"Database '{db_path}' is empty or could not be read properly.")
            # Decide how to handle: error out or run detection without search?
            # Let's proceed but indicate DB is empty on screen.
            db_embeddings = []
        else:
            db_embeddings = [] # List to hold (id, name, numpy_embedding)
            expected_dims = 512 # TODO: Get this from the processor/model if possible
            for rec_id, rec_name, rec_features_bytes, _ in db_records_raw:
                try:
                    embedding_np = np.frombuffer(rec_features_bytes, dtype=np.float32)
                    if embedding_np.size == expected_dims:
                        db_embeddings.append((rec_id, rec_name, embedding_np))
                    else:
                        logger.warning(f"Skipping DB entry ID '{rec_id}': Feature size mismatch (got {embedding_np.size}, expected {expected_dims}).")
                except Exception as e:
                    logger.error(f"Error processing DB entry ID '{rec_id}': {e}", exc_info=False)
            logger.info(f"Loaded {len(db_embeddings)} valid face records into memory for search.")

    except DatabaseError as e:
        logger.error(f"Failed to load database '{db_path}': {e}")
        raise e # Propagate DB errors to the CLI for clear failure
    except Exception as e: # Catch unexpected errors during DB load
        logger.error(f"Unexpected error loading database '{db_path}': {e}", exc_info=True)
        raise DatabaseError(f"Unexpected error loading database: {e}") from e


    # --- 2. Initialize Camera ---
    logger.info(f"Initializing camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error(f"Cannot open camera ID: {camera_id}")
        raise RuntimeError(f"Cannot open camera ID: {camera_id}")

    logger.debug("Adding short delay after camera init...")
    time.sleep(0.5)

    logger.info("Camera initialized. Starting live search loop (press 'q' to stop)...")
    frame_count = 0
    start_time = time.time()
    first_frame_read = False

    try:
        # --- 3. Live Feed Loop ---
        while True:
            frame_count += 1
            ret, frame = cap.read()
            if not ret:
                if not first_frame_read and frame_count == 1:
                    logger.error(f"Failed to grab the *first* frame from camera {camera_id}. Check permissions or if camera is in use.")
                    raise RuntimeError(f"Failed to capture initial frame from camera {camera_id}.")
                else:
                    logger.warning(f"Failed to grab frame from camera (frame {frame_count}). End of stream or temporary issue?")
                    break
            if not first_frame_read:
                first_frame_read = True
                logger.debug("Successfully read the first frame.")

            display_frame = frame.copy()
            status_text = NO_FACE_TEXT # Default text

            # Display DB status if empty
            if not db_embeddings:
                cv2.putText(display_frame, DB_EMPTY_TEXT, (10, 30), FONT, FONT_SCALE, INFO_COLOR, FONT_THICKNESS)

            try:
                # --- 3a. Detect Faces in Current Frame ---
                live_faces = processor.get_faces(frame)
                detected_count = len(live_faces)

                if detected_count == 0:
                    # No faces, draw general status text later
                    pass
                else:
                    status_text = f"Detected: {detected_count}" # Update status

                    # --- 3b. Process Each Detected Face ---
                    for face in live_faces:
                        bbox = face.bbox.astype(int)
                        live_embedding = face.normed_embedding
                        face_label = "Unknown" # Default label
                        box_color = NO_MATCH_COLOR # Default color

                        if live_embedding is None:
                            face_label = NO_EMBEDDING_TEXT
                            box_color = ERROR_COLOR
                            logger.warning(f"Frame {frame_count}: Detected face (bbox: {bbox}) has no embedding.")
                        elif not db_embeddings:
                            # DB is empty, label remains "Unknown"
                            pass
                        else:
                            # --- 3c. Compare with Database Entries ---
                            best_match_sim = -1.0
                            best_match_name = "Unknown"

                            for db_id, db_name, db_embedding_np in db_embeddings:
                                similarity = processor._cosine_similarity(live_embedding, db_embedding_np)
                                if similarity > best_match_sim:
                                    best_match_sim = similarity
                                    best_match_name = db_id # Store the name of the best match so far

                            # --- 3d. Determine Match & Label ---
                            if best_match_sim >= similarity_threshold:
                                face_label = f"{best_match_name} ({best_match_sim:.2f})"
                                box_color = MATCH_COLOR
                                logger.debug(f"Frame {frame_count}: Match found - {face_label} for face at {bbox}")
                            else:
                                # Best match wasn't good enough, remains "Unknown"
                                # face_label = "Unknown" # Already default
                                logger.debug(f"Frame {frame_count}: No match above threshold ({best_match_sim:.2f} < {similarity_threshold}) for face at {bbox}")

                        # --- 3e. Draw Bounding Box and Label ---
                        cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_color, BOX_THICKNESS)
                        text_y = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[3] + 20
                        cv2.putText(display_frame, face_label, (bbox[0], text_y), FONT, FONT_SCALE, box_color, FONT_THICKNESS)

            except ModelError as me:
                logger.error(f"Frame {frame_count}: Model processing error: {me}", exc_info=False)
                status_text = PROCESSING_ERROR_TEXT
                # Draw general error status text
                cv2.putText(display_frame, status_text, (10, 30), FONT, FONT_SCALE, ERROR_COLOR, FONT_THICKNESS)
            except Exception as frame_err:
                logger.error(f"Frame {frame_count}: Unexpected error processing frame: {frame_err}", exc_info=True)
                status_text = PROCESSING_ERROR_TEXT
                cv2.putText(display_frame, status_text, (10, 30), FONT, FONT_SCALE, ERROR_COLOR, FONT_THICKNESS)

            # --- 3f. Display General Status (if no faces were detected) ---
            if detected_count == 0 and status_text == NO_FACE_TEXT and not db_embeddings:
                # If DB is empty AND no faces detected, don't overwrite DB_EMPTY_TEXT
                pass
            elif detected_count == 0 and status_text == NO_FACE_TEXT:
                cv2.putText(display_frame, status_text, (10, 30), FONT, FONT_SCALE, INFO_COLOR, FONT_THICKNESS)

            # --- 3g. Display Frame ---
            cv2.imshow("Live Face Search (Press 'q' to quit)", display_frame)

            # --- 3h. Check for Exit Key ---
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Exit key 'q' pressed. Stopping live search.")
                break

    except KeyboardInterrupt:
        logger.info("Live search interrupted by user (Ctrl+C).")
    finally:
        # --- 4. Cleanup ---
        end_time = time.time()
        logger.info(f"Live search loop finished. Processed {frame_count} frames in {end_time - start_time:.2f} seconds.")
        if cap.isOpened():
            cap.release()
            logger.debug("Camera released.")
        cv2.destroyAllWindows()
        logger.debug("Display windows closed.")