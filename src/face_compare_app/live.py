# src/face_compare_app/live.py
"""Live camera face comparison logic."""
import logging
import time
from typing import Optional

from .core import extract_features
from .exceptions import ImageLoadError, NoFaceFoundError

logger = logging.getLogger(__name__)

def run_live_comparison(camera_id: int, reference_image_path: str):
    """
    Placeholder for running live face comparison using a camera.
    """
    logger.info(f"Starting live comparison using camera ID: {camera_id}")
    logger.info(f"Reference image: {reference_image_path}")

    try:
        # 1. Load reference image and extract its features
        logger.info("Loading reference image and extracting features...")
        reference_features = extract_features(reference_image_path)
        logger.info("Reference features extracted successfully.")

        # --- Placeholder Logic for Live Loop ---
        print(f"Placeholder: Initializing camera {camera_id}...")
        # In real code: cap = cv2.VideoCapture(camera_id)
        # if not cap.isOpened(): raise RuntimeError("Cannot open camera")

        print("Placeholder: Starting live feed loop (press Ctrl+C to stop)...")
        frame_count = 0
        while True:
            frame_count += 1
            print(f"\n--- Frame {frame_count} ---")
            # In real code: ret, frame = cap.read()
            # if not ret: break
            print("Placeholder: Captured frame from camera.")

            try:
                # Simulate processing the captured frame
                print("Placeholder: Detecting face in frame...")
                # Simulate finding a face
                print("Placeholder: Extracting features from detected face...")
                live_features = b'\x06\x07\x08...' # Dummy features

                # Simulate comparing live features with reference features
                # In real code: similarity = calculate_similarity(live_features, reference_features)
                import random
                similarity = random.uniform(0.4, 0.95)
                print(f"Placeholder: Comparing with reference. Similarity: {similarity:.2f}")

                threshold = 0.6 # Example threshold
                if similarity >= threshold:
                    print(f"MATCH FOUND! Similarity: {similarity:.2f} >= {threshold:.2f}")
                    logger.info(f"Match found in frame {frame_count}, Similarity: {similarity:.2f}")
                else:
                    print(f"No match. Similarity: {similarity:.2f} < {threshold:.2f}")

                # In real code: Display the frame with bounding boxes, similarity score etc. using cv2.imshow()
                print("Placeholder: Displaying frame...")
                # if cv2.waitKey(1) & 0xFF == ord('q'): break # Exit condition

            except NoFaceFoundError:
                print("No face detected in the current frame.")
                logger.debug("No face detected in frame.")
                # In real code: Display the frame without detection results
                print("Placeholder: Displaying frame...")
            except Exception as e:
                logger.error(f"Error processing frame: {e}", exc_info=True)
                print(f"Error processing frame: {e}")
                # Optionally break or continue

            time.sleep(1) # Simulate processing time / frame rate

        # In real code: cap.release(); cv2.destroyAllWindows()
        print("\nPlaceholder: Releasing camera and closing windows.")
        # --- End Placeholder ---

    except (ImageLoadError, NoFaceFoundError) as e:
        logger.error(f"Failed to start live comparison: {e}")
        print(f"ERROR: Failed to start live comparison: {e}")
    except KeyboardInterrupt:
        logger.info("Live comparison stopped by user.")
        print("\nLive comparison stopped.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during live comparison: {e}", exc_info=True)
        print(f"ERROR: An unexpected error occurred: {e}")