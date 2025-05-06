# src/face_compare_app/cli.py

import logging
from typing import Optional
from pathlib import Path
import typer
import sys
import os

# Add src directory to Python path to allow absolute imports
# This is one way, alternative is using package structure properly
# For development, running `python -m face_compare_app.cli ...` from src/ is better
# Or installing the package using `pip install -e .`
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Use relative imports assuming standard package execution
from . import core as core_func
from . import database as db_func
from . import live as live_func
from . import server as server_func
from . import utils # Import utils
from .exceptions import (
    FaceCompareError, InvalidInputError, ImageLoadError, DatabaseError,
    NoFaceFoundError, MultipleFacesFoundError, ModelError, EmbeddingError
) # Import custom exceptions

# --- Logger Configuration ---
# Basic configuration for demonstration. Consider using a more robust setup (e.g., file logging).
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("face_compare_app.cli")
# Set level for specific loggers if needed (e.g., reduce verbosity of lower levels)
logging.getLogger("face_compare_app.core").setLevel(logging.INFO)
logging.getLogger("face_compare_app.database").setLevel(logging.INFO)
logging.getLogger("face_compare_app.live").setLevel(logging.INFO)
logging.getLogger("face_compare_app.server").setLevel(logging.INFO)
logging.getLogger("face_compare_app.utils").setLevel(logging.INFO)
# --- End Logger Configuration ---


app = typer.Typer(help="A CLI application for face comparison tasks.")

@app.command()
def hello(name: Optional[str] = typer.Argument(None, help="The name to say hello to.")):
    """Simple greeting command"""
    if name:
        greeting = f"Hello {name}!"
        logger.info(greeting)
        print(greeting)
    else:
        greeting = "Hello World!"
        logger.info(greeting)
        print(greeting)


@app.command()
def compare(
    img1: str = typer.Argument(..., help="Path to the first image file."),
    img2: str = typer.Argument(..., help="Path to the second image file.")
):
    """Compares faces found in two images and prints the similarity score."""
    logger.info(f"CLI: Received compare command for '{img1}' and '{img2}'")
    try:
        similarity = core_func.compare_faces(img1, img2)
        if similarity is not None:
            print(f"Similarity Score: {similarity:.4f}")
            logger.info(f"Comparison successful. Similarity: {similarity:.4f}")
        else:
            # Specific errors should have been logged in core.compare_faces
            print("Comparison could not be completed. Check logs for details.")
            # Exit with non-zero status to indicate partial failure
            raise typer.Exit(code=1)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"Error: File not found - {e}")
        raise typer.Exit(code=1)
    except FaceCompareError as e: # Catch specific app errors
        logger.error(f"Face comparison error: {e}", exc_info=False) # Log only message
        print(f"Error: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True) # Log full traceback
        print(f"An unexpected error occurred. Check logs.")
        raise typer.Exit(code=1)

@app.command()
def live(
    camera: int = typer.Option(0, "--camera-id", "-c", help="ID of the camera device to use."),
    reference: str = typer.Option(..., "--ref", "-r", help="Path to the reference face image.")
):
    """Performs real-time face comparison against a reference image using a camera."""
    logger.info(f"CLI: Received live command with camera ID {camera} and reference '{reference}'")

    if core_func._face_processor_instance is None:
        logger.error("Face processor is not available (failed to initialize?). Cannot run live comparison.")
        print("Error: Face processing engine could not be initialized. Check logs for details.")
        raise typer.Exit(code=1)

    try:
        # Call the actual live function, passing the processor instance
        live_func.run_live_comparison(
            processor=core_func._face_processor_instance,
            camera_id=camera,
            reference_image_path=reference
            # Optional: Add CLI option for threshold and pass it here:
            # similarity_threshold=threshold_value
        )
        logger.info("Live comparison finished successfully.")
        print("\nLive comparison finished.")

    # Handle errors specific to reference image processing or initial setup
    except FileNotFoundError as e:
        logger.error(f"Reference image file not found: {e}")
        print(f"Error: Reference image not found - {e}")
        raise typer.Exit(code=1)
    except (ImageLoadError, NoFaceFoundError, MultipleFacesFoundError, ModelError, EmbeddingError) as e:
        # These errors come from processing the reference image via get_single_face_embedding
        logger.error(f"Error processing reference image '{reference}': {e}", exc_info=False)
        print(f"Error processing reference image: {e}")
        raise typer.Exit(code=1)
    # Handle camera opening error
    except RuntimeError as e:
        logger.error(f"Camera runtime error: {e}")
        print(f"Error: {e}")
        raise typer.Exit(code=1)
    # Handle other generic comparison errors
    except FaceCompareError as e:
        logger.error(f"Live comparison error: {e}", exc_info=False)
        print(f"Error: {e}")
        raise typer.Exit(code=1)
    # Catch unexpected errors during the live loop or setup
    except Exception as e:
        logger.error(f"An unexpected error occurred during live comparison: {e}", exc_info=True)
        print(f"An unexpected error occurred. Check logs for details: {e}")
        raise typer.Exit(code=1)

@app.command()
def insert(
    # Use Path type hint for Typer to handle conversion and validation
    db: Path = typer.Option(..., "--db", "-d", help="Path to the feature database file (SQLite). Will be created if it doesn't exist.", writable=True), # writable=True checks permissions somewhat
    id: str = typer.Option(..., "--id", help="Unique ID for the user/face. This will be the primary key."),
    name: str = typer.Option(..., "--name", "-n", help="Name associated with the face."),
    image: Path = typer.Option(..., "--img", "-i", help="Path to the face image file.", exists=True, file_okay=True, readable=True), # exists=True, readable=True adds checks
    meta: Optional[str] = typer.Option(None, "--meta", "-m", help="User metadata as a JSON string (e.g., '{\"dept\": \"IT\", \"role\": \"dev\"}').")
):
    """Extracts features from an image and inserts or replaces the face info in the database."""
    logger.info(f"CLI: Received insert command: ID='{id}', Name='{name}', DB='{db}', Image='{image}'")

    if core_func._face_processor_instance is None:
        logger.error("Face processor is not available. Cannot extract features.")
        print("Error: Face processing engine could not be initialized. Check logs.")
        raise typer.Exit(code=1)

    try:
        # 1. Parse metadata JSON string (optional)
        # This function raises InvalidInputError if JSON is malformed
        logger.debug(f"Parsing metadata string: {meta}")
        metadata_dict = utils.parse_metadata(meta)
        if metadata_dict:
            logger.debug(f"Parsed metadata: {metadata_dict}")

        # 2. Extract features from the image
        # This function handles image loading, face detection (expecting exactly one),
        # and embedding generation. It raises specific errors on failure.
        logger.info(f"Extracting features from image: {image}")
        # Pass the Path object directly to extract_features if it accepts it,
        # otherwise convert: str(image)
        # Let's assume extract_features handles Path object correctly (modify if needed)
        features_bytes = core_func.extract_features(str(image)) # core.extract_features expects str path
        logger.info(f"Successfully extracted features ({len(features_bytes)} bytes) for image: {image}")

        # 3. Add to database
        # This function handles DB connection, initialization, and insertion/replacement.
        logger.info(f"Adding face to database: {db}")
        db_func.add_face_to_db(
            db_path=db, # Pass the Path object
            user_id=id,
            name=name,
            features=features_bytes,
            metadata=metadata_dict # Pass the parsed dictionary
        )
        print(f"Successfully inserted/updated face ID '{id}' (Name: '{name}') into database '{db}'.")
        logger.info(f"Database insertion successful for ID='{id}'.")

    # --- Specific Error Handling ---
    except FileNotFoundError as e:
        # This is caught if the image path provided doesn't exist *before* core.extract_features is called
        # (Typer's exists=True check should catch this first, but good to have).
        # Or if core.extract_features raises it for some reason (less likely if Typer checks).
        logger.error(f"Input file not found: {e}")
        print(f"Error: Input file not found - {e}")
        raise typer.Exit(code=1)
    except ImageLoadError as e:
        # Raised by core.extract_features -> utils.load_image
        logger.error(f"Failed to load image '{image}': {e}", exc_info=False)
        print(f"Error: Failed to load image '{image}': {e.message}")
        raise typer.Exit(code=1)
    except NoFaceFoundError as e:
        # Raised by core.extract_features
        logger.error(f"No face detected in image '{image}': {e}", exc_info=False)
        print(f"Error: No face detected in image '{image}'. Cannot insert.")
        raise typer.Exit(code=1)
    except MultipleFacesFoundError as e:
        # Raised by core.extract_features
        logger.error(f"Multiple faces detected in image '{image}': {e}", exc_info=False)
        print(f"Error: Multiple faces found in '{image}'. Please provide an image with exactly one face for insertion.")
        raise typer.Exit(code=1)
    except ModelError as e:
        # Raised by core.extract_features (model loading/inference issue)
        logger.error(f"Model error during feature extraction: {e}", exc_info=False)
        print(f"Error: A model processing error occurred: {e.message}")
        raise typer.Exit(code=1)
    except EmbeddingError as e:
        # Raised by core.extract_features
        logger.error(f"Embedding error for image '{image}': {e}", exc_info=False)
        print(f"Error: Failed to generate features for the face in '{image}': {e.message}")
        raise typer.Exit(code=1)
    except InvalidInputError as e:
        # Raised by utils.parse_metadata
        logger.error(f"Invalid metadata JSON provided: {e}", exc_info=False)
        print(f"Error: Invalid format for metadata - {e.message}")
        print("Please provide metadata as a valid JSON string, e.g., '{\"key\": \"value\"}'")
        raise typer.Exit(code=1)
    except DatabaseError as e:
        # Raised by db_func.add_face_to_db
        logger.error(f"Database error during insertion for ID '{id}': {e}", exc_info=False)
        print(f"Error: Database operation failed - {e.message}")
        raise typer.Exit(code=1)
    except FaceCompareError as e:
        # Catch any other custom errors from the package that might have been missed
        logger.error(f"An application error occurred: {e}", exc_info=False)
        print(f"Error: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        # Catch any truly unexpected errors
        logger.error(f"An unexpected error occurred during insertion: {e}", exc_info=True)
        print(f"An unexpected error occurred. Please check the application logs for details.")
        raise typer.Exit(code=1)

# --- Add search command (placeholder or implementation) ---
@app.command()
def search(
    db: Path = typer.Option(..., "--db", "-d", help="Path to the feature database file (SQLite).", exists=True, file_okay=True, readable=True),
    image: Path = typer.Option(..., "--img", "-i", help="Path to the query face image file.", exists=True, file_okay=True, readable=True),
    threshold: Optional[float] = typer.Option(None, "--threshold", "-t", help="Optional similarity threshold (0.0 to 1.0). Overrides default.")
):
    """Searches the database for faces similar to the face in the query image."""
    logger.info(f"CLI: Received search command: DB='{db}', Image='{image}', Threshold='{threshold}'")

    if core_func._face_processor_instance is None:
        logger.error("Face processor is not available. Cannot extract features for search.")
        print("Error: Face processing engine could not be initialized. Check logs.")
        raise typer.Exit(code=1)

    try:
        # 1. Extract features from the query image
        logger.info(f"Extracting features from query image: {image}")
        query_features = core_func.extract_features(str(image))
        logger.info(f"Successfully extracted features from query image.")

        # 2. Load face data from the database
        logger.info(f"Loading face database from: {db}")
        face_database = db_func.get_all_faces_from_db(db)
        if not face_database:
            print(f"Database '{db}' is empty. Cannot perform search.")
            logger.warning(f"Search skipped: Database '{db}' is empty.")
            raise typer.Exit(code=0) # Not an error, just no data

        # 3. Perform the search using core function
        logger.info(f"Searching database ({len(face_database)} entries)...")
        # Pass threshold if provided, otherwise core.search_similar_face uses its internal default
        # Note: core.search_similar_face needs modification to accept an optional threshold
        # For now, let's assume it uses a default or we modify it later.
        # We'll need to update core.py's search_similar_face signature & logic.
        # Let's just call it without threshold for now. It uses a hardcoded 0.5 threshold.
        best_match = core_func.search_similar_face(
            target_features=query_features,
            face_database=face_database
            # threshold=threshold # Add this if core.search_similar_face is updated
        )

        # 4. Print results
        if best_match:
            matched_id, matched_name, similarity_score = best_match
            print("\n--- Best Match Found ---")
            print(f"  ID:         {matched_id}")
            print(f"  Name:       {matched_name}")
            print(f"  Similarity: {similarity_score:.4f}")
            # Optionally load and show metadata if needed:
            # find the matching entry in face_database and parse metadata_str
            # meta_str = next((entry[3] for entry in face_database if entry[0] == matched_id), None)
            # if meta_str:
            #     try:
            #         meta_dict = json.loads(meta_str)
            #         print(f"  Metadata:   {json.dumps(meta_dict, indent=2)}")
            #     except json.JSONDecodeError:
            #         print(f"  Metadata:   (Invalid JSON in DB: {meta_str})")

        else:
            print("\nNo similar face found in the database matching the criteria.")
            logger.info("Search completed. No match found above threshold.")

    # --- Error Handling specific to Search ---
    except FileNotFoundError as e:
        logger.error(f"Input file not found: {e}")
        print(f"Error: Input file not found - {e}")
        raise typer.Exit(code=1)
    except (ImageLoadError, NoFaceFoundError, MultipleFacesFoundError, ModelError, EmbeddingError) as e:
        # Errors during query image feature extraction
        logger.error(f"Error processing query image '{image}': {e}", exc_info=False)
        print(f"Error processing query image '{image}': {e.message}")
        raise typer.Exit(code=1)
    except DatabaseError as e:
        # Errors during loading data from DB
        logger.error(f"Database error during search: {e}", exc_info=False)
        print(f"Error: Database operation failed - {e.message}")
        raise typer.Exit(code=1)
    except InvalidInputError as e:
        # e.g., if target feature size mismatch in search_similar_face
        logger.error(f"Invalid input for search: {e}", exc_info=False)
        print(f"Error: {e.message}")
        raise typer.Exit(code=1)
    except FaceCompareError as e:
        # Catch any other custom errors
        logger.error(f"An application error occurred during search: {e}", exc_info=False)
        print(f"Error: {e.message}")
        raise typer.Exit(code=1)
    except Exception as e:
        # Catch any truly unexpected errors
        logger.error(f"An unexpected error occurred during search: {e}", exc_info=True)
        print(f"An unexpected error occurred. Please check the application logs for details.")
        raise typer.Exit(code=1)


@app.command()
def server(
    port: int = typer.Option(8080, "--port", "-p", help="Port number for the API server."),
    workers: int = typer.Option(4, "--workers", "-w", help="Number of worker processes for the server.")
):
    """Starts the REST API server (placeholder)."""
    logger.info(f"CLI: Received server command with port={port}, workers={workers}")
    try:
        server_func.start_server(port=port, workers=workers)
        # Note: The placeholder start_server function will return immediately.
        # A real server would likely run indefinitely until stopped (e.g., Ctrl+C).
        print("\nPlaceholder server function finished. See logs for details.")
        logger.info("Placeholder server function execution completed.")
    except Exception as e:
        logger.error(f"Failed to start server (placeholder): {e}", exc_info=True)
        print(f"An unexpected error occurred while trying to start the server. Check logs.")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()