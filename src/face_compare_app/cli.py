# src/face_compare_app/cli.py

import logging
from typing import Optional
import typer
import sys
import os

# Add src directory to Python path to allow absolute imports
# This is one way, alternative is using package structure properly
# For development, running `python -m face_compare_app.cli ...` from src/ is better
# Or installing the package using `pip install -e .`
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Use relative imports assuming standard package execution
from . import core
from . import database
from . import live
from . import server
from . import utils # Import utils
from .exceptions import FaceCompareError, InvalidInputError # Import custom exceptions

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
        similarity = core.compare_faces(img1, img2)
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
    try:
        live.run_live_comparison(camera_id=camera, reference_image_path=reference)
        logger.info("Live comparison finished.")
    except FileNotFoundError as e:
        logger.error(f"Reference image file not found: {e}")
        print(f"Error: Reference image not found - {e}")
        raise typer.Exit(code=1)
    except FaceCompareError as e:
        logger.error(f"Live comparison error: {e}", exc_info=False)
        print(f"Error: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during live comparison: {e}", exc_info=True)
        print(f"An unexpected error occurred. Check logs.")
        raise typer.Exit(code=1)


@app.command()
def insert(
    db: str = typer.Option(..., "--db", "-d", help="Path to the feature database file (SQLite)."),
    id: str = typer.Option(..., "--id", help="Unique ID for the user/face."),
    name: str = typer.Option(..., "--name", "-n", help="Name of the user."),
    image: str = typer.Option(..., "--img", "-i", help="Path to the user's face image file."),
    meta: Optional[str] = typer.Option(None, "--meta", "-m", help="User metadata as a JSON string (e.g., '{\"dept\": \"IT\"}').")
):
    """Extracts features from an image and inserts the face info into the database."""
    logger.info(f"CLI: Received insert command for ID='{id}', Name='{name}', DB='{db}', Image='{image}'")
    try:
        # 1. Parse metadata JSON string (optional)
        metadata_dict = utils.parse_metadata(meta) # Use the utility function

        # 2. Extract features from the image
        logger.info(f"Extracting features for image: {image}")
        features = core.extract_features(image) # Can raise NoFaceFoundError, ImageLoadError

        # 3. Add to database
        logger.info(f"Adding face to database: {db}")
        database.add_face_to_db(
            db_path=db,
            user_id=id,
            name=name,
            features=features,
            metadata=metadata_dict # Pass the parsed dictionary
        )
        print(f"Successfully inserted/updated face ID '{id}' ('{name}') into database '{db}'.")
        logger.info(f"Insertion successful for ID='{id}'.")

    except FileNotFoundError as e:
        logger.error(f"Image file not found: {e}")
        print(f"Error: Image file not found - {e}")
        raise typer.Exit(code=1)
    except InvalidInputError as e: # Catch bad JSON format
        logger.error(f"Invalid metadata format: {e}", exc_info=False)
        print(f"Error: {e}")
        raise typer.Exit(code=1)
    except FaceCompareError as e: # Catch NoFaceFound, DB errors, etc.
        logger.error(f"Failed to insert face: {e}", exc_info=False)
        print(f"Error: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during insertion: {e}", exc_info=True)
        print(f"An unexpected error occurred. Check logs.")
        raise typer.Exit(code=1)


@app.command()
def search(
    db: str = typer.Option(..., "--db", "-d", help="Path to the feature database file (SQLite)."),
    target: str = typer.Option(..., "--target", "-t", help="Path to the target face image to search for.")
):
    """Searches for a similar face in the database based on the target image."""
    logger.info(f"CLI: Received search command for target='{target}' in DB='{db}'")
    try:
        # 1. Extract features from the target image
        logger.info(f"Extracting features from target image: {target}")
        target_features = core.extract_features(target) # Can raise NoFaceFoundError, ImageLoadError

        # 2. Get all faces from the database
        logger.info(f"Loading faces from database: {db}")
        all_faces = database.get_all_faces_from_db(db)

        if not all_faces:
            print(f"Database '{db}' is empty or does not exist.")
            logger.warning(f"Search aborted: Database '{db}' is empty or could not be read.")
            raise typer.Exit()

        # 3. Perform the search
        logger.info(f"Searching for matches in {len(all_faces)} database entries.")
        best_match = core.search_similar_face(target_features, all_faces)

        # 4. Report results
        if best_match:
            match_id, match_name, match_score = best_match
            print("\n--- Best Match Found ---")
            print(f"  ID:         {match_id}")
            print(f"  Name:       {match_name}")
            print(f"  Similarity: {match_score:.4f}")
            logger.info(f"Search successful. Best match: ID='{match_id}', Score={match_score:.4f}")
        else:
            print("\nNo similar face found in the database meeting the threshold.")
            logger.info("Search completed. No sufficiently similar face found.")

    except FileNotFoundError as e:
        logger.error(f"Target image file not found: {e}")
        print(f"Error: Target image file not found - {e}")
        raise typer.Exit(code=1)
    except FaceCompareError as e: # Catch NoFaceFound, DB errors, etc.
        logger.error(f"Failed to search face: {e}", exc_info=False)
        print(f"Error: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during search: {e}", exc_info=True)
        print(f"An unexpected error occurred. Check logs.")
        raise typer.Exit(code=1)


@app.command()
def server(
    port: int = typer.Option(8080, "--port", "-p", help="Port number for the API server."),
    workers: int = typer.Option(4, "--workers", "-w", help="Number of worker processes for the server.")
):
    """Starts the REST API server (placeholder)."""
    logger.info(f"CLI: Received server command with port={port}, workers={workers}")
    try:
        server.start_server(port=port, workers=workers)
        # Note: The placeholder start_server function will return immediately.
        # A real server would likely run indefinitely until stopped (e.g., Ctrl+C).
        print("\nPlaceholder server function finished. See logs for details.")
        logger.info("Placeholder server function execution completed.")
    except Exception as e:
        logger.error(f"Failed to start server (placeholder): {e}", exc_info=True)
        print(f"An unexpected error occurred while trying to start the server. Check logs.")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    # Create the main application entry point
    # Consider moving this logic to a separate main.py or __main__.py
    # for better structure if the application grows complex.
    app()