# src/face_compare_app/core.py
"""Core face processing functions."""
import logging
from pathlib import Path
from typing import Optional, Any, List, Tuple, Dict
import time
import numpy as np
from insightface.app import FaceAnalysis


# Relative imports from the package
from .exceptions import (
    FaceCompareError,
    NoFaceFoundError,
    MultipleFacesFoundError,
    ImageLoadError,
    ModelError,
    EmbeddingError,
    InvalidInputError
)
from .utils import load_image

logger = logging.getLogger(__name__)


# --- FaceProcessor Class Definition ---
class FaceProcessor:
    """Face comparison processor with InsightFace backend"""

    def __init__(self, model_name: str = "buffalo_l", providers: list = None, config: dict = None):
        if FaceAnalysis is None:
            raise ModelError("Insightface library is not installed or loadable.")

        self.model_name = model_name
        # Default to CPUExecutionProvider if none specified
        self.providers = providers or ["CPUExecutionProvider"]
        self.config = config or {}
        self._app: Optional[FaceAnalysis] = None # Type hint for clarity

        # Initialize with lazy loading
        self._initialized = False
        logger.info(f"FaceProcessor created for model '{model_name}' with providers {self.providers} (lazy init).")

    def _initialize(self):
        """Lazy initialization of face analysis model"""
        if self._initialized:
            return
        logger.info(f"Initializing InsightFace model '{self.model_name}'...")
        start_time = time.time()
        try:
            self._app = FaceAnalysis(
                name=self.model_name,
                providers=self.providers,
                allowed_modules=['detection', 'recognition'] # Only load necessary modules
            )
            # Apply configuration defaults if not provided
            det_size = self.config.get("det_size", (640, 640))
            det_thresh = self.config.get("det_thresh", 0.5)
            logger.debug(f"Preparing model with det_size={det_size}, det_thresh={det_thresh}")
            self._app.prepare(
                ctx_id=0, # Use 0 for CPU or the first GPU, -1 forces CPU but ctx_id=0 with CPUExecutionProvider should work.
                det_size=det_size,
                det_thresh=det_thresh
            )
            self._initialized = True
            logger.info(f"InsightFace model '{self.model_name}' initialized successfully ({time.time() - start_time:.2f}s).")
        except Exception as e:
            logger.error(f"InsightFace model initialization failed: {e}", exc_info=True)
            # Use the specific ModelError exception
            raise ModelError(f"Model initialization failed: {str(e)}")

    @property
    def app(self) -> FaceAnalysis:
        """Get initialized face analysis application"""
        if not self._initialized:
            self._initialize() # Trigger lazy loading
        if self._app is None: # Check if initialization failed
             raise ModelError("FaceAnalysis application is not available (initialization failed).")
        return self._app

    def get_faces(self, image: np.ndarray) -> list:
        """Detect faces and extract basic info + embeddings from an image."""
        logger.debug(f"Detecting faces and extracting features from image with shape {image.shape}")
        try:
            faces = self.app.get(image)
            logger.debug(f"Detected {len(faces)} faces.")
            # You might want to add checks here if embeddings are missing, etc.
            for face in faces:
                if face.normed_embedding is None:
                     logger.warning(f"Detected face (bbox: {face.bbox}) has no embedding.")
                     # Decide how to handle this - raise error? Filter out?
            return faces # Return the list of Face objects from insightface
        except Exception as e:
            logger.error(f"Face detection/embedding failed: {e}", exc_info=True)
            raise ModelError(f"Face detection/processing failed: {str(e)}")

    def get_single_face_embedding(self, image_path: Path) -> np.ndarray:
        """Loads image, detects faces, ensures exactly one face, returns its embedding."""
        img_data = load_image(image_path) # Can raise FileNotFoundError, ImageLoadError
        faces = self.get_faces(img_data) # Can raise ModelError

        if len(faces) == 0:
            logger.error(f"No faces detected in image: {image_path}")
            raise NoFaceFoundError(f"No faces detected in image: {image_path}")
        if len(faces) > 1:
            logger.error(f"Multiple ({len(faces)}) faces detected in image: {image_path}")
            raise MultipleFacesFoundError(len(faces), f"Expected 1 face, found {len(faces)} in: {image_path}")

        embedding = faces[0].normed_embedding
        if embedding is None:
             logger.error(f"Failed to generate embedding for the face in {image_path}")
             raise EmbeddingError(f"Failed to generate embedding for the face in {image_path}")

        logger.debug(f"Successfully extracted single face embedding for {image_path}")
        return embedding

    def compare(self, image1_path: Path, image2_path: Path) -> Dict[str, Any]:
        """Compare two face images and return detailed results"""
        logger.info(f"Starting comparison between '{image1_path}' and '{image2_path}'")
        start_time = time.time()

        try:
            # Load images and get embeddings (handles detection and single face check)
            embedding1 = self.get_single_face_embedding(image1_path)
            embedding2 = self.get_single_face_embedding(image2_path)

            # Calculate similarity
            similarity = self._cosine_similarity(embedding1, embedding2)
            logger.info(f"Similarity calculated: {similarity:.4f}")

            processing_time = time.time() - start_time
            return {
                "similarity": float(similarity),
                "processing_time_sec": round(processing_time, 4),
                "image1": str(image1_path.resolve()),
                "image2": str(image2_path.resolve()),
                "model": self.model_name,
                # Since get_single_face_embedding ensures 1 face, we know this
                "faces_detected": {"image1": 1, "image2": 1}
            }
        # Catch specific errors from loading/processing and re-raise or handle
        except (FileNotFoundError, ImageLoadError, NoFaceFoundError, MultipleFacesFoundError, ModelError, EmbeddingError) as e:
            # Logged within the methods or here, re-raise the specific error
            # The calling function (compare_faces) will handle converting this to None
            logger.error(f"Comparison failed during processing: {e}")
            raise e # Re-raise the caught specific error
        except Exception as e:
            # Catch any other unexpected errors
            logger.error(f"Unexpected error during FaceProcessor.compare: {e}", exc_info=True)
            # Wrap in a generic FaceCompareError or re-raise
            raise FaceCompareError(9999, f"Unexpected comparison error: {e}") from e


    @staticmethod
    def _cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        # Ensure inputs are numpy arrays
        emb1 = np.asarray(emb1)
        emb2 = np.asarray(emb2)
        # Calculate dot product
        dot_product = np.dot(emb1, emb2)
        # Calculate norms
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        # Calculate similarity, handle potential division by zero if norm is zero
        if norm1 == 0 or norm2 == 0:
            logger.warning("Cannot calculate cosine similarity: one or both embeddings have zero norm.")
            return 0.0
        similarity = dot_product / (norm1 * norm2)
        # Clip similarity score to [-1, 1] range due to potential float precision issues
        return np.clip(similarity, -1.0, 1.0)

# --- End of FaceProcessor Class ---


# --- Module-level Instance ---
# Create a single instance to be reused (improves performance by loading model once)
try:
    # Configure processor options if needed (e.g., specify GPU providers if available)
    # providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] # Example for GPU
    # providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"] # Rank preference
    providers = ['CPUExecutionProvider'] # Default to CPU for broader compatibility
    # You could potentially load config from a file or env vars here
    _face_processor_instance = FaceProcessor(model_name="buffalo_l", providers=providers)
except Exception as e:
    # Catch potential errors during class instantiation itself (e.g., insightface not installed)
    logger.error(f"Fatal: Failed to create FaceProcessor instance: {e}", exc_info=True)
    # Mark instance as None so functions using it can fail gracefully
    _face_processor_instance = None
# --- End Module-level Instance ---


# --- Core API Functions ---

def compare_faces(img_path1: str, img_path2: str) -> Optional[float]:
    """
    Compares faces in two images using the FaceProcessor.

    Requires exactly one face to be detected in each image.

    Args:
        img_path1: Path string to the first image file.
        img_path2: Path string to the second image file.

    Returns:
        The cosine similarity score (float) if successful, otherwise None.
    """
    if _face_processor_instance is None:
        logger.error("Face comparison skipped: FaceProcessor is not available.")
        return None

    logger.info(f"Executing compare_faces for '{img_path1}' and '{img_path2}'")
    try:
        # Convert string paths to Path objects
        path1 = Path(img_path1)
        path2 = Path(img_path2)

        # Use the module-level processor instance's compare method
        result_dict = _face_processor_instance.compare(path1, path2)

        # Extract similarity score
        similarity = result_dict.get("similarity")

        # Log success details
        logger.info(f"Comparison successful. Similarity: {similarity:.4f}")
        logger.debug(f"Comparison details: {result_dict}")

        # Return the similarity score as a float
        return float(similarity)

    # Handle specific, expected errors from the processor or file loading
    except FileNotFoundError as e:
        logger.error(f"Comparison failed: Image file not found - {e}")
        return None
    except ImageLoadError as e:
        logger.error(f"Comparison failed: {e.message} (Code: {e.code})")
        return None
    except NoFaceFoundError as e:
        logger.error(f"Comparison failed: {e.message} (Code: {e.code})")
        return None
    except MultipleFacesFoundError as e:
        logger.error(f"Comparison failed: {e.message} (Code: {e.code})")
        return None
    except ModelError as e:
        logger.error(f"Comparison failed due to model issue: {e.message} (Code: {e.code})")
        return None
    except EmbeddingError as e:
         logger.error(f"Comparison failed due to embedding issue: {e.message} (Code: {e.code})")
         return None
    except FaceCompareError as e: # Catch other specific FaceCompareErrors
        logger.error(f"Comparison failed: {e.message} (Code: {e.code})")
        return None
    # Catch any other unexpected errors
    except Exception as e:
        logger.error(f"An unexpected error occurred during comparison: {e}", exc_info=True)
        return None


# --- Placeholder functions (TODO: Implement using FaceProcessor) ---

def extract_features(image_path: str) -> bytes:
    """
    Extracts face features (embedding) from a single face in an image.

    Args:
        image_path: Path string to the image file.

    Returns:
        The face embedding as bytes.

    Raises:
        FileNotFoundError, ImageLoadError, NoFaceFoundError, MultipleFacesFoundError,
        ModelError, EmbeddingError: If processing fails.
        TypeError: If FaceProcessor is not available.
    """
    if _face_processor_instance is None:
        logger.error("Feature extraction skipped: FaceProcessor is not available.")
        # Or raise an error appropriate for the caller (CLI vs Server)
        raise TypeError("FaceProcessor is not initialized, cannot extract features.")

    logger.info(f"Executing extract_features for '{image_path}'")
    try:
        img_path_obj = Path(image_path)
        # Use the processor method that ensures a single face
        embedding_np = _face_processor_instance.get_single_face_embedding(img_path_obj)

        # Serialize numpy array to bytes
        feature_bytes = embedding_np.tobytes()
        logger.info(f"Successfully extracted features for '{image_path}' ({len(feature_bytes)} bytes).")
        return feature_bytes

    # Let specific errors propagate up to the caller (e.g., CLI command)
    # which can then handle them (print message, exit code)
    except (FileNotFoundError, ImageLoadError, NoFaceFoundError, MultipleFacesFoundError, ModelError, EmbeddingError) as e:
         logger.error(f"Feature extraction failed: {e}")
         raise e # Re-raise for the caller to handle
    except Exception as e:
        logger.error(f"An unexpected error occurred during feature extraction: {e}", exc_info=True)
        # Wrap in a generic error or re-raise
        raise FaceCompareError(9999, f"Unexpected feature extraction error: {e}") from e


def search_similar_face(target_features: bytes, face_database: List[Tuple[str, str, bytes, Optional[str]]]) -> Optional[Tuple[str, str, float]]:
    """
    Searches the database for the face most similar to the target features.

    Args:
        target_features: Feature embedding (bytes) of the face to search for.
        face_database: List of tuples: (id, name, feature_bytes, metadata_str).

    Returns:
        Tuple (matched_id, matched_name, similarity_score) for the best match
        above a threshold, or None.
    """
    if _face_processor_instance is None:
        logger.error("Face search skipped: FaceProcessor is not available.")
        return None
    if not face_database:
         logger.warning("Face search skipped: Database is empty.")
         return None

    logger.info(f"Searching for similar face among {len(face_database)} entries.")
    try:
        # Deserialize target features into numpy array
        # Assuming features are stored as float32 (common for embeddings)
        target_embedding = np.frombuffer(target_features, dtype=np.float32)
        # Check if buffer size matches expected embedding dimension (e.g., 512 for buffalo_l)
        # This depends on the model used. Let's assume 512 for buffalo_l.
        expected_dims = 512 # TODO: Make this configurable or get from model?
        if target_embedding.size != expected_dims:
             logger.error(f"Target feature size mismatch. Expected {expected_dims}, got {target_embedding.size}.")
             # Handle this error - maybe raise InvalidInputError?
             raise InvalidInputError(f"Invalid target feature size ({target_embedding.size}), expected {expected_dims}.")


        best_match = None
        highest_similarity = -1.0
        # TODO: Make threshold configurable
        similarity_threshold = 0.5 # Example threshold (adjust based on model/use case)

        start_time = time.time()
        for db_id, db_name, db_features_bytes, _ in face_database:
            try:
                # Deserialize database features
                db_embedding = np.frombuffer(db_features_bytes, dtype=np.float32)

                if db_embedding.size != expected_dims:
                     logger.warning(f"Skipping entry ID '{db_id}': Feature size mismatch (got {db_embedding.size}, expected {expected_dims}).")
                     continue # Skip this entry

                # Calculate similarity using the static method
                similarity = FaceProcessor._cosine_similarity(target_embedding, db_embedding)

                logger.debug(f"Comparing with DB ID '{db_id}' ({db_name}): Similarity={similarity:.4f}")

                if similarity > highest_similarity and similarity >= similarity_threshold:
                    highest_similarity = similarity
                    best_match = (db_id, db_name, float(similarity)) # Store as float

            except Exception as entry_err:
                # Log error for specific entry but continue search
                logger.error(f"Error processing database entry ID '{db_id}': {entry_err}", exc_info=True)
                continue # Skip to the next entry

        search_time = time.time() - start_time
        logger.info(f"Database search completed in {search_time:.4f}s.")

        if best_match:
            logger.info(f"Best match found: ID={best_match[0]}, Name={best_match[1]}, Similarity={best_match[2]:.4f}")
        else:
            logger.info("No similar face found above the threshold.")

        return best_match

    except InvalidInputError as e: # Catch the specific error from size check
         logger.error(f"Face search failed: {e}")
         raise e # Propagate error
    except Exception as e:
        logger.error(f"An unexpected error occurred during face search: {e}", exc_info=True)
        # Wrap in generic error
        raise FaceCompareError(9999, f"Unexpected face search error: {e}") from e