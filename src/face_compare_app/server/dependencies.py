# src/face_compare_app/server/dependencies.py
import logging
# Import both Request and WebSocket for type hinting or potential dual use
from fastapi import Request, WebSocket, HTTPException
from ..core import FaceProcessor # Import the class

logger = logging.getLogger(__name__)

# Keep the original one for HTTP routes if needed elsewhere, or modify it
async def get_initialized_processor_http(request: Request) -> FaceProcessor:
    """Dependency for HTTP routes to get the processor instance."""
    logger.debug(f"get_initialized_processor_http called. Request object type: {type(request)}")
    processor = getattr(request.app.state, "face_processor", None)
    if processor is None:
        logger.error("Face processor not available in app state (initialization likely failed during startup).")
        raise HTTPException(status_code=503, detail="Face processing service is unavailable (initialization failed).")
    return processor

# Create a new dependency specifically for WebSocket routes
async def get_initialized_processor_ws(websocket: WebSocket) -> FaceProcessor:
    """Dependency for WebSocket routes to get the processor instance."""
    logger.debug(f"get_initialized_processor_ws called. WebSocket object: {websocket}")
    # Access app state via websocket.app.state
    processor = getattr(websocket.app.state, "face_processor", None)
    if processor is None:
        logger.error("Face processor not available in app state (initialization likely failed during startup).")
        # For WebSockets, raising HTTPException doesn't work directly to close the connection cleanly before accept.
        # We might need to handle this differently, maybe by letting it raise an attribute error
        # or by checking in the websocket handler itself after getting the dependency.
        # For now, let's raise an error that will likely cause the connection to fail.
        raise ValueError("Face processor service is unavailable (initialization failed).")
    return processor

# --- Optional: Database Path Dependency ---
# This doesn't depend on Request or WebSocket, so it can be used by both.
from pathlib import Path
DEFAULT_DB_PATH = Path("data/face_db.sqlite")
def get_database_path() -> Path:
    db_path = DEFAULT_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Using database path: {db_path}")
    return db_path