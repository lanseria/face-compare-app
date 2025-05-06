# src/face_compare_app/server.py
"""Web server (REST API) logic."""
import logging
import uvicorn # Import uvicorn

logger = logging.getLogger(__name__)

# The FastAPI app instance is expected to be importable from server.main
# Adjust the import path if your structure differs slightly
APP_MODULE_STR = "face_compare_app.server.main:app"

def start_server(host: str = "0.0.0.0", port: int = 8080, workers: int = 4, reload: bool = False):
    """
    Starts the FastAPI server using Uvicorn.
    """
    logger.info(f"Attempting to start API server on {host}:{port} with {workers} workers.")
    logger.info(f"Reloading enabled: {reload}")
    logger.info(f"Uvicorn target app: {APP_MODULE_STR}")

    # Use uvicorn.run to start the server
    uvicorn.run(
        APP_MODULE_STR,
        host=host,
        port=port,
        workers=workers,
        reload=reload, # Enable auto-reload for development if needed
        log_level="info" # Set uvicorn's log level
        # You might need loop='uvloop' or loop='asyncio' depending on your system/performance needs
        # Lifespan 'on' is default and needed for startup/shutdown events
    )

    # Note: uvicorn.run will block until the server is stopped (e.g., Ctrl+C)
    logger.info("Uvicorn server process has finished.") # This line might only be reached after stopping