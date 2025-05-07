# src/face_compare_app/server/main.py
import logging
from pathlib import Path # Import Path
from contextlib import asynccontextmanager # Import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException # Import Request
from fastapi.responses import HTMLResponse # Import HTMLResponse
from fastapi.staticfiles import StaticFiles # If you add CSS/JS files later
from fastapi.templating import Jinja2Templates # Import Jinja2Templates

# Import routers from the routes module
from .routes import compare, faces, search, live
from .routes.live import live_search_db
from .. import core as core_func # Import core module
from ..core import FaceProcessor # Import the class
from .dependencies import get_database_path # Import the dependency function
from ..exceptions import ModelError # Import ModelError

logger = logging.getLogger(__name__)

# --- Setup Templating ---
# Assuming 'templates' directory is at the project root level (same as 'src')
# Adjust path if your structure is different
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent # Project Root
TEMPLATES_DIR = BASE_DIR / "templates"
logger.info(f"Looking for templates in: {TEMPLATES_DIR}")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# --- Lifespan Context Manager (Remains the same) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup: Accessing pre-configured FaceProcessor...")
    initialized_processor_instance = None
    try:
        # The global instance should have been configured by the CLI
        # by calling core_func.initialize_global_processor()
        if core_func._face_processor_instance is None:
            # This case should ideally not happen if CLI called initialize_global_processor.
            # It might happen if Uvicorn is run directly on main:app without the CLI wrapper.
            logger.error(
                "CRITICAL: Global FaceProcessor instance is None at startup. "
                "It should have been initialized by the CLI. Attempting default initialization."
            )
            # Fallback to default initialization if not set by CLI (e.g., direct Uvicorn run)
            # This ensures the app can still try to start.
            core_func.initialize_global_processor() # Call with defaults

        # Now, _face_processor_instance should exist (or an error was raised by initialize_global_processor)
        if core_func._face_processor_instance:
            # Trigger the actual model loading (lazy load) by accessing .app
            _ = core_func._face_processor_instance.app
            initialized_processor_instance = core_func._face_processor_instance
            logger.info(
                f"Face processor '{initialized_processor_instance.model_name}' (pre-configured by CLI or default) "
                "is initialized and ready."
            )
        else:
            logger.critical("CRITICAL: FaceProcessor instance is still None after attempting initialization.")
        app.state.face_processor = initialized_processor_instance

        # --- Pre-load search DB ---
        logger.info("Pre-loading search database...")
        # Get the database path using the dependency logic
        # NOTE: Cannot use Depends() here, call the function directly if simple
        # Or pass config/path via app state if more complex
        db_path = get_database_path() # Make sure this function is accessible
        live_search_db.load(db_path)
        logger.info(f"Pre-loaded live search DB: {len(live_search_db.embeddings)} faces in {live_search_db.load_time:.3f}s.")
    except ModelError as e:
        logger.critical(f"CRITICAL - Model Initialization Error during startup (lifespan): {e}", exc_info=True)
        app.state.face_processor = None
    except Exception as e:
        logger.critical(f"CRITICAL - Unexpected Error during startup lifespan phase: {e}", exc_info=True)
        app.state.face_processor = None
    
    # --- Lifespan execution pauses ---
    yield
    # --- Shutdown logic ---
    logger.info("Application shutdown: Cleaning up resources...")
    app.state.face_processor = None
    logger.info("Face processor cleared from app state.")

# --- Create FastAPI instance with the lifespan manager ---
app = FastAPI(
    title="Face Comparison API",
    description="API for comparing faces, managing a face database, and performing live searches.",
    version="0.1.0",
    lifespan=lifespan # Assign the lifespan context manager
)

# --- Include Routers ---
# The prefix="/api/v1" is applied within each router file now
app.include_router(compare.router)
app.include_router(faces.router)
app.include_router(search.router)
app.include_router(live.router) 
# Includes both WebSocket endpoints

# --- Root Endpoint & HTML Test Page Routes ---
@app.get("/", response_class=HTMLResponse, tags=["Root & Test Pages"])
async def read_root(request: Request):
    """Serves the main info page with links to test pages."""
    logger.info("Root endpoint '/' accessed.")
    # Simple HTML with links, or use Jinja template if preferred
    html_content = """
    <html>
        <head><title>Face Compare API</title></head>
        <body>
            <h1>Welcome to the Face Comparison API</h1>
            <p>See <a href="/docs">/docs</a> for API documentation.</p>
            <h2>Test Pages:</h2>
            <ul>
                <li><a href="/test/live-compare">Live Comparison Test Page</a></li>
                <li><a href="/test/live-search">Live Search Test Page</a></li>
            </ul>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/test/live-compare", response_class=HTMLResponse, tags=["Root & Test Pages"])
async def get_live_compare_test_page(request: Request):
    """Serves the HTML test page for the live comparison WebSocket."""
    logger.info("Serving live compare test page.")
    return templates.TemplateResponse("live_compare_test.html", {"request": request})

@app.get("/test/live-search", response_class=HTMLResponse, tags=["Root & Test Pages"])
async def get_live_search_test_page(request: Request):
    """Serves the HTML test page for the live search WebSocket."""
    logger.info("Serving live search test page.")
    return templates.TemplateResponse("live_search_test.html", {"request": request})

