# src/face_compare_app/database.py
"""Database interaction functions."""
import logging
import sqlite3 # Using SQLite
import json # <-- Added: Needed for metadata serialization
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path # Good practice to use Path objects

# Relative imports
from .exceptions import DatabaseError

logger = logging.getLogger(__name__)

def _get_connection(db_path: Path) -> sqlite3.Connection: # Use Path object
    """Establishes a connection to the SQLite database."""
    db_path_str = str(db_path.resolve())
    try:
        # Ensure the directory exists before trying to connect
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # `check_same_thread=False` might be needed if accessed by multiple threads (e.g., web server)
        # but Typer CLI usually runs sequentially per command. Keep True for now.
        conn = sqlite3.connect(db_path_str, timeout=10, check_same_thread=True) # Added timeout
        conn.row_factory = sqlite3.Row # Access columns by name
        logger.debug(f"Connected to database: {db_path_str}")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to database {db_path_str}: {e}")
        raise DatabaseError(f"Could not connect to database: {e}") from e

def initialize_db(db_path: Path): # Use Path object
    """Initializes the database schema if it doesn't exist."""
    db_path_str = str(db_path)
    logger.info(f"Initializing database schema if needed: {db_path_str}")
    try:
        with _get_connection(db_path) as conn:
            cursor = conn.cursor()
            # Added some basic indexing for potentially faster lookups if needed later
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS faces (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    features BLOB NOT NULL,
                    metadata TEXT, -- Store JSON as TEXT
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_faces_name ON faces(name);") # Index on name
            conn.commit()
            logger.debug("Database schema checked/created successfully.")
    except sqlite3.Error as e:
        logger.error(f"Error initializing database schema for {db_path_str}: {e}")
        raise DatabaseError(f"Could not initialize database schema: {e}") from e

def add_face_to_db(db_path: Path, user_id: str, name: str, features: bytes, metadata: Optional[Dict[str, Any]] = None): # Use Path object
    """Adds or replaces a face record in the database."""
    db_path_str = str(db_path)
    logger.info(f"Adding/updating face in DB '{db_path_str}': ID='{user_id}', Name='{name}'")
    metadata_str: Optional[str] = None # Explicit type hint
    if metadata:
        try:
            metadata_str = json.dumps(metadata)
        except TypeError as e:
            logger.error(f"Metadata for ID='{user_id}' is not JSON serializable: {metadata}")
            raise DatabaseError(f"Metadata is not valid JSON: {e}") from e

    try:
        # Ensure DB is initialized before adding
        initialize_db(db_path) # Safe to call multiple times

        with _get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                # Use INSERT OR REPLACE for simple upsert based on primary key (id)
                "INSERT OR REPLACE INTO faces (id, name, features, metadata) VALUES (?, ?, ?, ?)",
                (user_id, name, features, metadata_str)
            )
            conn.commit()
            if cursor.rowcount > 0:
                 logger.info(f"Successfully added/updated face ID='{user_id}' in the database '{db_path_str}'.")
            else:
                 # Should not happen with INSERT OR REPLACE unless there's a weird issue
                 logger.warning(f"No rows were affected when adding/updating face ID='{user_id}' in '{db_path_str}'.")

    except sqlite3.Error as e:
        logger.error(f"Error adding/updating face ID='{user_id}' to database {db_path_str}: {e}")
        raise DatabaseError(f"Could not add/update face to database: {e}") from e
    except Exception as e: # Catch other potential errors
        logger.error(f"Unexpected error adding/updating face ID='{user_id}' in {db_path_str}: {e}", exc_info=True)
        raise DatabaseError(f"Unexpected error adding/updating face: {e}") from e


def get_all_faces_from_db(db_path: Path) -> List[Tuple[str, str, bytes, Optional[str]]]: # Use Path object
    """Retrieves all face records from the database for search."""
    db_path_str = str(db_path)
    logger.info(f"Retrieving all faces from database: {db_path_str}")
    result: List[Tuple[str, str, bytes, Optional[str]]] = []
    try:
        # Ensure DB exists, even if empty (initializes if needed)
        initialize_db(db_path)

        with _get_connection(db_path) as conn:
            cursor = conn.cursor()
            # Select only columns needed for the search function
            cursor.execute("SELECT id, name, features, metadata FROM faces")
            rows = cursor.fetchall()
            # Convert sqlite3.Row objects to the specific tuple format expected by search_similar_face
            # Ensure 'features' (BLOB) is returned directly as bytes
            result = [(row['id'], row['name'], row['features'], row['metadata']) for row in rows]
            logger.info(f"Retrieved {len(result)} face records from the database '{db_path_str}'.")
            return result
    except sqlite3.Error as e:
        logger.error(f"Error retrieving faces from database {db_path_str}: {e}")
        raise DatabaseError(f"Could not retrieve faces from database: {e}") from e
    except Exception as e: # Catch other potential errors
        logger.error(f"Unexpected error retrieving faces from {db_path_str}: {e}", exc_info=True)
        raise DatabaseError(f"Unexpected error retrieving faces: {e}") from e

# Potential additions for the future (not strictly required by the request):
# - Function to delete a face by ID
# - Function to get a single face by ID
# - More robust error handling or connection management if used in a concurrent environment