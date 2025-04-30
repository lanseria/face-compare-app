# src/face_compare_app/database.py
"""Database interaction functions."""
import logging
import sqlite3 # Example using SQLite
from typing import Optional, List, Tuple, Dict, Any

from .exceptions import DatabaseError

logger = logging.getLogger(__name__)

def _get_connection(db_path: str) -> sqlite3.Connection:
    """Establishes a connection to the SQLite database."""
    try:
        # `check_same_thread=False` might be needed if accessed by multiple threads (e.g., web server)
        # but Typer CLI usually runs sequentially per command.
        conn = sqlite3.connect(db_path, check_same_thread=True)
        conn.row_factory = sqlite3.Row # Access columns by name
        logger.debug(f"Connected to database: {db_path}")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to database {db_path}: {e}")
        raise DatabaseError(f"Could not connect to database: {e}") from e

def initialize_db(db_path: str):
    """Initializes the database schema if it doesn't exist."""
    logger.info(f"Initializing database schema if needed: {db_path}")
    try:
        with _get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS faces (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    features BLOB NOT NULL,
                    metadata TEXT, -- Store JSON as TEXT
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            logger.debug("Database schema checked/created successfully.")
    except sqlite3.Error as e:
        logger.error(f"Error initializing database schema: {e}")
        raise DatabaseError(f"Could not initialize database schema: {e}") from e

def add_face_to_db(db_path: str, user_id: str, name: str, features: bytes, metadata: Optional[Dict[str, Any]] = None):
    """Adds a face record to the database."""
    logger.info(f"Adding face to DB: ID='{user_id}', Name='{name}'")
    metadata_str = json.dumps(metadata) if metadata else None
    try:
        # Ensure DB is initialized before adding
        initialize_db(db_path) # Safe to call multiple times

        with _get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO faces (id, name, features, metadata) VALUES (?, ?, ?, ?)",
                (user_id, name, features, metadata_str)
            )
            conn.commit()
            logger.info(f"Successfully added/updated face ID='{user_id}' in the database.")
    except sqlite3.Error as e:
        logger.error(f"Error adding face ID='{user_id}' to database: {e}")
        raise DatabaseError(f"Could not add face to database: {e}") from e
    except Exception as e: # Catch other potential errors like JSON serialization
        logger.error(f"Unexpected error adding face ID='{user_id}': {e}", exc_info=True)
        raise DatabaseError(f"Unexpected error adding face: {e}") from e


def get_all_faces_from_db(db_path: str) -> List[Tuple[str, str, bytes, Optional[str]]]:
    """Retrieves all face records from the database."""
    logger.info(f"Retrieving all faces from database: {db_path}")
    try:
        # Ensure DB exists, even if empty
        initialize_db(db_path)

        with _get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, features, metadata FROM faces")
            rows = cursor.fetchall()
            # Convert rows to the desired tuple format
            result = [(row['id'], row['name'], row['features'], row['metadata']) for row in rows]
            logger.info(f"Retrieved {len(result)} face records from the database.")
            return result
    except sqlite3.Error as e:
        logger.error(f"Error retrieving faces from database: {e}")
        raise DatabaseError(f"Could not retrieve faces from database: {e}") from e

# Need to import json at the top of database.py
import json