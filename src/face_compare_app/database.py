# src/face_compare_app/database.py
"""Database interaction functions."""
import logging
import sqlite3 # Using SQLite
import json # <-- Added: Needed for metadata serialization
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path # Good practice to use Path objects
from datetime import datetime, timezone

# Relative imports
from .exceptions import DatabaseError

logger = logging.getLogger(__name__)

def _get_connection(db_path: Path) -> sqlite3.Connection:
    db_path_str = str(db_path.resolve())
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path_str, timeout=10, check_same_thread=True)
        conn.row_factory = sqlite3.Row
        logger.debug(f"Connected to database: {db_path_str}")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to database {db_path_str}: {e}")
        raise DatabaseError(f"Could not connect to database: {e}")

def initialize_db(db_path: Path):
    db_path_str = str(db_path)
    logger.info(f"Initializing database schema if needed: {db_path_str}")
    try:
        with _get_connection(db_path) as conn:
            cursor = conn.cursor()

            # 1. CREATE THE TABLE FIRST (if it doesn't exist)
            # This version of the table already includes model_name and updated_at
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS faces (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    features BLOB NOT NULL,
                    metadata TEXT,
                    model_name TEXT NOT NULL DEFAULT 'unknown',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit() # Commit after table creation

            # 2. NOW, handle potential schema migration for 'model_name' and 'updated_at'
            #    for databases created with an older schema.
            cursor.execute("PRAGMA table_info(faces)")
            columns = [column['name'] for column in cursor.fetchall()]

            if 'model_name' not in columns:
                logger.info("Migrating schema: Adding 'model_name' column to 'faces' table.")
                cursor.execute("ALTER TABLE faces ADD COLUMN model_name TEXT NOT NULL DEFAULT 'unknown'")
                conn.commit() # Commit after altering
                logger.info("'model_name' column added.")
            
            if 'updated_at' not in columns:
                logger.info("Migrating schema: Adding 'updated_at' column to 'faces' table.")
                # For existing rows, updated_at can be set to created_at or current time
                cursor.execute("ALTER TABLE faces ADD COLUMN updated_at TIMESTAMP")
                cursor.execute("UPDATE faces SET updated_at = created_at WHERE updated_at IS NULL")
                conn.commit() # Commit after altering and updating
                logger.info("'updated_at' column added and populated for existing rows.")

            # Create indexes (safe to call multiple times)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_faces_name ON faces(name);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_faces_model_name ON faces(model_name);")
            conn.commit() # Final commit for indexes

            logger.debug("Database schema checked/created/updated successfully.")
    except sqlite3.Error as e:
        logger.error(f"Error initializing/updating database schema for {db_path_str}: {e}", exc_info=True) # Add exc_info for more details
        raise DatabaseError(f"Could not initialize/update database schema: {e}")


def add_face_to_db(db_path: Path, face_id: str, name: str, features: bytes, model_name: str, metadata: Optional[Dict[str, Any]] = None):
    """Adds or replaces a face record in the database."""
    # face_id is now generated outside and passed in
    db_path_str = str(db_path)
    logger.info(f"Adding/updating face in DB '{db_path_str}': ID='{face_id}', Name='{name}', Model='{model_name}'")
    metadata_str: Optional[str] = None
    if metadata:
        try:
            metadata_str = json.dumps(metadata)
        except TypeError as e:
            raise DatabaseError(f"Metadata for ID='{face_id}' is not JSON serializable: {e}")

    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        initialize_db(db_path)
        with _get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO faces (id, name, features, metadata, model_name, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (face_id, name, features, metadata_str, model_name, now_iso, now_iso) # created_at is set on insert
            )
            conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"Successfully added/updated face ID='{face_id}' (Model: {model_name}) in DB.")
            else:
                logger.warning(f"No rows affected for face ID='{face_id}'.")
    except sqlite3.Error as e:
        raise DatabaseError(f"Could not add/update face ID='{face_id}': {e}")

def update_face_details(db_path: Path, face_id: str, name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Updates a face's name and/or metadata."""
    if name is None and metadata is None:
        return False

    fields_to_update = []
    params = []
    if name is not None:
        fields_to_update.append("name = ?")
        params.append(name)
    if metadata is not None: # Client sends {} for empty, or actual content
        fields_to_update.append("metadata = ?")
        params.append(json.dumps(metadata) if metadata else None) # Store {} as "{}", None as NULL

    fields_to_update.append("updated_at = ?")
    params.append(datetime.now(timezone.utc).isoformat())
    params.append(face_id)

    sql = f"UPDATE faces SET {', '.join(fields_to_update)} WHERE id = ?"
    try:
        with _get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, tuple(params))
            conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"Details for face ID='{face_id}' updated.")
                return True
            logger.warning(f"Face ID='{face_id}' not found for update.")
            return False # Not found or no change
    except sqlite3.Error as e:
        raise DatabaseError(f"Could not update face ID='{face_id}': {e}")


def get_face_by_id(db_path: Path, face_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a single face record by ID."""
    try:
        with _get_connection(db_path) as conn:
            cursor = conn.cursor()
            # Include model_name and updated_at
            cursor.execute("SELECT id, name, metadata, model_name, created_at, updated_at FROM faces WHERE id = ?", (face_id,))
            row = cursor.fetchone()
            if row:
                data = dict(row)
                data['metadata'] = json.loads(row['metadata']) if row['metadata'] else None
                # For PersonResponse, we need 'embeddings_info' structure
                # Since we have one feature per row with model_name directly, adapt
                data['embeddings_info'] = [{"model_name": data["model_name"], "created_at": data["created_at"]}]
                data['person_id'] = data['id'] # Alias for Pydantic model
                return data
            return None
    except sqlite3.Error as e:
        raise DatabaseError(f"Could not retrieve face ID='{face_id}': {e}")

def get_all_faces(db_path: Path, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """Retrieves all face records (paginated), just basic info + model_name."""
    results = []
    try:
        with _get_connection(db_path) as conn:
            cursor = conn.cursor()
            # Fetch necessary fields
            cursor.execute("""
                SELECT id, name, metadata, model_name, created_at, updated_at 
                FROM faces 
                ORDER BY created_at DESC 
                LIMIT ? OFFSET ?
            """, (limit, offset))
            for row in cursor.fetchall():
                data = dict(row)
                data['metadata'] = json.loads(row['metadata']) if row['metadata'] else None
                # Adapt for PersonResponse model
                data['embeddings_info'] = [{"model_name": data["model_name"], "created_at": data["created_at"]}]
                data['person_id'] = data['id'] # Alias for Pydantic model
                results.append(data)
            return results
    except sqlite3.Error as e:
        raise DatabaseError(f"Could not retrieve all faces: {e}")

# This function is used by current search. Needs to consider model_name
def get_all_faces_from_db(db_path: Path, model_name_filter: Optional[str] = None) -> List[Tuple[str, str, bytes, Optional[str]]]:
    """
    Retrieves (id, name, features_blob, metadata_json_str) for search.
    Can filter by model_name.
    """
    logger.info(f"Retrieving faces from DB: {db_path} (Model Filter: {model_name_filter or 'Any'})")
    results = []
    sql = "SELECT id, name, features, metadata FROM faces"
    params = []
    if model_name_filter:
        sql += " WHERE model_name = ?"
        params.append(model_name_filter)
    
    try:
        initialize_db(db_path) # Ensure table/column exists
        with _get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
            for row in rows:
                results.append((row['id'], row['name'], row['features'], row['metadata']))
            logger.info(f"Retrieved {len(results)} face records for search (model: {model_name_filter or 'Any'}).")
            return results
    except sqlite3.Error as e:
        raise DatabaseError(f"Could not retrieve faces from database for search: {e}")

# TODO: Add delete_face_by_id(db_path, face_id)