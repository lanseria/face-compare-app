# src/face_compare_app/server/routes/faces.py
import logging
import time
import json
import tempfile
import shutil
import uuid # For generating ID
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends, Query, Body, Response, status # Added Response, status


# Updated models
from ..models import FaceInsertResponse, FaceUpdateResponse, PersonResponse, FaceInsertData, FaceUpdateData
from ... import core as core_func
from ... import database as db_func
from ...exceptions import (
    ImageLoadError, NoFaceFoundError, MultipleFacesFoundError,
    ModelError, EmbeddingError, DatabaseError, InvalidInputError
)
from ...core import FaceProcessor # ACTIVE_MODELS_FOR_EMBEDDING not needed if one model per insert
from ..dependencies import get_initialized_processor_http, get_database_path

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/faces", tags=["Faces Management"]) # Renamed tag

# --- POST /api/v1/faces (New face record) ---
@router.post("", response_model=FaceInsertResponse, status_code=201)
async def api_create_face_entry(
    image: UploadFile = File(..., description="Image file containing the face."),
    name: Optional[str] = Form(None, description="Name associated with the person/face."),
    meta: Optional[str] = Form(None, description="Optional metadata as a JSON string."),
    processor: FaceProcessor = Depends(get_initialized_processor_http),
    db_path: Path = Depends(get_database_path)
):
    face_id_str = str(uuid.uuid4())
    model_name_used = processor.model_name # Get model name from the loaded processor
    logger.info(f"Request to create face. Generated ID: {face_id_str}, Name: {name}, Model: {model_name_used}")

    metadata_dict = None
    if meta:
        try:
            metadata_dict = json.loads(meta)
            if not isinstance(metadata_dict, dict): raise ValueError("JSON Object required")
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=422, detail=f"Invalid JSON metadata: {e}")

    temp_file_path: Optional[str] = None
    try:
        suffix = Path(image.filename or ".tmp").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file_path = temp_file.name
            try: await image.seek(0); shutil.copyfileobj(image.file, temp_file)
            finally: image.file.close()

        features_bytes = core_func.extract_features(temp_file_path) # Uses the configured processor
        
        db_func.add_face_to_db(
            db_path,
            face_id=face_id_str,
            name=name or f"Face_{face_id_str[:8]}", # Default name
            features=features_bytes,
            model_name=model_name_used, # Store the model name
            metadata=metadata_dict
        )
        return FaceInsertResponse(
            id=uuid.UUID(face_id_str),
            name=name or f"Face_{face_id_str[:8]}",
            model_name=model_name_used,
            message="Face entry created successfully."
        )
    except (ImageLoadError, NoFaceFoundError, MultipleFacesFoundError, EmbeddingError, ModelError) as e:
        logger.error(f"Feature extraction failed for new face {face_id_str}: {e}")
        raise HTTPException(status_code=400, detail=f"Feature extraction failed: {e.message}")
    except DatabaseError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e.message}")
    except Exception as e:
        logger.error(f"Unexpected error creating face '{face_id_str}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")
    finally:
        if temp_file_path and Path(temp_file_path).exists(): Path(temp_file_path).unlink()


# --- PUT /api/v1/faces/{face_id} (Update name, meta, or image/features) ---
@router.put("/{face_id}", response_model=FaceUpdateResponse)
async def api_update_face_entry(
    face_id: str, # Path parameter (UUID string)
    name: Optional[str] = Form(None),
    meta: Optional[str] = Form(None), # JSON string or "" to clear
    image: Optional[UploadFile] = File(None),
    processor: FaceProcessor = Depends(get_initialized_processor_http),
    db_path: Path = Depends(get_database_path)
):
    logger.info(f"Request to update face ID '{face_id}'. Name: {name}, Image provided: {image is not None}")

    existing_face_data = db_func.get_face_by_id(db_path, face_id)
    if not existing_face_data:
        raise HTTPException(status_code=404, detail=f"Face with ID '{face_id}' not found.")

    # Update name/metadata
    metadata_dict_to_update: Optional[Dict[str, Any]] = None
    if meta is not None: # meta Form field was provided
        if meta == "": metadata_dict_to_update = {} # Clear
        else:
            try:
                metadata_dict_to_update = json.loads(meta)
                if not isinstance(metadata_dict_to_update, dict): raise ValueError("JSON Object")
            except (json.JSONDecodeError, ValueError) as e:
                raise HTTPException(status_code=422, detail=f"Invalid JSON metadata: {e}")
    
    db_func.update_face_details(db_path, face_id, name=name, metadata=metadata_dict_to_update)

    features_were_updated = False
    if image:
        logger.info(f"New image for face '{face_id}'. Re-extracting features with model '{processor.model_name}'...")
        temp_file_path: Optional[str] = None
        try:
            suffix = Path(image.filename or ".tmp").suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file_path = temp_file.name
                try: await image.seek(0); shutil.copyfileobj(image.file, temp_file)
                finally: image.file.close()
            
            new_features_bytes = core_func.extract_features(temp_file_path)
            # This will REPLACE the existing record due to INSERT OR REPLACE in add_face_to_db
            # We need the current name and metadata if they weren't part of this request's form data
            current_name = name if name is not None else existing_face_data['name']
            current_meta_dict = metadata_dict_to_update if meta is not None else existing_face_data['metadata']

            db_func.add_face_to_db(
                db_path,
                face_id=face_id,
                name=current_name, # Use new or existing name
                features=new_features_bytes,
                model_name=processor.model_name, # Features are from current processor
                metadata=current_meta_dict # Use new or existing meta
            )
            features_were_updated = True
            logger.info(f"Features for face ID '{face_id}' updated using model '{processor.model_name}'.")
        except (ImageLoadError, NoFaceFoundError, MultipleFacesFoundError, EmbeddingError, ModelError) as e:
            raise HTTPException(status_code=400, detail=f"Feature re-extraction failed: {e.message}")
        except DatabaseError as e:
            raise HTTPException(status_code=500, detail=f"Database error during feature update: {e.message}")
        finally:
            if temp_file_path and Path(temp_file_path).exists(): Path(temp_file_path).unlink()

    # Fetch the final state of the record
    updated_face_data = db_func.get_face_by_id(db_path, face_id)
    if not updated_face_data:
        raise HTTPException(status_code=500, detail="Failed to retrieve updated face data.")

    return FaceUpdateResponse(
        id=uuid.UUID(updated_face_data['id']),
        name=updated_face_data['name'],
        metadata=updated_face_data['metadata'],
        model_name=updated_face_data['model_name'],
        updated_at=updated_face_data['updated_at'],
        message="Face entry updated successfully.",
        features_updated=features_were_updated
    )

# --- GET /api/v1/faces (List all faces) ---
@router.get("", response_model=List[PersonResponse])
async def api_list_faces(
    db_path: Path = Depends(get_database_path),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    model_name: Optional[str] = Query(None, description="Filter by model name (e.g., 'buffalo_s')") # Optional filter
):
    logger.info(f"Request to list faces. Skip: {skip}, Limit: {limit}, Model: {model_name}")
    try:
        # Modify get_all_faces to accept model_name filter if needed,
        # or filter here if get_all_faces returns all.
        # For now, assume get_all_faces doesn't filter by model, so we filter post-fetch.
        # Better: db_func.get_all_faces(db_path, limit=limit, offset=skip, model_name_filter=model_name)
        all_faces_data = db_func.get_all_faces(db_path, limit=limit, offset=skip) # Needs model_name filter in DB layer
        
        if model_name:
            all_faces_data = [f for f in all_faces_data if f.get('model_name') == model_name]
            
        return [PersonResponse(**p) for p in all_faces_data]
    except DatabaseError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e.message}")

# --- GET /api/v1/faces/{face_id} (Get a single face) ---
@router.get("/{face_id}", response_model=PersonResponse)
async def api_get_face(
    face_id: str,
    db_path: Path = Depends(get_database_path)
):
    logger.info(f"Request to get face ID '{face_id}'.")
    try:
        face_data = db_func.get_face_by_id(db_path, face_id)
        if not face_data:
            raise HTTPException(status_code=404, detail=f"Face with ID '{face_id}' not found.")
        return PersonResponse(**face_data)
    except DatabaseError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e.message}")


# --- DELETE /api/v1/faces/{face_id} (Delete a face entry) ---
@router.delete("/{face_id}", status_code=status.HTTP_204_NO_CONTENT) # Use 204 No Content for successful delete
async def api_delete_face_entry(
    face_id: str, # Path parameter (UUID string)
    db_path: Path = Depends(get_database_path)
):
    logger.info(f"Request to delete face ID '{face_id}'.")
    try:
        deleted = db_func.delete_face_by_id(db_path, face_id)
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Face with ID '{face_id}' not found.")
        # For 204 No Content, we don't return a body.
        # FastAPI handles this if the function returns None or no explicit Response.
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except DatabaseError as e:
        # Log the full error for server-side debugging
        logger.error(f"Database error during deletion of face ID '{face_id}': {e}", exc_info=True)
        # Return a generic error to the client
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error: Could not delete face entry.")
    except Exception as e:
        logger.error(f"Unexpected error deleting face ID '{face_id}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error while deleting face entry.")