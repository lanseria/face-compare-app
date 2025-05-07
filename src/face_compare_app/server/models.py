# src/face_compare_app/server/models.py
from pydantic import BaseModel, Field, UUID4
from typing import List, Optional, Dict, Any, Tuple

# --- Request Models (Mostly handled by Form/Query/File, but useful for clarity/complex bodies) ---
# None needed for current specifications as inputs are simple types via Form/Query/File

# --- Response Models ---

class CompareResponse(BaseModel):
    """Response model for the /compare endpoint."""
    similarity: Optional[float] = Field(None, description="Cosine similarity score (0.0 to 1.0)")
    is_match: Optional[bool] = Field(None, description="Whether the similarity meets the threshold")
    elapsed_ms: int = Field(..., description="Processing time in milliseconds")
    error: Optional[str] = Field(None, description="Error message if comparison failed") # Added for error reporting

class FaceInsertResponse(BaseModel):
    """Response model for the /faces endpoint."""
    face_id: str = Field(..., description="The unique ID provided for the face")
    feature_size: Optional[int] = Field(None, description="Dimension of the extracted feature vector (e.g., 512)")
    message: str = Field("Face data received successfully.", description="Status message")
    error: Optional[str] = Field(None, description="Error message if insertion failed") # Added for error reporting

class SearchResultItem(BaseModel):
    """Individual search result item."""
    face_id: str = Field(..., description="Matched face ID from the database")
    name: Optional[str] = Field(None, description="Name associated with the matched face ID")
    similarity: float = Field(..., description="Cosine similarity score with the query face")
    meta: Optional[Dict[str, Any]] = Field(None, description="Metadata associated with the matched face")

class SearchResponse(BaseModel):
    """Response model for the /search endpoint."""
    results: List[SearchResultItem] = Field(..., description="List of matching faces found")
    search_time_ms: int = Field(..., description="Time taken for the search operation in milliseconds")
    error: Optional[str] = Field(None, description="Error message if search failed") # Added for error reporting

# --- WebSocket Message Models (Used for structuring WS communication) ---

class LiveCompareWSResponse(BaseModel):
    """WebSocket message model for live comparison."""
    status: str = Field(..., description="Status like 'processing', 'match_found', 'no_match', 'no_face', 'error'")
    similarity: Optional[float] = Field(None, description="Similarity score if a face is found")
    is_match: Optional[bool] = Field(None, description="Match status if a face is found")
    reference_id: Optional[str] = Field(None, description="The reference ID being compared against")
    message: Optional[str] = Field(None, description="Optional message (e.g., for errors)")
    # --- ADDED FOR FACE BOXES ---
    detection_box: Optional[List[int]] = Field(None, description="Bounding box [x1, y1, x2, y2] of the primary detected face (if status involves a single face)")
    all_detection_boxes: Optional[List[List[int]]] = Field(None, description="List of bounding boxes if multiple_faces status")
    # --- END ADDED ---

# Keep LiveSearchMatchDetail as it is:
class LiveSearchMatchDetail(BaseModel):
    face_id: str
    name: Optional[str]
    similarity: float
    meta: Optional[Dict[str, Any]]

# --- Models for Multi-Face Live Search ---
class LiveSearchSingleFaceResult(BaseModel):
    """Result for a single detected face in a live search frame."""
    status: str = Field(..., description="'match_found', 'no_match_found', 'error_embedding'")
    match_detail: Optional[LiveSearchMatchDetail] = Field(None, description="Details if a match was found")
    detection_box: List[int] = Field(..., description="Bounding box [x1, y1, x2, y2] of this detected face")
    message: Optional[str] = Field(None, description="Error message specific to this face's processing")

class MultiLiveSearchWSResponse(BaseModel):
    """WebSocket message model for live search, supporting multiple faces per frame."""
    # General frame status (optional, could be inferred from faces_results)
    # frame_status: str = Field("processed", description="'processed', 'error_decoding', 'no_faces_in_frame'")
    faces_results: List[LiveSearchSingleFaceResult] = Field(default_factory=list, description="Results for each detected face in the frame")
    processed_frame_timestamp_ms: int = Field(..., description="Server timestamp when frame processing finished")
    processing_time_ms: int = Field(..., description="Total time taken by the server to process this frame and generate the response")
    # Global error message if frame processing itself failed before face detection
    frame_error_message: Optional[str] = Field(None, description="Error message if the entire frame processing failed")


class EmbeddingInfo(BaseModel):
    model_name: str
    created_at: str # Or datetime object

class PersonResponse(BaseModel): # This will represent a row from the 'faces' table
    person_id: UUID4 = Field(..., alias="id") # Alias 'id' from DB to 'person_id'
    name: Optional[str]
    metadata: Optional[Dict[str, Any]]
    created_at: str
    updated_at: str
    # Since each row in 'faces' has one feature and one model_name:
    embeddings_info: List[EmbeddingInfo] # Will contain a single item

class FaceInsertData(BaseModel): # For request body if not using Form for all
    name: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    # image is UploadFile

class FaceInsertResponse(BaseModel):
    id: UUID4 # The generated ID of the face record
    name: Optional[str]
    model_name: str
    message: str

class FaceUpdateData(BaseModel): # For request body of PUT
    name: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    # image is UploadFile

class FaceUpdateResponse(BaseModel):
    id: UUID4
    name: Optional[str]
    metadata: Optional[Dict[str, Any]]
    model_name: str # Include current model_name
    updated_at: str
    message: str
    features_updated: bool = False # Flag if features were changed