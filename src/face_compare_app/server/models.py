# src/face_compare_app/server/models.py
from pydantic import BaseModel, Field
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

class LiveSearchMatchDetail(BaseModel):
    """Details of a match found during live search."""
    face_id: str
    name: Optional[str]
    similarity: float
    meta: Optional[Dict[str, Any]]

class LiveSearchWSResponse(BaseModel):
    """WebSocket message model for live search."""
    status: str = Field(..., description="Status like 'match_found', 'no_match_found', 'no_face_detected', 'error'")
    match: Optional[LiveSearchMatchDetail] = Field(None, description="Details of the best match found")
    detection_box: Optional[List[int]] = Field(None, description="Bounding box [x1, y1, x2, y2] of detected face (optional)")
    processed_frame_timestamp_ms: Optional[int] = Field(None, description="Server timestamp when frame was processed (optional)")
    processing_time_ms: Optional[int] = Field(None, description="Time taken by the server to process this frame and generate the response, in milliseconds.")
    message: Optional[str] = Field(None, description="Optional message (e.g., for errors)")