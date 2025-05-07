# Face Compare App

A comprehensive Python application for face comparison, recognition, and management, featuring a command-line interface (CLI) and a RESTful API server with real-time capabilities. Powered by InsightFace for state-of-the-art face analysis.

## Features

*   **Core Face Analysis:**
    *   Face Detection: Locates faces in images.
    *   Face Feature Extraction: Generates unique embeddings (features) for faces using configurable InsightFace models (e.g., `buffalo_l`, `buffalo_s`).
    *   Face Comparison: Calculates similarity between two faces.
*   **Command-Line Interface (CLI):**
    *   Compare faces in two images.
    *   Extract and store face features in an SQLite database.
    *   Search for similar faces in the database against a query image.
    *   Live face comparison against a reference image using a camera.
    *   Live face search against the database using a camera.
    *   Start the API server with configurable model parameters.
*   **RESTful API Server (FastAPI):**
    *   **CRUD Operations for Faces:**
        *   `POST /api/v1/faces`: Create a new face entry (generates UUID, extracts features, stores name, metadata, model used).
        *   `GET /api/v1/faces`: List all stored face entries (with pagination and model filtering).
        *   `GET /api/v1/faces/{face_id}`: Retrieve details of a specific face entry.
        *   `PUT /api/v1/faces/{face_id}`: Update name, metadata, or re-extract features with a new image.
        *   `DELETE /api/v1/faces/{face_id}`: Delete a face entry.
    *   **Face Comparison API:**
        *   `POST /api/v1/compare`: Compare two uploaded images and get similarity.
    *   **Database Search API:**
        *   `POST /api/v1/search`: Search the database for faces similar to an uploaded query image.
    *   **Real-time WebSocket Endpoints:**
        *   `GET /api/v1/live/compare/ws`: Live face comparison against a reference ID from the database via WebSocket.
        *   `GET /api/v1/live/search/ws`: Live multi-face search against the database via WebSocket.
    *   Interactive API documentation (Swagger UI & ReDoc) available at `/docs` and `/redoc`.
*   **Configurable Models:** Supports different InsightFace models (e.g., `buffalo_l`, `buffalo_s`) and detection parameters, configurable at server startup.
*   **SQLite Database:** Stores face IDs, names, features, metadata, and the model used for feature extraction.
*   **Project Structure:** Organized with Typer for CLI and FastAPI for the web server.

## Prerequisites

*   Python 3.8+
*   An environment where `onnxruntime` (and `onnxruntime-gpu` if using CUDA) can be installed.
*   OpenCV (`opencv-python`)
*   Access to a camera for live features.

## Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone https://github.com/lanseria/face-compare-app
    cd face-compare-app
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Recommended: Use the Python version you intend to deploy with (e.g., 3.10, 3.11, 3.12)
    python3 -m venv myenv
    source myenv/bin/activate  # On macOS/Linux
    # myenv\Scripts\activate.bat   # On Windows Command Prompt
    # myenv\Scripts\Activate.ps1 # On Windows PowerShell
    ```

3.  **Install dependencies:**
    *   For CPU-based inference:
        ```bash
        pip install -e .
        ```
    *   For GPU-based inference (NVIDIA GPU with CUDA):
        First, ensure you have the correct NVIDIA drivers and CUDA Toolkit installed. Then:
        ```bash
        pip install -e .[gpu] 
        # Or: pip install -e . && pip install onnxruntime-gpu
        ```
        *(You might need to adjust `pyproject.toml` to define the `[gpu]` extra if it's not already there, e.g., `onnxruntime-gpu`)*

## Configuration

*   **Database:** The application uses an SQLite database. The default path is `data/face_database.db` (for server operations) or specified via CLI arguments for CLI commands.
*   **Face Recognition Models:** The server can be configured to use different InsightFace models via CLI arguments when starting the server (see `cli server --help`).

## Usage

The application is named `face-compare-app` when installed. You can invoke its CLI commands or start the server.

### CLI Commands

*   **Say Hello:**
    ```bash
    cli hello "Your Name"
    ```
*   **Compare two images:**
    ```bash
    cli compare path/to/image1.jpg path/to/image2.jpg
    ```
*   **Insert a face into the database (CLI manages DB path):**
    *(Note: The API `POST /faces` now auto-generates IDs. This CLI command might need updating if it requires a user-provided ID for the old DB structure.)*
    ```bash
    cli insert --db data/cli_face_db.sqlite --id <unique_id> --name "Person Name" --img path/to/face.jpg --meta '{"department":"IT"}'
    ```
*   **Search for a face in the database (CLI manages DB path):**
    ```bash
    face-compare_app search --db data/cli_face_db.sqlite --img path/to/query_face.jpg [--threshold 0.6]
    ```
*   **Live comparison using camera:**
    ```bash
    cli live --ref path/to/reference_face.jpg [--camera-id 0]
    ```
*   **Live search using camera (CLI manages DB path):**
    ```bash
    cli live-search --db data/cli_face_db.sqlite [--camera-id 0] [--threshold 0.55]
    ```
*   **Start the API Server:**
    *   With default settings (CPU, `buffalo_l` model):
        ```bash
        cli server --port 8080
        ```
    *   With `buffalo_s` model on CPU, specific detection parameters:
        ```bash
        cli server --port 8080 --model-name buffalo_s --det-size-w 320 --det-size-h 320 --det-thresh 0.4
        ```
    *   With GPU (CUDAExecutionProvider) and `buffalo_s` model:
        ```bash
        face_compare_app server --port 8080 --model-name buffalo_s --provider CUDAExecutionProvider --det-size-w 320 --det-size-h 320
        ```
    See `cli server --help` for all options.

### API Server Endpoints

Once the server is running (e.g., `cli server --port 8080`), you can access:

*   **Interactive API Docs (Swagger):** `http://localhost:8080/docs`
*   **Alternative API Docs (ReDoc):** `http://localhost:8080/redoc`
*   **Test Pages (if implemented):**
    *   `http://localhost:8080/test/crud` (for Faces CRUD)
    *   `http://localhost:8080/test/live-compare`
    *   `http://localhost:8080/test/live-search`

**Key API Endpoints:**

*   `POST /api/v1/faces`: Add a new face.
    *   Body: `image` (file), `name` (form), `meta` (form, JSON string).
*   `GET /api/v1/faces`: List faces.
    *   Query params: `skip`, `limit`, `model_name`.
*   `GET /api/v1/faces/{face_id}`: Get specific face details.
*   `PUT /api/v1/faces/{face_id}`: Update face (name, meta, or new image for features).
    *   Body: `name` (form, optional), `meta` (form, optional JSON string), `image` (file, optional).
*   `DELETE /api/v1/faces/{face_id}`: Delete a face.
*   `POST /api/v1/compare`: Compare two images.
    *   Body: `image1` (file), `image2` (file), `threshold` (form, optional).
*   `POST /api/v1/search`: Search database for a face.
    *   Body: `image` (file), `top_k` (form, optional), `threshold` (form, optional).
*   `WS /api/v1/live/compare/ws?reference_id=<id>`: WebSocket for live comparison.
*   `WS /api/v1/live/search/ws`: WebSocket for live multi-face search.

## Development

*   **Setup:** Follow the installation steps.
*   **Run Tests:**
    ```bash
    pytest
    ```
*   **Linting/Formatting:** (Consider adding tools like Black, Flake8, isort)
    ```bash
    # Example with Black
    # pip install black
    # black src tests
    ```

## Project Structure


.
├── data/ # Default directory for SQLite databases
├── src/
│ └── face_compare_app/
│ ├── init.py
│ ├── cli.py # Typer CLI application logic
│ ├── core.py # Core face processing functions (InsightFace wrapper)
│ ├── database.py # SQLite database interactions
│ ├── exceptions.py # Custom application exceptions
│ ├── server/ # FastAPI server components
│ │ ├── init.py
│ │ ├── dependencies.py # FastAPI dependencies (e.g., get_processor)
│ │ ├── main.py # FastAPI app instantiation, lifespan, root routes
│ │ ├── models.py # Pydantic models for API requests/responses
│ │ └── routes/ # API route modules
│ │ ├── init.py
│ │ ├── compare.py
│ │ ├── faces.py
│ │ ├── live_compare_ws.py
│ │ ├── live_search_ws.py
│ │ └── search.py
│ └── utils.py # Utility functions
├── templates/ # HTML templates for test pages
│ ├── faces_crud_test.html
│ ├── live_compare_test.html
│ └── live_search_test.html
├── tests/ # Pytest test cases
├── pyproject.toml # Project metadata and dependencies (using Poetry or Hatch)
├── README.md # This file
└── ... (other config files like .gitignore)

## Future Enhancements / TODO

*(This section can list items from your previous discussion or new ideas)*
*   Implement robust multi-model feature storage (e.g., separate `persons` and `face_embeddings` tables).
*   Allow generating features for multiple specified models during a single face insertion.
*   Full test coverage for API endpoints.
*   Configuration management (e.g., using Pydantic Settings, .env file).
*   Authentication and authorization for sensitive API endpoints.
*   More advanced database search filtering (e.g., by metadata fields).
*   Integration with vector databases (Faiss, Milvus) for very large-scale search.
*   Liveness detection integration.
*   Face quality assessment during enrollment.
*   Asynchronous task queue (Celery/RQ) for long-running operations.

## License

MIT

## References

*   **InsightFace:** [https://github.com/deepinsight/insightface](https://github.com/deepinsight/insightface)
*   **InsightFace Model Zoo:** [https://github.com/deepinsight/insightface/tree/master/model_zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo)
*   **FastAPI:** [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
*   **Typer:** [https://typer.tiangolo.com/](https://typer.tiangolo.com/)