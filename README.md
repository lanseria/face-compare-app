# Face Compare App

A minimal Python command-line application template using Typer and Rich.

## Features

- Basic CLI structure with Typer
- Rich console output
- Python project configuration (pyproject.toml)
- Test setup with pytest

## Installation

```bash
# Create virtual environment
python3 -m venv myenv
python3.12 -m venv myenv
source myenv/bin/activate  # macOS/Linux
# .\myenv\Scripts\Activate.ps1  # Windows

# Install dependencies
pip install -e .
```

## Usage

```bash
cli hello [name]
cli compare data/img2.jpeg data/img3.jpeg
cli live --ref data/img3.jpeg --camera-id 0
cli insert \
    --db data/face_db.sqlite \
    --id img003_zc \
    --name "zc" \
    --img data/img4.jpeg \
    --meta '{"source": "upload", "capture_date": "2024-05-06"}'
cli search \
    --db data/face_db.sqlite \
    --img data/img4.jpeg
cli live-search --camera-id 0 --db data/face_db.sqlite
cli live-search -c 1 -d data/face_db.sqlite -t 0.65
cli server --port 8080 --model-name buffalo_s --det-size-w 320 --det-size-h 320
cli server --port 8080 \
    --model-name buffalo_s \
    --provider CUDAExecutionProvider \
    --det-size-w 320 \
    --det-size-h 320 \
    --det-thresh 0.4
```

## Development

Run tests:
```bash
pytest
```

## Project Structure

```
src/
  face_compare_app/
    __init__.py    # Package initialization
    cli.py         # CLI commands
    main.py        # App entry point
tests/             # Test cases
pyproject.toml     # Project configuration
```

## License

MIT

## 参考

https://github.com/deepinsight/insightface
https://github.com/deepinsight/insightface/tree/master/model_zoo