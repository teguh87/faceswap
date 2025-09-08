# Face Swap Advanced

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/face-swap-advanced.svg)](https://badge.fury.io/py/face-swap-advanced)

Advanced face swapping tool with anomaly detection and similarity filtering for both images and videos.

## Features

- **High-quality face swapping** using InsightFace and ONNX models
- **Anomaly detection** to filter out poor-quality faces
- **Similarity filtering** to ensure accurate face matching
- **Batch processing** for efficient video processing
- **GPU acceleration** support (CUDA)
- **Configurable thresholds** for quality control
- **Debug mode** with detailed processing information
- **Both image and video support**

## Installation

### From PyPI (Recommended)

```bash
# Basic installation
pip install face-swap-advanced

# With GPU support
pip install face-swap-advanced[gpu]

# With development tools
pip install face-swap-advanced[dev]

# Complete installation
pip install face-swap-advanced[all]
```

### From Source

```bash
# Clone the repository
git clone https://github.com/teguh87/face-swap-advanced.git
cd face-swap-advanced

# Install in development mode
pip install -e .

# Or install with extras
pip install -e .[gpu,dev]
```

### Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

## Quick Start

### Command Line Usage

```bash
# Basic face swap
face-swap --src source_face.jpg --ref reference_face.jpg --tgt target_video.mp4 --model inswapper_128.onnx

# With custom settings
face-swap --src source.jpg --ref reference.jpg --tgt target.mp4 --model model.onnx \
    --min-similarity 0.6 --device cuda:0 --debug

# Process image instead of video
face-swap --src source.jpg --ref reference.jpg --tgt target_image.jpg --model model.onnx
```

### Python API Usage

```python
from face_swap_advanced import FaceSwapConfig, FaceSwapper

# Create configuration
config = FaceSwapConfig(
    src_img_path="source_face.jpg",
    ref_img_path="reference_face.jpg", 
    tgt_path="target_video.mp4",
    model_path="inswapper_128.onnx",
    min_similarity=0.5,
    device="cuda:0"
)

# Perform face swap
result_path = FaceSwapper.face_swap(config)
print(f"Result saved to: {result_path}")
```

## Configuration Options

### Core Parameters

- `--src`: Path to source face image (face to swap in)
- `--ref`: Path to reference face image (face to match in target)
- `--tgt`: Path to target image or video
- `--model`: Path to INSwapper ONNX model
- `--out`: Output directory (default: "output")

### Quality Control

- `--min-similarity`: Minimum cosine similarity threshold (0.0-1.0, default: 0.3)
- `--max-face-ratio`: Maximum face area as ratio of frame area (default: 0.8)
- `--min-face-size`: Minimum face size in pixels (default: 50)

### Performance

- `--device`: Execution device (auto, cpu, cuda, cuda:0, etc.)
- `--batch`: Batch size for video processing (default: 8)
- `--skip`: Frame skip factor for video (default: 1)
- `--providers`: Override ONNX providers (comma-separated)

### Debug

- `--debug`: Enable detailed debug output and logging

## Model Requirements

You need to download the INSwapper ONNX model:

1. Download `inswapper_128.onnx` from the [InsightFace model zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo)
2. Place it in your working directory or specify the path with `--model`

## How It Works

1. **Face Detection**: Uses InsightFace to detect faces in all images/video frames
2. **Quality Filtering**: Applies anomaly detection to filter out low-quality faces:
   - Face size validation
   - Boundary checking  
   - Area ratio validation
   - Confidence scoring
3. **Similarity Matching**: Finds the best matching face using cosine similarity
4. **Face Swapping**: Uses INSwapper to perform the actual face replacement
5. **Post-processing**: Saves results with detailed statistics

## Performance Tips

- Use GPU acceleration with `--device cuda:0` for faster processing
- Increase batch size with `--batch 16` for better GPU utilization  
- Use frame skipping with `--skip 2` for faster video processing
- Adjust similarity threshold based on your quality requirements

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU mode
2. **No faces detected**: Check image quality and face visibility
3. **Low similarity scores**: Adjust `--min-similarity` threshold
4. **Slow processing**: Enable GPU acceleration or increase batch size

### Debug Mode

Use `--debug` flag to get detailed information about:
- Face detection results
- Similarity scores
- Processing statistics
- Frame-by-frame analysis

## Development

### Setting up Development Environment

```bash
# Clone repository
git clone https://github.com/teguh87/face-swap-advanced.git
cd face-swap-advanced

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=face_swap_advanced

# Run specific test
pytest tests/test_face_swap.py
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code  
flake8 src/ tests/

# Type checking
mypy src/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) for face analysis
- [ONNX Runtime](https://onnxruntime.ai/) for model inference
- [OpenCV](https://opencv.org/) for image processing

## Changelog

### v1.2.0
- Added anomaly detection and similarity filtering
- Improved video processing with batch support
- Enhanced error handling and logging
- Added comprehensive configuration options
- Restructured code into classes and modules

### v1.1.0
- Initial class-based refactor
- Added configuration management
- Improved code organization

### v1.0.0
- Initial release
- Basic face swapping functionality