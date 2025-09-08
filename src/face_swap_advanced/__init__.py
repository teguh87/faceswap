"""
Face Swap Advanced - Advanced face swapping tool with anomaly detection
"""

__version__ = "1.2.0"
__author__ = "Face Swap Developer"
__email__ = "developer@example.com"
__description__ = "Advanced face swapping tool with anomaly detection and similarity filtering"

from .face_swap import (
    FaceSwapConfig,
    FaceSwapUtils, 
    FaceSwapper,
    create_argument_parser
)

__all__ = [
    "FaceSwapConfig",
    "FaceSwapUtils",
    "FaceSwapper", 
    "create_argument_parser",
    "__version__",
    "__author__",
    "__email__",
    "__description__"
]