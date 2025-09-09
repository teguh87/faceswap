#!/usr/bin/env python3
"""
Setup script for Face Swap application
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements(filename):
    """Read requirements from file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="faceswap",
    version="1.2.0",
    author="Face Swap Developer",
    author_email="developer@example.com",
    description="Advanced face swapping tool with anomaly detection and similarity filtering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/teguh87/faceswap",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.7.0",
        "numpy>=1.21.0",
        "tqdm>=4.64.0",
        "insightface>=0.7.3",
        "pathlib2>=2.3.7; python_version<'3.4'",
    ],
    extras_require={
        "cpu": ["onnxruntime>=1.15.0"],
        "gpu": ["onnxruntime-gpu>=1.15.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "all": [
            "onnxruntime-gpu>=1.15.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "face-swap=face_swap_advanced.cli:main",
            "faceswap=face_swap_advanced.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "face_swap_advanced": [
            "models/*.onnx",
            "config/*.json",
            "*.md",
        ],
    },
    zip_safe=False,
    keywords=[
        "face-swap",
        "deepfake",
        "computer-vision", 
        "image-processing",
        "video-processing",
        "ai",
        "machine-learning",
        "onnx",
        "insightface"
    ],
    project_urls={
        "Bug Reports": "https://github.com/teguh87/faceswap/issues",
        "Source": "https://github.com/teguh87/faceswap",
        "Documentation": "https://faceswap.readthedocs.io/",
    },
)