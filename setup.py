"""
Setup configuration for SmartNotes package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="smartnotes-ocr",
    version="1.0.0",
    description="Intelligent Handwritten Text Recognition System using CRNN + CTC",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SmartNotes Contributors",
    author_email="contributors@smartnotes.local",
    url="https://github.com/Kanike-tea/SmartNotes",
    license="MIT",
    
    packages=find_packages(exclude=["tests", "tests.*", "datasets", "checkpoints"]),
    
    python_requires=">=3.8",
    
    install_requires=[
        "torch>=2.1.2",
        "torchvision>=0.16.2",
        "numpy>=1.24.3",
        "opencv-python>=4.8.1.78",
        "Pillow>=10.0.1",
        "tqdm>=4.66.1",
        "scikit-learn>=1.3.2",
        "matplotlib>=3.8.2",
        "editdistance>=0.8.1",
    ],
    
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.12.0",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
        ],
        "lm": [
            "pyctcdecode>=0.5.0",
            "kenlm>=2020.12.17",
        ],
        "ui": [
            "gradio>=4.0.0",
        ],
    },
    
    entry_points={
        "console_scripts": [
            "smartnotes-train=src.training.train_ocr:main",
            "smartnotes-infer=src.inference.test_ocr:main",
        ],
    },
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    
    keywords=[
        "ocr",
        "handwriting",
        "recognition",
        "deep-learning",
        "pytorch",
        "crnn",
        "ctc",
    ],
    
    project_urls={
        "Bug Tracker": "https://github.com/Kanike-tea/SmartNotes/issues",
        "Documentation": "https://github.com/Kanike-tea/SmartNotes#readme",
        "Source Code": "https://github.com/Kanike-tea/SmartNotes",
    },
)
