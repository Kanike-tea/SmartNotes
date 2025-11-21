# Multi-stage build for SmartNotes OCR application

# Stage 1: Base image with Python and system dependencies
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Development image with all dependencies
FROM base as builder

WORKDIR /tmp/smartnotes

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --user --no-warn-script-location -r requirements.txt

# Stage 3: Production image
FROM base as production

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Set PATH to include user pip installations
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . /app

# Create necessary directories
RUN mkdir -p /app/datasets \
    /app/checkpoints \
    /app/results \
    /app/lm

# Set environment for PyTorch
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.model.ocr_model import CRNN; print('OK')" || exit 1

# Default command: training
CMD ["python", "src/training/train_ocr.py"]

# Stage 4: Development image with additional tools
FROM production as development

# Install development dependencies
RUN pip install --user \
    pytest==7.4.3 \
    pytest-cov==4.1.0 \
    black==23.12.0 \
    flake8==6.1.0 \
    ipython==8.18.1

# Development command: bash shell
CMD ["/bin/bash"]

# Labels
LABEL maintainer="SmartNotes Contributors" \
      description="SmartNotes: Intelligent Handwritten Text Recognition" \
      version="1.0"
