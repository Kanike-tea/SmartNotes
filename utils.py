"""
Logging and utility functions for SmartNotes OCR project.

This module provides centralized logging setup and common utility functions
used throughout the project.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Any
import json
import traceback


class SmartNotesLogger:
    """Centralized logger for SmartNotes project."""
    
    _logger: Optional[logging.Logger] = None
    
    @classmethod
    def get_logger(cls, name: str = "SmartNotes", config: Optional[Any] = None) -> logging.Logger:
        """
        Get or create a logger instance.
        
        Args:
            name: Logger name
            config: Configuration object with logging settings
            
        Returns:
            Configured logger instance
        """
        if cls._logger is not None:
            return cls._logger
        
        # Import here to avoid circular dependency
        from config import LoggingConfig
        if config is None:
            config = LoggingConfig()
        
        cls._logger = logging.getLogger(name)
        cls._logger.setLevel(getattr(logging, config.LOG_LEVEL))
        
        # Prevent duplicate handlers
        if cls._logger.hasHandlers():
            return cls._logger
        
        formatter = logging.Formatter(config.LOG_FORMAT)
        
        # Console handler
        if config.LOG_TO_CONSOLE:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            cls._logger.addHandler(console_handler)
        
        # File handler
        if config.LOG_TO_FILE:
            log_file = Path(config.LOG_FILE)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5
            )
            file_handler.setFormatter(formatter)
            cls._logger.addHandler(file_handler)
        
        return cls._logger
    
    @classmethod
    def reset(cls) -> None:
        """Reset logger instance."""
        cls._logger = None


def get_logger(name: str = "SmartNotes") -> logging.Logger:
    """Get logger instance."""
    return SmartNotesLogger.get_logger(name)


def log_error(logger: logging.Logger, message: str, exception: Optional[Exception] = None) -> None:
    """
    Log an error with optional exception traceback.
    
    Args:
        logger: Logger instance
        message: Error message
        exception: Optional exception object
    """
    if exception:
        logger.error(f"{message}\n{traceback.format_exc()}")
    else:
        logger.error(message)


def log_config(logger: logging.Logger, config_dict: dict) -> None:
    """
    Log configuration in a formatted way.
    
    Args:
        logger: Logger instance
        config_dict: Configuration dictionary
    """
    logger.info("Configuration:")
    logger.info(json.dumps(config_dict, indent=2, default=str))


# ============================================================================
# Device utilities
# ============================================================================

def get_device(use_cuda: bool = True, use_mps: bool = True, force_cpu: bool = False):
    """
    Get the appropriate device (CPU, CUDA, or MPS).
    
    Args:
        use_cuda: Whether to use CUDA if available
        use_mps: Whether to use MPS (Apple Metal) if available
        force_cpu: Force CPU usage
        
    Returns:
        torch.device instance
    """
    import torch
    logger = get_logger(__name__)
    
    if force_cpu:
        logger.info("Forced CPU usage")
        return torch.device("cpu")
    
    if use_mps and torch.backends.mps.is_available():
        logger.info("Using MPS (Apple Metal Performance Shaders)")
        # Enable MPS fallback for ops not implemented on MPS (like CTC)
        import os
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        return torch.device("mps")
    
    if use_cuda and torch.cuda.is_available():
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    
    logger.info("Using CPU")
    return torch.device("cpu")


# ============================================================================
# Path utilities
# ============================================================================

def ensure_path_exists(path: str, is_file: bool = False) -> Path:
    """
    Ensure a path exists, creating directories as needed.
    
    Args:
        path: Path to create
        is_file: If True, create parent directories only; if False, create all dirs
        
    Returns:
        Path object
        
    Raises:
        ValueError: If path is invalid
    """
    try:
        path_obj = Path(path)
        if is_file:
            path_obj.parent.mkdir(parents=True, exist_ok=True)
        else:
            path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj
    except Exception as e:
        raise ValueError(f"Failed to create path {path}: {e}")


def file_exists(path: str) -> bool:
    """Check if file exists."""
    return Path(path).exists() and Path(path).is_file()


def dir_exists(path: str) -> bool:
    """Check if directory exists."""
    return Path(path).exists() and Path(path).is_dir()


# ============================================================================
# Validation utilities
# ============================================================================

def validate_image_path(path: str) -> bool:
    """
    Validate that image path exists and has valid extension.
    
    Args:
        path: Path to image file
        
    Returns:
        True if valid, False otherwise
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    path_obj = Path(path)
    return path_obj.exists() and path_obj.suffix.lower() in valid_extensions


def validate_checkpoint_path(path: str) -> bool:
    """
    Validate that checkpoint file exists.
    
    Args:
        path: Path to checkpoint file
        
    Returns:
        True if valid, False otherwise
    """
    path_obj = Path(path)
    return path_obj.exists() and path_obj.suffix == '.pth'


# ============================================================================
# Metrics utilities
# ============================================================================

def calculate_cer(predicted: str, ground_truth: str) -> float:
    """
    Calculate Character Error Rate (CER).
    
    Args:
        predicted: Predicted text
        ground_truth: Ground truth text
        
    Returns:
        CER as a float between 0 and 1
    """
    if not ground_truth:
        return 1.0 if predicted else 0.0
    
    # Simple edit distance calculation
    from difflib import SequenceMatcher
    matcher = SequenceMatcher(None, ground_truth, predicted)
    ratio = matcher.ratio()
    return 1.0 - ratio


def calculate_wer(predicted: str, ground_truth: str) -> float:
    """
    Calculate Word Error Rate (WER).
    
    Args:
        predicted: Predicted text
        ground_truth: Ground truth text
        
    Returns:
        WER as a float between 0 and 1
    """
    pred_words = predicted.split()
    gt_words = ground_truth.split()
    
    if not gt_words:
        return 1.0 if pred_words else 0.0
    
    # Simple word-level comparison
    from difflib import SequenceMatcher
    matcher = SequenceMatcher(None, gt_words, pred_words)
    ratio = matcher.ratio()
    return 1.0 - ratio
