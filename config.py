"""
Configuration module for SmartNotes OCR project.

This module centralizes all configuration parameters for training, inference,
and data processing. Modify values here instead of hardcoding them throughout
the codebase.
"""

from pathlib import Path
from typing import Dict, Any, Optional

# ============================================================================
# Paths
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
LM_DIR = PROJECT_ROOT / "lm"

# Ensure directories exist
CHECKPOINTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ============================================================================
# Dataset Configuration
# ============================================================================
class DatasetConfig:
    """Configuration for dataset loading and preprocessing."""
    
    # Root directory for datasets
    ROOT_DIR: str = str(DATASETS_DIR)
    
    # Train/val split ratio
    TRAIN_VAL_SPLIT: float = 0.85
    
    # Image preprocessing
    IMG_WIDTH: int = 128
    IMG_HEIGHT: int = 32
    
    # Text tokenization
    ALLOWED_CHARS: str = "abcdefghijklmnopqrstuvwxyz0123456789"
    
    # Dataset sampling (use subset for faster training)
    MAX_TRAIN_SAMPLES: int = 20000  # Set to None to use all
    MAX_VAL_SAMPLES: int = 5000     # Set to None to use all
    
    # Number of workers for data loading
    NUM_WORKERS: int = 4


# ============================================================================
# Model Configuration
# ============================================================================
class ModelConfig:
    """Configuration for the CRNN model."""
    
    # CNN backbone
    CNN_CHANNELS: list = [1, 64, 128, 256, 256, 512]
    CNN_KERNEL_SIZE: int = 3
    CNN_PADDING: int = 1
    
    # RNN (LSTM)
    RNN_INPUT_SIZE: int = 1024  # 512 channels * 2 height
    RNN_HIDDEN_SIZE: int = 256
    RNN_NUM_LAYERS: int = 2
    RNN_BIDIRECTIONAL: bool = True
    
    # Dropout (optional enhancement)
    DROPOUT_RATE: float = 0.1


# ============================================================================
# Training Configuration
# ============================================================================
class TrainingConfig:
    """Configuration for training parameters."""
    
    # Training hyperparameters
    NUM_EPOCHS: int = 20
    BATCH_SIZE: int = 16
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-5
    
    # Learning rate scheduler
    LR_SCHEDULER_STEP_SIZE: int = 5
    LR_SCHEDULER_GAMMA: float = 0.5
    
    # Loss function
    CTC_BLANK_INDEX: Optional[int] = None  # Set dynamically based on vocab size
    CTC_ZERO_INFINITY: bool = True
    
    # Device setup
    USE_MPS: bool = True  # Use Apple Metal Performance Shaders if available
    USE_CUDA: bool = True
    
    # Checkpoint saving
    SAVE_DIR: str = str(CHECKPOINTS_DIR)
    SAVE_FREQUENCY: int = 5  # Save every N epochs
    SAVE_BEST_ONLY: bool = False
    
    # Logging
    LOG_FREQUENCY: int = 100  # Log metrics every N batches
    PRINT_SAMPLES: int = 3    # Number of sample predictions to print per epoch


# ============================================================================
# Inference Configuration
# ============================================================================
class InferenceConfig:
    """Configuration for inference."""
    
    # Checkpoint to load - Use epoch 6 model (strong performance, 4.65% CER)
    CHECKPOINT_PATH: str = str(CHECKPOINTS_DIR / "ocr_epoch_6.pth")
    
    # Inference device
    USE_CPU: bool = False  # Force CPU usage
    
    # LM decoding
    USE_LM: bool = True
    LM_PATH: str = str(LM_DIR / "smartnotes.arpa")
    LM_WEIGHT: float = 0.3  # Blend LM scores with OCR scores
    BEAM_WIDTH: int = 5    # Beam search width for decoding
    
    # Output settings
    MAX_SAMPLES_TO_DISPLAY: int = 5


# ============================================================================
# Preprocessing Configuration
# ============================================================================
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    
    # Subject classification
    USE_SUBJECT_CLASSIFIER: bool = True
    
    # Line segmentation
    LINE_SEGMENT_METHOD: str = "adaptive"  # 'adaptive' or 'heuristic'
    
    # Text cleaning
    CLEAN_TEXT: bool = True
    REMOVE_EXTRA_WHITESPACE: bool = True


# ============================================================================
# Logging Configuration
# ============================================================================
class LoggingConfig:
    """Configuration for logging."""
    
    # Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_LEVEL: str = "INFO"
    
    # Log file
    LOG_FILE: str = str(PROJECT_ROOT / "smartnotes.log")
    
    # Log format
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Whether to log to console
    LOG_TO_CONSOLE: bool = True
    
    # Whether to log to file
    LOG_TO_FILE: bool = True


# ============================================================================
# Aggregated Configuration
# ============================================================================
class Config:
    """Aggregated configuration class for easy access."""
    
    # Sub-configurations
    dataset = DatasetConfig()
    model = ModelConfig()
    training = TrainingConfig()
    inference = InferenceConfig()
    preprocessing = PreprocessingConfig()
    logging = LoggingConfig()
    
    @staticmethod
    def to_dict() -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {
            'dataset': {k: v for k, v in DatasetConfig.__dict__.items() if not k.startswith('_')},
            'model': {k: v for k, v in ModelConfig.__dict__.items() if not k.startswith('_')},
            'training': {k: v for k, v in TrainingConfig.__dict__.items() if not k.startswith('_')},
            'inference': {k: v for k, v in InferenceConfig.__dict__.items() if not k.startswith('_')},
            'preprocessing': {k: v for k, v in PreprocessingConfig.__dict__.items() if not k.startswith('_')},
            'logging': {k: v for k, v in LoggingConfig.__dict__.items() if not k.startswith('_')},
        }
    
    @staticmethod
    def print_config() -> None:
        """Pretty print the current configuration."""
        import json
        print("\n" + "="*80)
        print("SMARTNOTES CONFIGURATION")
        print("="*80)
        print(json.dumps(Config.to_dict(), indent=2, default=str))
        print("="*80 + "\n")


# Default instance
config = Config()
