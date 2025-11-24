"""
Path utilities for SmartNotes.

Provides consistent path resolution and module import handling
across the entire project, regardless of execution location.
"""

import sys
from pathlib import Path


def get_project_root():
    """
    Get the SmartNotes project root directory.
    
    Returns the directory containing the SmartNotes package
    (parent of src/, preprocessing/, smartnotes/, etc.)
    
    Returns:
        Path: Absolute path to project root
    """
    # Start from this file's location (smartnotes/paths.py)
    current_file = Path(__file__).resolve()
    
    # Go up: smartnotes/paths.py → smartnotes/ → SmartNotes/ (project root)
    project_root = current_file.parent.parent
    
    return project_root


def setup_imports():
    """
    Setup Python path for consistent imports.
    
    Call this at the start of any script to ensure imports work
    regardless of execution location.
    """
    project_root = get_project_root()
    
    # Add project root to path if not already there
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def get_checkpoint_dir():
    """Get checkpoints directory path."""
    return get_project_root() / "checkpoints"


def get_dataset_dir(dataset_name=None):
    """
    Get datasets directory path.
    
    Args:
        dataset_name: Optional specific dataset name (GNHK, CensusHWR, IAM, etc.)
        
    Returns:
        Path to datasets directory or specific dataset
    """
    datasets_dir = get_project_root() / "datasets"
    if dataset_name:
        return datasets_dir / dataset_name
    return datasets_dir


def get_results_dir():
    """Get results directory path."""
    return get_project_root() / "results"


def get_lm_dir():
    """Get language model directory path."""
    return get_project_root() / "lm"


# Auto-setup imports when module is imported
setup_imports()
