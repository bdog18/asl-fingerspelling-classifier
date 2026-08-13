# Configuration settings for the ASL Fingerspelling Classifier project

import os
import yaml
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not locate repo root (no pyproject.toml found)")


# Base directory for the project
BASE_DIR = _find_repo_root(Path(__file__).resolve())

# Data directories
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
INTERIM_DATA_DIR = os.path.join(DATA_DIR, 'interim')

# Model directories
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Results directories
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
REPORTS_DIR = os.path.join(RESULTS_DIR, 'reports')
HYPERPARAMETER_SEARCH_DIR = os.path.join(RESULTS_DIR, 'hyperparameter_search')

# Configuration files
CONFIGS_DIR = os.path.join(BASE_DIR, 'configs')
MODEL_CONFIG_PATH = os.path.join(CONFIGS_DIR, 'model_config.yaml')
TRAINING_CONFIG_PATH = os.path.join(CONFIGS_DIR, 'training_config.yaml')

# Logging settings
LOGGING_LEVEL = 'INFO'  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

# Other constants
NUM_CLASSES = 28  # Number of ASL classes
IMAGE_SIZE = (200, 200)  # Input image size for the model


def _find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not locate repo root (no pyproject.toml found)")

# Function to print configuration settings
def print_config():
    """
    Print the configuration settings for the project.
    """
    print("Configuration Settings:")
    print(f"Base Directory: {BASE_DIR}")
    print(f"Raw Data Directory: {RAW_DATA_DIR}")
    print(f"Processed Data Directory: {PROCESSED_DATA_DIR}")
    print(f"Interim Data Directory: {INTERIM_DATA_DIR}")
    print(f"Model Directory: {MODEL_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"Figures Directory: {FIGURES_DIR}")
    print(f"Reports Directory: {REPORTS_DIR}")
    print(f"Hyperparameter Search Directory: {HYPERPARAMETER_SEARCH_DIR}")
    print(f"Model Config Path: {MODEL_CONFIG_PATH}")
    print(f"Training Config Path: {TRAINING_CONFIG_PATH}")
    print(f"Logging Level: {LOGGING_LEVEL}")
    print(f"Number of Classes: {NUM_CLASSES}")
    print(f"Image Size: {IMAGE_SIZE}")
    
def load_config(config_path):
    """
    Load configuration settings from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.
    
    Returns:
        dict: Configuration settings as a dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config