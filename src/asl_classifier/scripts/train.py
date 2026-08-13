import os
import sys
import numpy as np
import tensorflow as tf
from asl_classifier.data.download_data import download_data, extract_data
from asl_classifier.data.loaders import load_asl_data
from asl_classifier.models.cnn_architectures import build_baseline_model
from asl_classifier.models.hyperparameter_tuning import run_hyperparameter_search
from asl_classifier.utils.config import load_config, RAW_DATA_DIR, IMAGE_SIZE, NUM_CLASSES

DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/debashishsau/aslamerican-sign-language-aplhabet-dataset"

def main():
    # Load configuration
    config = load_config('configs/training_config.yaml')

    # Download and extract data if it isn't already present locally
    dataset_dir = os.path.join(RAW_DATA_DIR, "ASL_Alphabet_Dataset")
    if not os.path.isdir(dataset_dir):
        zip_path = os.path.join(RAW_DATA_DIR, "asl-dataset.zip")
        download_data(DATASET_URL, zip_path)
        extract_data(zip_path, RAW_DATA_DIR)
    train_ds, val_ds = load_asl_data(dataset_dir)

    # Build and train the model
    model = build_baseline_model(IMAGE_SIZE + (3,), NUM_CLASSES)
    history = model.fit(train_ds, validation_data=val_ds, epochs=config['training']['epochs'])

    # Save the trained model
    model.save(os.path.join('models', 'baseline_model.h5'))

    # Optionally tune hyperparameters
    if config['training'].get('tune_hyperparameters', False):
        run_hyperparameter_search(train_ds, val_ds)

if __name__ == "__main__":
    main()
