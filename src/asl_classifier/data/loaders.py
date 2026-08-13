"""
Data loading utilities for ASL fingerspelling classifier.

This module provides functions to load and preprocess the ASL alphabet dataset
for training and evaluation.
"""

import os
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory


def load_asl_data(
    data_path: str,
    image_size: tuple = (200, 200),
    batch_size: int = 32,
    validation_split: float = 0.2,
    seed: int = 420,
    shuffle_buffer_size: int = 1000
):
    """
    Load and preprocess ASL alphabet dataset.
    
    Args:
        data_path (str): Path to the ASL dataset directory
        image_size (tuple): Target image size for resizing
        batch_size (int): Batch size for training
        validation_split (float): Fraction of data to use for validation
        seed (int): Random seed for reproducibility
        shuffle_buffer_size (int): Buffer size for shuffling
        
    Returns:
        tuple: (train_ds, val_ds) - preprocessed TensorFlow datasets
    """
    
    # Verify dataset path exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset path not found: {data_path}")
    
    # Load training and validation datasets
    train_ds, val_ds = image_dataset_from_directory(
        data_path,
        image_size=image_size,
        batch_size=batch_size,
        validation_split=validation_split,
        subset="both",
        seed=seed
    )
    
    # Apply preprocessing
    train_ds, val_ds = preprocess_datasets(
        train_ds, 
        val_ds, 
        shuffle_buffer_size=shuffle_buffer_size
    )
    
    return train_ds, val_ds


def preprocess_datasets(train_ds, val_ds, shuffle_buffer_size: int = 1000):
    """
    Apply preprocessing to datasets including normalization and optimization.
    
    Args:
        train_ds: Training dataset
        val_ds: Validation dataset
        shuffle_buffer_size (int): Buffer size for shuffling
        
    Returns:
        tuple: (preprocessed_train_ds, preprocessed_val_ds)
    """
    
    # Normalization layer
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    autotune = tf.data.AUTOTUNE
    
    # Preprocess training dataset
    train_ds = train_ds.shuffle(shuffle_buffer_size) \
        .map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=autotune) \
        .prefetch(buffer_size=autotune)
    
    # Preprocess validation dataset
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=autotune) \
        .prefetch(buffer_size=autotune)
    
    return train_ds, val_ds


def get_class_names(data_path: str):
    """
    Get class names from the dataset directory.
    
    Args:
        data_path (str): Path to the dataset directory
        
    Returns:
        list: List of class names
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset path not found: {data_path}")
    
    # Load a temporary dataset just to get class names
    temp_ds = image_dataset_from_directory(
        data_path,
        image_size=(200, 200),
        batch_size=1,
        validation_split=0.01,
        subset="training",
        seed=420
    )
    
    return temp_ds.class_names


def load_test_data(data_path: str, image_size: tuple = (200, 200), batch_size: int = 32):
    """
    Load test dataset (if available).
    
    Args:
        data_path (str): Path to test dataset
        image_size (tuple): Target image size
        batch_size (int): Batch size
        
    Returns:
        tf.data.Dataset: Preprocessed test dataset
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Test dataset path not found: {data_path}")
    
    test_ds = image_dataset_from_directory(
        data_path,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False  # Don't shuffle test data
    )
    
    # Apply normalization
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    autotune = tf.data.AUTOTUNE
    
    test_ds = test_ds.map(
        lambda x, y: (normalization_layer(x), y), 
        num_parallel_calls=autotune
    ).prefetch(buffer_size=autotune)
    
    return test_ds


def create_data_augmentation_pipeline():
    """
    Create a data augmentation pipeline for improved model robustness.
    
    Returns:
        tf.keras.Sequential: Data augmentation pipeline
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.1),
        tf.keras.layers.RandomContrast(0.1)
    ])


def load_asl_data_with_augmentation(
    data_path: str,
    image_size: tuple = (200, 200),
    batch_size: int = 32,
    validation_split: float = 0.2,
    seed: int = 420,
    shuffle_buffer_size: int = 1000,
    use_augmentation: bool = True
):
    """
    Load ASL data with optional data augmentation.
    
    Args:
        data_path (str): Path to the ASL dataset directory
        image_size (tuple): Target image size for resizing
        batch_size (int): Batch size for training
        validation_split (float): Fraction of data to use for validation
        seed (int): Random seed for reproducibility
        shuffle_buffer_size (int): Buffer size for shuffling
        use_augmentation (bool): Whether to apply data augmentation
        
    Returns:
        tuple: (train_ds, val_ds) - preprocessed TensorFlow datasets
    """
    
    # Load base datasets
    train_ds, val_ds = load_asl_data(
        data_path=data_path,
        image_size=image_size,
        batch_size=batch_size,
        validation_split=validation_split,
        seed=seed,
        shuffle_buffer_size=shuffle_buffer_size
    )
    
    # Apply data augmentation to training set if requested
    if use_augmentation:
        data_augmentation = create_data_augmentation_pipeline()
        train_ds = train_ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    return train_ds, val_ds
