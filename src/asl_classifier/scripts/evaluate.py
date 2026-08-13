import os
import sys
import numpy as np
import tensorflow as tf
from asl_classifier.evaluation.metrics import calculate_metrics, plot_confusion_matrix
from asl_classifier.utils.config import load_config

def evaluate_model(model_path, val_dataset, class_names):
    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Evaluate the model on the validation dataset
    val_loss, val_accuracy = model.evaluate(val_dataset, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Generate predictions
    y_true = []
    y_pred = []

    for images, labels in val_dataset:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))

    # Calculate metrics
    report, cm = calculate_metrics(y_true, y_pred, class_names)
    print("Metrics:", report)

    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names)

def main():
    config = load_config('configs/model_config.yaml')
    model_path = config['model_path']
    val_dataset = ...  # Load or define your validation dataset here
    class_names = ...  # Load or define your class names here

    evaluate_model(model_path, val_dataset, class_names)

if __name__ == "__main__":
    main()
