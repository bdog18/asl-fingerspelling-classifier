import os
import numpy as np
import tensorflow as tf
from asl_classifier.utils.config import load_config
from asl_classifier.utils.helpers import preprocess_image

def load_model(model_path):
    """Load the trained model from the specified path."""
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")

def predict(image_path, model):
    """Predict the class of the given image using the loaded model."""
    image = preprocess_image(image_path)  # Preprocess the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class

def main():
    config = load_config("configs/model_config.yaml")  # Load model configuration
    model_path = config['model_path']  # Get model path from config
    model = load_model(model_path)  # Load the model

    # Example usage
    test_image_path = "path/to/test/image.jpg"  # Replace with actual image path
    predicted_class = predict(test_image_path, model)
    print(f"Predicted class: {predicted_class}")

if __name__ == "__main__":
    main()
