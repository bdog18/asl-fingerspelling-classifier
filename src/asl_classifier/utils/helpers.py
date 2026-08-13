import tensorflow as tf
from asl_classifier.utils.config import IMAGE_SIZE

def preprocess_image(image_path, target_size=IMAGE_SIZE):
    """
    Preprocess an image for use in the ASL Fingerspelling Classifier.

    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Target size for the image.

    Returns:
        numpy.ndarray: Preprocessed image as a NumPy array.
    """
    image = tf.keras.utils.load_img(image_path, target_size=target_size)
    array = tf.keras.utils.img_to_array(image)
    return array / 255.0