# ASL Fingerspelling Classifier

A deep learning project that classifies American Sign Language (ASL) fingerspelling gestures using Convolutional Neural Networks (CNNs). The model recognizes 28 different hand signs corresponding to the 26 English alphabet letters plus "space" and "nothing" from 200x200 RGB images.

## Overview

This project demonstrates the application of computer vision and deep learning techniques to assistive technology. The CNN model achieves over 99% validation accuracy on the ASL Alphabet dataset, showcasing the potential for real-world applications in ASL translation and accessibility tools.

## Dataset

The project uses the [ASL Alphabet Dataset](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset) from Kaggle, which contains:
- Over 223,000 RGB images
- 28 classes (A-Z, space, nothing)
- 200x200 pixel resolution
- Controlled lighting conditions

## Model Architecture

The final optimized CNN architecture includes:
- 3 Convolutional layers with ReLU activation (32, 64, 128 filters)
- MaxPooling layers for dimensionality reduction
- Flatten layer followed by Dense layer (256 units)
- Dropout regularization (0.5) to prevent overfitting
- Softmax output layer for 28-class classification

## Key Features

- **Comprehensive Data Analysis**: Exploratory data analysis with class distribution visualization
- **Model Comparison**: Baseline vs. optimized CNN architectures
- **Hyperparameter Tuning**: Manual random search across multiple model configurations
- **Performance Evaluation**: Detailed metrics including confusion matrix and classification reports
- **Error Analysis**: Investigation of misclassified examples to understand model limitations

## Results

- **Validation Accuracy**: 99.22%
- **Model Convergence**: Fast training with early stopping
- **Generalization**: Strong performance across most ASL characters
- **Error Patterns**: Minor confusion between visually similar signs (e.g., M vs N)

## Installation and Setup

1. Clone the repository:
```bash
git clone https://github.com/bdog18/asl-fingerspelling-classifier.git
cd asl-fingerspelling-classifier
```

2. Install required dependencies:
```bash
pip install tensorflow keras scikit-learn matplotlib pillow numpy
```

3. Run the Jupyter notebook:
```bash
jupyter notebook main.ipynb
```

## Usage

The main notebook (`main.ipynb`) contains the complete pipeline:

1. **Data Loading and Preprocessing**: Automatic dataset download and preparation
2. **Exploratory Data Analysis**: Visualization of sample images and class distributions
3. **Baseline Model**: Simple CNN for performance comparison
4. **Model Development**: Improved CNN with regularization techniques
5. **Hyperparameter Tuning**: Optimization across multiple trials
6. **Evaluation**: Comprehensive performance analysis with visualizations

## Project Structure

```
asl-fingerspelling-classifier/
├── main.ipynb                 # Main Jupyter notebook
├── README.md                  # Project documentation
├── ASL_Alphabet_Dataset/      # Dataset (downloaded automatically)
│   ├── asl_alphabet_train/    # Training images
│   └── asl_alphabet_test/     # Test images
├── asl_tuning/               # Saved model files
└── tuning_results/           # Training logs and checkpoints
```

## Technical Details

### Data Preprocessing
- Pixel normalization (0-255 → 0-1 range)
- Image resizing to 200x200 pixels
- Data shuffling and prefetching for optimal training

### Training Strategy
- Adam optimizer with sparse categorical crossentropy loss
- Early stopping with patience of 2 epochs
- Model checkpointing to save best weights
- 80/20 train/validation split

### Hyperparameter Search Space
- Convolutional filters: [32, 64] and [64, 128]
- Dense layer units: [128, 256]
- Dropout rates: [0.2, 0.3, 0.5]
- Network depth: [2, 3] convolutional blocks

## Performance Metrics

The model demonstrates excellent performance across all evaluation metrics:
- High precision and recall for most classes
- Minimal bias across different ASL characters
- Strong generalization capabilities

## Limitations and Considerations

- **Environmental Constraints**: Dataset captured under controlled conditions
- **Real-world Variability**: Performance may vary with different lighting, backgrounds, or camera angles
- **Similar Gestures**: Some confusion between visually similar hand signs
- **Evaluation Setup**: Uses validation split rather than separate test set

## Future Enhancements

- **Data Augmentation**: Implement rotation, scaling, and brightness variations
- **Transfer Learning**: Leverage pre-trained models (ResNet, EfficientNet)
- **Real-time Application**: Develop webcam-based live recognition system
- **Model Deployment**: Create web interface using Streamlit or Gradio
- **Mobile Integration**: Optimize for mobile deployment with TensorFlow Lite

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to help improve this project.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Dataset provided by [Debashish Sau on Kaggle](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset)
- Built with TensorFlow and Keras
- Inspired by the importance of accessibility technology for the Deaf and Hard of Hearing community
