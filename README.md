# ASL Fingerspelling Classifier

This project builds a deep learning model to classify American Sign Language (ASL) fingerspelling gestures from images. The system uses convolutional neural networks (CNNs) to recognize 28 hand signs (26 letters A–Z, plus "space" and "nothing") from 200×200 RGB images. It explores baseline and optimized CNN architectures, performs manual hyperparameter tuning, and evaluates model performance using standard classification metrics.

---

## Key Features

- End-to-end training pipeline using TensorFlow/Keras
- Image preprocessing and class distribution analysis
- Baseline CNN model for comparison
- Improved CNN architecture with dropout regularization
- Manual random hyperparameter tuning across 10 trials
- Training history visualization and model selection
- Evaluation with classification report and confusion matrix
- Misclassification analysis for model debugging

---

## Example Predictions

- Input: image of gesture "B" → Predicted: B  
- Input: image of gesture "M" → Predicted: N  

Most misclassifications occur between visually similar signs.

---

## Project Structure

```
asl-fingerspelling-classifier/
├── ASL_Alphabet_Dataset/ # Raw image dataset
├── asl_tuning/ # Saved model .h5 files
├── tuning_results/ # Saved model checkpoints and logs
├── main.ipynb # Main notebook (EDA, training, evaluation)
├── README.md
└── .gitignore
```

---

## Technologies Used

- TensorFlow / Keras
- NumPy, Matplotlib
- scikit-learn
- PIL (image loading)
- KerasTuner HyperParameters (manual tuning)

---

## Evaluation Metrics

- Validation Accuracy: ~99.22%
- Confusion Matrix: High performance across nearly all classes
- Classification Report: Strong precision and recall, with minor confusion among similar gestures (e.g., M vs. N)

---

## Limitations

- Dataset is taken under controlled conditions; model may not generalize well to real-world input
- Confusion persists between visually similar characters
- Validation set used for evaluation; no separate holdout test set included

---

## Future Work

- Add data augmentation for robustness (rotation, brightness, noise)
- Apply transfer learning with pretrained CNNs (e.g., ResNet50, MobileNetV2)
- Build real-time webcam interface for live gesture recognition
- Deploy via Streamlit or Gradio for interactive demo

---

## Notes

This project is under active development and intended for research, experimentation, and further extension. Contributions and feedback are welcome.
