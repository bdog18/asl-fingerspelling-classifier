import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def calculate_metrics(y_true, y_pred, class_names):
    """Calculate and return classification metrics."""
    report = classification_report(y_true, y_pred, target_names=class_names)
    cm = confusion_matrix(y_true, y_pred)
    return report, cm

def print_metrics(report, cm):
    """Print the classification report and confusion matrix."""
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)

def plot_confusion_matrix(cm, class_names):
    """Plot the confusion matrix."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix")
    plt.show()