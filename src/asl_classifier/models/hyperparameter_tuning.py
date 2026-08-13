import os
import random
import tensorflow as tf
from tensorflow.keras import layers
try:
    from keras_tuner import HyperParameters
except ImportError:
    from kerastuner import HyperParameters

from asl_classifier.utils.config import NUM_CLASSES, MODEL_DIR

def build_model(hp):
    """Build CNN model with specified hyperparameters"""
    model = tf.keras.models.Sequential()
    
    # Extract hyperparameters
    filters1 = hp.get('filters1')
    filters2 = hp.get('filters2') 
    dense_units = hp.get('dense_units')
    dropout_rate = hp.get('dropout')
    num_blocks = hp.get('num_conv_blocks')
    
    # Build architecture
    model.add(layers.Conv2D(filters1, (3, 3), activation='relu', input_shape=(200, 200, 3)))
    model.add(layers.MaxPooling2D(2, 2))
    
    model.add(layers.Conv2D(filters2, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    
    # Optional third block
    if num_blocks == 3:
        model.add(layers.Conv2D(filters2 * 2, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(2, 2))
    
    # Classification head
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model

class HistoryCallback(tf.keras.callbacks.Callback):
    """Custom callback to track training history"""
    def on_train_begin(self, logs=None):
        self.history = {}
    
    def on_epoch_end(self, epoch, logs=None):
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

def run_hyperparameter_search(train_ds, val_ds, max_trials=10):
    """Run manual random search for hyperparameter optimization"""
    
    # Define search space
    search_space = {
        'filters1': [32, 64],
        'filters2': [64, 128], 
        'dense_units': [128, 256],
        'dropout': [0.2, 0.3, 0.5],
        'num_conv_blocks': [2, 3]
    }
    
    trial_results = []
    
    print(f"Starting hyperparameter search with {max_trials} trials...")
    print("=" * 60)
    
    for trial in range(max_trials):
        # Sample hyperparameters randomly
        hp = HyperParameters()
        for param, values in search_space.items():
            hp.Fixed(param, random.choice(values))
        
        try:
            # Build and train model
            model = build_model(hp)
            history_callback = HistoryCallback()
            trial_dir = os.path.join(MODEL_DIR, "asl_tuning")
            os.makedirs(trial_dir, exist_ok=True)
            trial_model_path = os.path.join(trial_dir, f"best_model_trial_{trial}.h5")

            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2, verbose=0),
                tf.keras.callbacks.ModelCheckpoint(trial_model_path, monitor='val_accuracy', 
                                                 save_best_only=True, verbose=0),
                history_callback
            ]
            
            print(f"Trial {trial + 1}/{max_trials}")
            print(f"Config: {dict(hp.values)}")
            
            model.fit(train_ds, validation_data=val_ds, epochs=10, 
                     callbacks=callbacks, verbose=0)
            
            final_val_acc = max(history_callback.history.get('val_accuracy', [0]))
            
            trial_results.append({
                'trial_id': trial,
                'hyperparameters': dict(hp.values),
                'val_accuracy': final_val_acc,
                'history': history_callback.history,
                'model_path': trial_model_path
            })
            
            print(f"Validation Accuracy: {final_val_acc:.4f}")
            print("-" * 40)
            
        except Exception as e:
            print(f"Trial {trial + 1} failed: {str(e)}")
            trial_results.append({
                'trial_id': trial,
                'hyperparameters': None,
                'val_accuracy': 0.0,
                'history': None,
                'model_path': None
            })
    
    print("Hyperparameter search completed!")
    return trial_results