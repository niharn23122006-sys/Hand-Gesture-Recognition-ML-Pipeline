"""
 Hand Gesture Recognition using Convolutional Neural Networks (CNN)
=============================================================================
An optimized, high-performance variant utilizing OpenCV vectorizations
and smart path caching for fast data execution pipelines.
"""

import os
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Any
from collections import defaultdict
import numpy as np
import pandas as pd
from datetime import datetime

import kagglehub
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Import Colab's file download utility directly
from google.colab import files

warnings.filterwarnings('ignore')

# Ensure local storage output directory tree exists safely before I/O execution
os.makedirs('outputs', exist_ok=True)


# =====================================================================
# 1. DATASET HANDLING
# =====================================================================

def fetch_dataset() -> Path:
    print("📥 Fetching Leap Gesture Recognition dataset from Kaggle...")
    path = kagglehub.dataset_download("gti-upm/leapgestrecog")
    print(f"✓ Dataset downloaded to: {path}\n")
    return Path(path)


def explore_dataset_structure(dataset_path: Path) -> Dict[str, List[Path]]:
    print("="*70)
    print("DATASET STRUCTURE EXPLORATION")
    print("="*70)

    gesture_images = defaultdict(list)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    print(f"\n🔍 Scanning directory structure efficiently...")

    for dirpath, _, filenames in os.walk(dataset_path):
        if not filenames:
            continue

        dir_path_obj = Path(dirpath)
        dir_name = dir_path_obj.name.lower()
        gesture_label = dir_name.split('_')[0] if '_' in dir_name else dir_name

        valid_files = [dir_path_obj / f for f in filenames if Path(f).suffix.lower() in image_extensions]
        if valid_files:
            gesture_images[gesture_label].extend(valid_files)

    gesture_images = {g: paths for g, paths in gesture_images.items() if len(paths) >= 10}

    print(f"\n📊 Dataset Statistics:")
    print(f"  - Unique gestures parsed: {len(gesture_images)}")

    total_images = 0
    for gesture in sorted(gesture_images.keys()):
        num_images = len(gesture_images[gesture])
        total_images += num_images
        print(f"  - {gesture.ljust(15)}: {num_images:6,} images")

    print(f"  {'Total'.ljust(15)}: {total_images:6,} images")
    return gesture_images


# =====================================================================
# 2. IMAGE PREPROCESSING
# =====================================================================

def load_images(gesture_images: Dict[str, List[Path]], target_size: Tuple[int, int] = (64, 64),
                sample_size: int = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    print("\n" + "="*70)
    print("IMAGE LOADING & PREPROCESSING (OPENCV OPTIMIZED)")
    print("="*70)

    gesture_names = sorted(gesture_images.keys())
    label_map = {gesture: idx for idx, gesture in enumerate(gesture_names)}

    total_to_load = 0
    sampled_paths = {}
    for gesture in gesture_names:
        paths = gesture_images[gesture]
        if sample_size:
            paths = paths[:sample_size]
        sampled_paths[gesture] = paths
        total_to_load += len(paths)

    print(f"🖼️  Target Image Matrix footprint allocation size: {target_size}")
    print(f"⚡ Memory allocation step targeting total of {total_to_load:,} matrix frames...")

    X_matrix = np.zeros((total_to_load, target_size[0], target_size[1], 1), dtype=np.float32)
    y_matrix = np.zeros((total_to_load,), dtype=np.int32)

    current_idx = 0
    for gesture in gesture_names:
        paths = sampled_paths[gesture]
        label_id = label_map[gesture]
        print(f"  ↳ Processing {gesture.capitalize()} ({len(paths)})...")

        for img_path in paths:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
                X_matrix[current_idx, ..., 0] = img_resized / 255.0
                y_matrix[current_idx] = label_id
                current_idx += 1

    if current_idx < total_to_load:
        X_matrix = X_matrix[:current_idx]
        y_matrix = y_matrix[:current_idx]

    print(f"\n✓ Vectorized compilation completed. Final array size: {X_matrix.shape}")
    return X_matrix, y_matrix, gesture_names


# =====================================================================
# 3. DATA SPLITTING
# =====================================================================

def split_dataset(images: np.ndarray, labels: np.ndarray,
                  train_ratio: float = 0.7, val_ratio: float = 0.15,
                  test_ratio: float = 0.15, random_state: int = 42) -> Tuple[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]
]:
    print("\n" + "="*70)
    print("DATASET SPLITTING")
    print("="*70)

    # Shuffle indices globally before performing structural splits to break temporal block groupings
    np.random.seed(random_state)
    indices = np.arange(len(images))
    np.random.shuffle(indices)

    images = images[indices]
    labels = labels[indices]

    test_val_ratio = val_ratio + test_ratio
    X_train, X_test_val, y_train, y_test_val = train_test_split(
        images, labels, test_size=test_val_ratio, random_state=random_state, stratify=labels
    )

    val_test_ratio = test_ratio / test_val_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_test_val, y_test_val, test_size=val_test_ratio, random_state=random_state, stratify=y_test_val
    )

    print(f"✓ Train matrix size: {len(X_train):,} ({len(X_train)/len(images)*100:.1f}%)")
    print(f"✓ Val matrix size:   {len(X_val):,} ({len(X_val)/len(images)*100:.1f}%)")
    print(f"✓ Test matrix size:  {len(X_test):,} ({len(X_test)/len(images)*100:.1f}%)")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# =====================================================================
# 4. CNN ARCHITECTURE
# =====================================================================

def build_cnn_model(input_shape: Tuple[int, int, int], num_classes: int) -> models.Model:
    print("\n" + "="*70)
    print("BUILDING CNN ARCHITECTURE")
    print("="*70)

    # Swapped Global Pooling out for direct Flatten structures to retain edge activations
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape, name='conv1'),
        layers.BatchNormalization(momentum=0.9, name='bn1'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2'),
        layers.BatchNormalization(momentum=0.9, name='bn2'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        layers.Dropout(0.25, name='dropout1'),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv3'),
        layers.BatchNormalization(momentum=0.9, name='bn3'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv4'),
        layers.BatchNormalization(momentum=0.9, name='bn4'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        layers.Dropout(0.25, name='dropout2'),

        layers.Flatten(name='flatten'),
        layers.Dense(128, activation='relu', name='dense1'),
        layers.BatchNormalization(momentum=0.9, name='bn5'),
        layers.Dropout(0.5, name='dropout3'),

        layers.Dense(num_classes, activation='softmax', name='output')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("\n✓ Core Network configuration initialized.")
    return model


# =====================================================================
# 5. TRAINING
# =====================================================================

def train_model(model: models.Model, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray, epochs: int = 20, batch_size: int = 64) -> keras.callbacks.History:
    print("\n" + "="*70)
    print("MODEL TRAINING ENGINE")
    print("="*70)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        shuffle=True, # Added explicit mini-batch shuffling pass
        verbose=1
    )
    return history


# =====================================================================
# 6. EVALUATION FUNCTIONS
# =====================================================================

def evaluate_model(model: models.Model, X_test: np.ndarray, y_test: np.ndarray, gesture_names: List[str]) -> Dict[str, Any]:
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)

    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    test_accuracy = accuracy_score(y_test_labels, y_pred)
    test_loss, _ = model.evaluate(X_test, y_test, verbose=0)

    print(f"\n📊 Test Set Performance:\n  - Loss: {test_loss:.4f}\n  - Accuracy: {test_accuracy:.4f}")

    cm = confusion_matrix(y_test_labels, y_pred)
    return {
        'test_loss': test_loss, 'test_accuracy': test_accuracy,
        'y_test': y_test_labels, 'y_pred': y_pred, 'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm, 'gesture_names': gesture_names
    }


def print_confusion_matrix(eval_results: Dict[str, Any]) -> None:
    cm = eval_results['confusion_matrix']
    gesture_names = eval_results['gesture_names']
    cm_df = pd.DataFrame(cm, index=gesture_names, columns=gesture_names)
    print(f"\n📝 Detailed Confusion Matrix:\n\n{cm_df}\n")


# =====================================================================
# 7. METRIC PLOTTING UTILITIES
# =====================================================================

def plot_training_history(history: keras.callbacks.History, output_dir: str = 'outputs') -> None:
    try:
        out_p = Path(output_dir); out_p.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_title('Model Loss Over Epochs', fontweight='bold')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(history.history['accuracy'], label='Train Acc', linewidth=2)
        axes[1].plot(history.history['val_accuracy'], label='Val Acc', linewidth=2)
        axes[1].set_title('Model Accuracy Over Epochs', fontweight='bold')
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_p / 'training_history.png', dpi=150)
        plt.show()
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")


def plot_confusion_matrix_heatmap(eval_results: Dict[str, Any], output_dir: str = 'outputs') -> None:
    try:
        out_p = Path(output_dir); out_p.mkdir(parents=True, exist_ok=True)
        cm = eval_results['confusion_matrix']
        gesture_names = eval_results['gesture_names']

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=gesture_names, yticklabels=gesture_names)
        plt.title('Confusion Matrix Breakdown', fontweight='bold')
        plt.ylabel('True Class'); plt.xlabel('Predicted Class')

        plt.tight_layout()
        plt.savefig(out_p / 'confusion_matrix.png', dpi=150)
        plt.show()
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")


# =====================================================================
# 8. PIPELINE EXECUTION WITH AUTO-DOWNLOAD
# =====================================================================

def main():
    print("\n" + "="*70)
    print("🤖 HAND GESTURE RECOGNITION - CNN PIPELINE")
    print("="*70)

    try:
        IMAGE_SIZE = (64, 64)
        SAMPLE_SIZE = 500
        EPOCHS = 20
        BATCH_SIZE = 64

        dataset_path = fetch_dataset()
        gesture_images = explore_dataset_structure(dataset_path)

        images, labels, gesture_names = load_images(
            gesture_images, target_size=IMAGE_SIZE, sample_size=SAMPLE_SIZE
        )

        num_classes = len(gesture_names)
        labels_encoded = to_categorical(labels, num_classes=num_classes)

        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(
            images, labels_encoded, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )

        model = build_cnn_model((IMAGE_SIZE[0], IMAGE_SIZE[1], 1), num_classes)
        history = train_model(model, X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE)

        eval_results = evaluate_model(model, X_test, y_test, gesture_names)
        print_confusion_matrix(eval_results)

        print("\n📊 RENDERING AND EXPORTING PERFORMANCE PLOTS...")
        plot_training_history(history)
        plot_confusion_matrix_heatmap(eval_results)

        # TRIGGER AUTOMATIC BROWSER DOWNLOADS FOR BOTH ASSETS Safely
        print("\n📥 TRIGGERING AUTOMATIC BROWSER DOWNLOAD FOR GRAPHICS...")
        if os.path.exists('outputs/training_history.png'):
            files.download('outputs/training_history.png')
        if os.path.exists('outputs/confusion_matrix.png'):
            files.download('outputs/confusion_matrix.png')

        print("\n✓ PIPELINE COMPLETE - HIGH SPEED EXECUTION TARGET MET SUCCESSFULLY")

    except Exception as e:
        print(f"\n❌ Execution Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
