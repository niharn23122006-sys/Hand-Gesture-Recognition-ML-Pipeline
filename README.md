# 🖐️ Hand Gesture Recognition Pipeline via Computer Vision

An end-to-end Computer Vision and Machine Learning framework built to capture, preprocess, feature-extract, and classify distinct human hand gestures from raw image streams.

## 🚀 Pipeline Features
* **Dynamic Frame Preprocessing:** Uses `OpenCV` to apply Gaussian blurring, Otsu's thresholding, and morphological transformations to isolate the hand skin region.
* **Region of Interest (ROI) Segmentation:** Automatically isolates and masks background noise to focus feature tracking entirely on the gesture silhouette.
* **Dimensionality Reduction & Flattening:** Compresses structural contour patterns into optimized numerical feature vectors.
* **Supervised Classification Engine:** Implements a highly robust machine learning classifier to accurately map hand shapes to specific control gestures.
* **Real-time Performance Dashboard:** Automatically records model confusion splits and predictive accuracy metrics.

## 📊 Evaluation & Gesture Layout
The optimization metrics, contour tracking models, and live classification reports are saved straight into the project workspace upon execution.

### 🪐 Performance Metrics & Sample Recognition
![Gesture Model Performance](outputs/gesture_recognition_performance.png)

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Core Packages:** OpenCV (opencv-python), Scikit-Learn, NumPy, Matplotlib, Seaborn, Kagglehub
