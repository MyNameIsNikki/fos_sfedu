"""
Baseline for ЛР-1: Viola-Jones + HOG+SVM.

This is a structural demonstration showing the expected pipeline,
NOT a full solution. Students must implement their own version.
"""

import cv2
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import average_precision_score


def run_viola_jones_demo(image_path: str):
    """Apply pretrained Haar cascade — serves as a starting point."""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    print(f"  Detected {len(faces)} faces (Viola-Jones)")
    return faces


def extract_hog_features(images: list[np.ndarray]) -> np.ndarray:
    """Extract HOG features from a list of grayscale images."""
    hog = cv2.HOGDescriptor()
    features = []
    for img in images:
        resized = cv2.resize(img, (64, 128))
        feat = hog.compute(resized).flatten()
        features.append(feat)
    return np.array(features)


def train_svm(X: np.ndarray, y: np.ndarray) -> LinearSVC:
    """Train a linear SVM — students should tune C and class_weight."""
    svm = LinearSVC(C=1.0, class_weight="balanced", max_iter=1000)
    svm.fit(X, y)
    print(f"  SVM trained (C=1.0), support vectors: {len(svm.support_vectors_)}")
    return svm


def evaluate_svm(svm: LinearSVC, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate SVM and compute mAP."""
    y_score = svm.decision_function(X_test)
    ap = average_precision_score(y_test, y_score)
    print(f"  mAP: {ap:.3f}")
    return ap


if __name__ == "__main__":
    print("Baseline for ЛР-1 (structural demo only)")
    print("This script shows the pipeline. Load real data to evaluate.")
