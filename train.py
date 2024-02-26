import os
import csv
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from joblib import dump

# Function to load dataset
def load_dataset(dataset_file):
    X = []
    y = []

    with open(dataset_file, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:
            image_path, label = line.strip().split(',')
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # Resize the image to match the input size of the model
            resized_image = cv2.resize(image, (160, 160))
            X.append(resized_image.flatten())
            y.append(label)

    return np.array(X), np.array(y)


def start():
    # Directory to save the dataset
    dataset_directory = 'dataset'

    # Create the dataset


    # Path to the dataset CSV file
    dataset_file = 'dataset.csv'

    # Load the dataset
    X, y = load_dataset(dataset_file)

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize SVM classifier
    clf = SVC(kernel='linear')

    # Train the classifier
    clf.fit(X_train, y_train)

    # Predict labels for test set
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)

    # Save the trained model to disk
    model_file = 'face_recognition_model.joblib'
    dump(clf, model_file)
    print('Model saved to', model_file)


