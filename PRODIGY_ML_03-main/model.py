import zipfile
import os
import cv2
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

# Step 1: Extract the zip file
zip_file_path = '/content/cats and dogs.zip'  # Path to your uploaded zip file
extract_dir = '/content/dataset/'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# Step 2: Load the CSV file with image paths and labels
train_labels_path = '/content/dataset/train.csv'
train_labels = pd.read_csv(train_labels_path)

# Step 3: Function to load images from 'cat' and 'dog' folders based on CSV
def load_images(folder, labels_df, img_size=(128, 128)):  # Increase image size
    images = []
    labels = []
    for i, row in labels_df.iterrows():
        img_path = os.path.join(extract_dir, row['image:FILE'])  # This path comes from the CSV
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)  # Resize the image to 128x128
            img = img.flatten()  # Flatten the image into a 1D vector
            images.append(img)
            labels.append(row['category'])  # Assuming 'label' column contains 0 for cat and 1 for dog
    return np.array(images), np.array(labels)

# Step 4: Load and preprocess the training data
train_folder = '/content/dataset/train'  # Folder where 'cat' and 'dog' images are stored
train_images, train_labels = load_images(train_folder, train_labels)

# Step 5: Normalize the data using Min-Max Scaler
scaler = MinMaxScaler()
train_images = scaler.fit_transform(train_images)

# Step 6: Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Step 7: Train the SVM model using RBF kernel
svm_model = svm.SVC(kernel='rbf')  # Use RBF kernel instead of linear
svm_model.fit(X_train, y_train)

# Step 8: Test the model on the test set
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Step 9: Save the trained model and scaler
joblib.dump(svm_model, 'svm_cats_dogs_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Output the accuracy
print(f"Accuracy: {accuracy * 100:.2f}%")
