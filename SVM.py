from sklearn.feature_selection import RFE
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import librosa
import numpy as np

# Load the data
features_df = pd.read_csv('features_extracted.csv')

# Define the feature columns and target variable
featurecol = [f'mfcc_{i}' for i in range(1, 41)]
featurecol = features_df.columns[:66].tolist()
X = features_df[featurecol].values
y = features_df['sound_type'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the classifier
base_classifier = SVC(kernel='linear', C=1.0, random_state=42)

# Use RFE to select the top n features
n_features_to_select = 20  # Adjust the number of features you want to select for optimal performance
selector = RFE(estimator=base_classifier, n_features_to_select=n_features_to_select, step=1)
selector = selector.fit(X_train, y_train)

# Transform the training and test data to keep only the selected features
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Train the classifier on the selected features
classifier = SVC(kernel='linear', C=1.0, random_state=42)
classifier.fit(X_train_selected, y_train)

# Predict on the test set using selected features
y_pred = classifier.predict(X_test_selected)

# Define label order using unique labels from `y`
labels = np.unique(y)

print("Classification Report:")
print(classification_report(y_test, y_pred, labels=labels))

# Generate the confusion matrix with consistent labels
cm = confusion_matrix(y_test, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print("\nConfusion Matrix:")
print(cm_df)

# Function to extract MFCC and predict using selected features
def extract_mfcc(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
    # Average the MFCCs over time
    mfccs_mean = np.mean(mfccs.T, axis=0)
    
    return mfccs_mean

def predict_instrument(file_path):
    mfcc_features = extract_mfcc(file_path)
    
    # Select only the features used by the classifier
    mfcc_features_selected = selector.transform([mfcc_features])
    
    # Make prediction
    prediction = classifier.predict(mfcc_features_selected)
    print("Audio is predicted as:", prediction[0])
    return prediction[0]

# Input audio file path
audio_file = "C:\\Users\\nandh\\Downloads\\Music_samples\\Test_submission\\Test_submission\\warm-piano-logo-116098.wav"
predict_instrument(audio_file)
