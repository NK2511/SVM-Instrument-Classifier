import os
import numpy as np
import pandas as pd
import librosa

# Load the metadata
df = pd.read_csv('Copy of file_names bys.csv')

# Specify the range of files to process
start_index = 2000  # Starting at file number 401
end_index = len(df)   # Ending at file number 500

# Function to extract features
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    
    # MFCCs
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
    
    # Chroma
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
    
    # Spectral Contrast
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sample_rate).T, axis=0)
    
    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)
    
    # Root Mean Square Energy (RMSE)
    rmse = np.mean(librosa.feature.rms(y=audio).T, axis=0)
    
    # Harmonic-to-Noise Ratio (HNR)
    hnr = np.mean(librosa.effects.hpss(audio)[1])  # Harmonic part for HNR
    
    # Spectral Centroid
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate).T, axis=0)
    
    # Spectral Bandwidth
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate).T, axis=0)
    
    # Spectral Flatness
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=audio).T, axis=0)
    
    # Spectral Rolloff
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, roll_percent=0.85).T, axis=0)
    
    stft = np.abs(librosa.stft(audio))  # Short-time Fourier Transform
    spectral_flux = np.mean(np.sqrt(np.sum(np.diff(stft, axis=1)**2, axis=0)))  # Mean spectral flux
    # Combine all features into a single array
    return np.hstack((mfccs, chroma, spectral_contrast, zcr, rmse, [hnr], spectral_centroid, spectral_bandwidth, spectral_flatness, spectral_rolloff, spectral_flux))

# Extract features for files from start_index to end_index
features = []
labels = []  # List to store the sound types
file_names = []  # List to store the file names

# Process files within the specified range
for index, row in df.iterrows():
    if index < start_index:
        continue
    if index >= end_index:  # Stop after reaching the last file in the range
        break
    
    file_path = os.path.join("C:\\Users\\nandh\\Downloads\\Music_samples\\Train_submission\\Train_submission", row['file_name'])
    
    # Check if the file exists before processing
    if os.path.isfile(file_path):
        data = extract_features(file_path)
        features.append(data)
        labels.append(row[1])  # Append the sound type from the second column
        file_names.append(row['file_name'])  # Append the file name from the first column
        print(f"Row {index + 1} finished")  # Print progress after each row is processed
    else:
        print(f"File not found: {file_path}")

# Convert to DataFrame
features_df = pd.DataFrame(features)

# Add the sound type and file name as new columns
features_df['sound_type'] = labels
features_df['file_name'] = file_names

# Debug: Check the number of columns in the features DataFrame
print(f"Number of features extracted: {features_df.shape[1]}")  # Print the number of features

# Prepare the column headings
mfcc_columns = [f'mfcc_{i+1}' for i in range(40)]  # 40 MFCC features
chroma_columns = [f'chroma_{i+1}' for i in range(12)]  # 12 Chroma features
spectral_contrast_columns = [f'spectral_contrast_{i+1}' for i in range(7)]  # 7 Spectral Contrast features
zcr_columns = ['zcr']  # 1 ZCR feature
rmse_columns = ['rmse']  # 1 RMSE feature
hnr_columns = ['hnr']  # 1 HNR feature
spectral_centroid_columns = ['spectral_centroid']  # 1 Spectral Centroid feature
spectral_bandwidth_columns = ['spectral_bandwidth']  # 1 Spectral Bandwidth feature
spectral_flatness_columns = ['spectral_flatness']  # 1 Spectral Flatness feature
spectral_rolloff_columns = ['spectral_rolloff']  # 1 Spectral Rolloff feature
spectral_flux_columns = ['spectral_flux']  # 1 Spectral Flux feature

# Combine all columns for DataFrame
column_headers = (mfcc_columns + chroma_columns + spectral_contrast_columns +
                  zcr_columns + rmse_columns + hnr_columns +
                  spectral_centroid_columns + spectral_bandwidth_columns +
                  spectral_flatness_columns + spectral_rolloff_columns +
                  spectral_flux_columns + ['sound_type', 'file_name'])

# Ensure the number of column headers matches the number of features extracted
if len(column_headers) == features_df.shape[1]:
    features_df.columns = column_headers  # Set the feature column headers
else:
    print(f"Mismatch in columns: Expected {len(column_headers)} but got {features_df.shape[1]}.")

# Save features DataFrame to CSV
features_df.to_csv('features_extracted.csv', index=False)
print("Completed feature extraction and saved to 'features_extracted.csv'.")
