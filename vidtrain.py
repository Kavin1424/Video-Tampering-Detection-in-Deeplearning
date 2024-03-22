import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Constants
VIDEO_SIZE = (128, 128)
NUM_CLASSES = 2
LEARNING_RATE = 0.001
EPOCHS = 20
BATCH_SIZE = 32

# Function to load and preprocess video data
def load_video_data(directory, label):
    data = []
    labels = []
    for filename in os.listdir(directory):
        try:
            if filename.endswith((".avi", ".mp4")):
                cap = cv2.VideoCapture(os.path.join(directory, filename))
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.resize(frame, VIDEO_SIZE)
                    data.append(frame)
                    labels.append(label)
                cap.release()
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    return data, labels

# Load video data
try:
    forged_videos, forged_labels = load_video_data('datasets/forged_videos', 1)
    non_forged_videos, non_forged_labels = load_video_data('datasets/non_forged_videos', 0)
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Combine data and labels
try:
    all_videos = np.array(forged_videos + non_forged_videos)
    all_labels = np.array(forged_labels + non_forged_labels)
    indices = np.arange(all_videos.shape[0])
    np.random.shuffle(indices)
    all_videos = all_videos[indices]
    all_labels = all_labels[indices]

    all_labels = np.eye(NUM_CLASSES)[all_labels]
except Exception as e:
    print(f"Error combining data: {e}")
    exit()

# Split data into train and test sets
try:
    X_train, X_test, y_train, y_test = train_test_split(all_videos, all_labels, test_size=0.2, random_state=42)
except Exception as e:
    print(f"Error splitting data: {e}")
    exit()

# Build the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(VIDEO_SIZE[0], VIDEO_SIZE[1], 3)))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(lr=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True)

# Train the model
try:
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, 
                        validation_data=(X_test, y_test), callbacks=[early_stopping, model_checkpoint])
except Exception as e:
    print(f"Error training model: {e}")
    exit()

# Evaluate the model
try:
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test loss: {loss}')
    print(f'Test accuracy: {accuracy}')
except Exception as e:
    print(f"Error evaluating model: {e}")
