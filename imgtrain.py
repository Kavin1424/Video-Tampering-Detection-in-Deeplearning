import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model


# Constants
IMAGE_SIZE = (128, 128)
NUM_CLASSES = 2
LEARNING_RATE = 0.001
EPOCHS = 20
BATCH_SIZE = 32

# Function to load and preprocess image data
def load_image_data(directory, label):
    data = []
    labels = []
    for filename in os.listdir(directory):
        try:
            if filename.lower().endswith((".jpg", ".jpeg")):
                img = cv2.imread(os.path.join(directory, filename))
                img = cv2.resize(img, IMAGE_SIZE)
                data.append(img)
                labels.append(label)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    return data, labels

# Load image data
try:
    forged_images, forged_labels = load_image_data('datasets/forged_images', 1)
    non_forged_images, non_forged_labels = load_image_data('datasets/non_forged_images', 0)
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Combine data and labels
try:
    all_images = np.array(forged_images + non_forged_images)
    all_labels = np.array(forged_labels + non_forged_labels)
    indices = np.arange(all_images.shape[0])
    np.random.shuffle(indices)
    all_images = all_images[indices]
    all_labels = all_labels[indices]

    all_labels = np.eye(NUM_CLASSES)[all_labels]
except Exception as e:
    print(f"Error combining data: {e}")
    exit()

# Split data into train and test sets
try:
    X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)
except Exception as e:
    print(f"Error splitting data: {e}")
    exit()

# Build or load the model
model_path = 'best_model.h5'
if os.path.exists(model_path):
    model = load_model(model_path)
    print("Loaded existing model.")
else:
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer=Adam(lr=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
    print("Created new model.")

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', mode='max', save_best_only=True)

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

