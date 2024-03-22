import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# Constants
IMAGE_SIZE = (128, 128)
THRESHOLD = 0.5

# Load the pre-trained deep learning model
model_path = 'best_model.h5'
model = load_model(model_path)

def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, IMAGE_SIZE)
    preprocessed_frame = preprocess_input(resized_frame)
    return preprocessed_frame

def predict_tampering(frame):
    input_array = np.expand_dims(frame, axis=0)
    prediction = model.predict(input_array)
    return prediction[0, 1]

def detect_and_highlight_tampering(original, tampered):
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    tampered_gray = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(original_gray, tampered_gray)
    _, threshold = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    significant_contours = [contour for contour in contours if cv2.contourArea(contour) > 100]

    result_frame = tampered.copy()
    for contour in significant_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 20), 2) 
        result_frame[y:y+h, x:x+w] = cv2.medianBlur(result_frame[y:y+h, x:x+w], 15)  

    return result_frame

video_capture = cv2.VideoCapture('1.mp4')  
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

ret, baseline_frame = video_capture.read()

while video_capture.isOpened():
    ret, current_frame = video_capture.read()
    if not ret:
        break

    preprocessed_frame = preprocess_frame(current_frame)

    is_tampered = predict_tampering(preprocessed_frame)

    result_frame = detect_and_highlight_tampering(baseline_frame, current_frame)

    out.write(result_frame)

    cv2.imshow('Tampering Detection', result_frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

video_capture.release()
out.release()
cv2.destroyAllWindows()

