import cv2
import numpy as np
import time
import os
import csv
import json
import logging
from collections import deque
from datetime import datetime
from imutils import resize
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
from model import Model  

log_filename = 'detection_log.txt'
if os.path.exists(log_filename):
    os.remove(log_filename)
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

violence_log_filename = 'violence_detection_log.txt'
if os.path.exists(violence_log_filename):
    os.remove(violence_log_filename)
violence_logging = logging.getLogger('violence_log')
violence_logging.setLevel(logging.INFO)
violence_handler = logging.FileHandler(violence_log_filename)
violence_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
violence_logging.addHandler(violence_handler)

# Function to log violence detection events
def log_violence_detection(label_to_display,last_log_time):
    current_time = time.time()
    if current_time - last_log_time >= 1:  
        violence_logging.info(f"detected:{label_to_display}")
        last_log_time = current_time
    return last_log_time
    #timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    #violence_logging.info(f'{timestamp} - Violence Detected: {label_to_display}')

# Constants for violence detection
DETECTION_THRESHOLD = 10
TIME_WINDOW = 3
ALERT_COOLDOWN = 10
BATCH_SIZE = 9
MOTION_THRESHOLD = 500

# YOLO Configuration
YOLO_WEIGHTS = "YOLOv4-tiny/yolov4-tiny.weights"
YOLO_CONFIG = "YOLOv4-tiny/yolov4-tiny.cfg"
FRAME_SIZE = 416  # Resolution for YOLO processing
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# Deep SORT configuration
MAX_COSINE_DISTANCE = 0.7
NN_BUDGET = None
TRACK_MAX_AGE = 30
MODEL_FILENAME = 'model_data/mars-small128.pb'

# Initialize the crowd detection model
net = cv2.dnn.readNetFromDarknet(YOLO_CONFIG, YOLO_WEIGHTS)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

encoder = gdet.create_box_encoder(MODEL_FILENAME, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", MAX_COSINE_DISTANCE, NN_BUDGET)
tracker = Tracker(metric, max_age=TRACK_MAX_AGE)

# Create output directory
if not os.path.exists('processed_data'):
    os.makedirs('processed_data')
crowd_data_file = open('processed_data/crowd_data.csv', 'w')
crowd_data_writer = csv.writer(crowd_data_file)
if os.path.getsize('processed_data/crowd_data.csv') == 0:
    crowd_data_writer.writerow(['Time', 'Human Count', 'Violence Alert'])

# def log_label(label):
#     """Log the detected label with the current timestamp."""
#     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     logging.info(f'{timestamp} - Detected label: {label}')

def check_for_alert(detection_times, last_alert_time):
    current_time = time.time()
    
    # Remove timestamps older than TIME_WINDOW seconds from the buffer
    while detection_times and current_time - detection_times[0] > TIME_WINDOW:
        detection_times.popleft()

    # If we have DETECTION_THRESHOLD or more detections in the last TIME_WINDOW seconds, trigger alert
    if len(detection_times) >= DETECTION_THRESHOLD:
        # Trigger alert only if ALERT_COOLDOWN has passed since the last alert
        if current_time - last_alert_time >= ALERT_COOLDOWN:
            print("Alert: Violence detected!")
            violence_logging.info("Alert: Violence detected!")
            return current_time  
    
    return last_alert_time

def alerts(model, frames, detection_times, last_alert_time, last_log_time):
    labels = model.predict_batch(frames)
    current_time = time.time()
    label_to_display = ""
    label_to_log = ""

    for label_dict in labels:
        label = label_dict['label'].lower()
        label_to_log = label

        if any(keyword in label for keyword in ['violence', 'fight', 'fire', 'crash']):
            detection_times.append(current_time)
            last_alert_time = check_for_alert(detection_times, last_alert_time)
            if len(detection_times) >= DETECTION_THRESHOLD and current_time - last_alert_time >= ALERT_COOLDOWN:
                violence_logging.info(f"Alert: {label.capitalize()} detected!")
                last_alert_time = current_time
            label_to_display = label.capitalize()  # Update display variable
            last_log_time = log_violence_detection(label_to_display, last_log_time)
        else:
            detection_times.clear()

    return label_to_display, label_to_log, last_alert_time


def process_real_time():
    print("Starting real-time processing...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access the webcam")
        return
    
    #window_name = "Crowd and Violence Detection"
    #cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
    #cv2.resizeWindow(window_name, 1080, 720)  

    model = Model()  
    detection_times = deque()
    last_alert_time = 0
    frame_batch = []
    frame_count = 0
    last_log_time = time.time()
    label_to_display = ""
    label_to_log = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame from the webcam")
            break

        frame = resize(frame, width=FRAME_SIZE)

        # YOLO crowd detection
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (FRAME_SIZE, FRAME_SIZE), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(ln)

        boxes, confidences, class_ids = [], [], []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > CONFIDENCE_THRESHOLD and class_id == 0:
                    box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (center_x, center_y, width, height) = box.astype("int")
                    x = int(center_x - (width / 2))
                    y = int(center_y - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=CONFIDENCE_THRESHOLD, nms_threshold=NMS_THRESHOLD)
        detections = []
        if len(indices) > 0:
            boxes_for_encoding = [boxes[i] for i in indices.flatten()]
            x, y, w, h = box  # Extract the bounding box dimensions
            centroid = (x + w / 2, y + h / 2)
            features = encoder(frame, boxes_for_encoding)
            for i, box in enumerate(boxes_for_encoding):
                detections.append(
                    Detection(tlwh=box, confidence=confidences[indices.flatten()[i]], feature=features[i], centroid=centroid)
                )

        tracker.predict()
        tracker.update(detections, time=frame_count)

        human_count = sum(1 for track in tracker.tracks if track.is_confirmed() and track.time_since_update <= 1)

        for track in tracker.tracks:
            if track.is_confirmed() and track.time_since_update <= 1:
                bbox = track.to_tlbr()
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

        # Batch violence detection
        frame_batch.append(frame)
        if len(frame_batch) == BATCH_SIZE:
            label_to_display, label_to_log, last_alert_time = alerts(model, frame_batch, detection_times, last_alert_time,last_log_time)
            frame_batch = []

        # Log detection periodically
        current_time = time.time()
        if current_time - last_log_time >= 0.5:  
            logging.info(f"{label_to_log}")
            last_log_time = current_time

        cv2.putText(frame, f"Human Count: {human_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if label_to_display:  
            cv2.putText(frame, f"{label_to_display}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        current_time_str = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        crowd_data_writer.writerow([current_time_str, human_count])

        cv2.imshow("Crowd and Violence Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Real-time processing ended.")

if __name__ == "__main__":
    process_real_time()
    crowd_data_file.close()


