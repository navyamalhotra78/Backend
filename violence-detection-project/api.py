import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from model import Model
from PIL import Image
import io
import numpy as np
import base64
import logging
from twilio.rest import Client  
from dotenv import load_dotenv
import os
from collections import deque
import time
import httpx

load_dotenv()

TIME_WINDOW = 3 
DETECTION_THRESHOLD = 10  
ALERT_COOLDOWN = 10  
recording = False
video_writer = None
clip_start_time = None

detection_times = deque()
last_alert_time = 0

def check_for_alert(detection_times, last_alert_time,label):
    current_time = time.time()
    
    while detection_times and current_time - detection_times[0] > TIME_WINDOW:
        detection_times.popleft()

    if len(detection_times) >= DETECTION_THRESHOLD:
        if current_time - last_alert_time >= ALERT_COOLDOWN:
            print(f"Alert: {label} detected!")
            logger.info(f"Alert: {label} detected!")
            return current_time  
    
    return last_alert_time

async def send_email_alert(subject, text, to):
    email_data = {
        "subject": subject,
        "text": text,
        "to": to
    }
    async with httpx.AsyncClient() as client:
        response = await client.post("http://localhost:3000/send-email", json=email_data)
        if response.status_code != 200:
            logger.error(f"Failed to send email: {response.text}")
        else:
            logger.info("Email sent successfully")

recording_states = {}

def save_video_clip(label, image_np):
    """Save a 10-second video clip for a specific label."""
    global recording_states
                    
                    # Ensure the label exists in the recording states dictionary
    if label not in recording_states:
        recording_states[label] = {"recording": False, "start_time": None, "video_writer": None}

        current_state = recording_states[label]

        if not current_state["recording"]:
                        # Start recording
            clip_filename = f"processed_data/{label}_clip.avi"
            current_state["video_writer"] = cv2.VideoWriter(
                clip_filename,
                cv2.VideoWriter_fourcc(*'XVID'),
                10,  # FPS
                (image_np.shape[1], image_np.shape[0])  # Frame dimensions
                )
            current_state["recording"] = True
            current_state["start_time"] = time.time()
            logger.info(f"Started recording for {label}: {clip_filename}")
        else:
                        # Continue recording
            elapsed_time = time.time() - current_state["start_time"]
            if elapsed_time < 10:  # Recording duration in seconds
                    current_state["video_writer"].write(image_np)
            else:
                            # Stop recording after 10 seconds
                current_state["video_writer"].release()
                current_state["video_writer"] = None
                current_state["recording"] = False
                current_state["start_time"] = None
                logger.info(f"Recording for {label} completed and saved.")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Twilio credentials
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
client = Client(account_sid, auth_token)

app = FastAPI()
origins = ["http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = Model()

@app.post("/predict/")
async def predict_image(
    file: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    timestamp: str = Form(...),):

    last_alert_time = 0
    current_time = time.time()
    label_to_display = ""
    logger.info("Received a file for prediction.")
    
    # Log file details
    logger.info(f"File received: {file.filename}, size: {file.size}")
    # Read the file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    image_np = np.array(image)

    logger.info(f"Image shape: {image_np.shape}")

    labels = model.predict_batch([image_np])
    for label_dict in labels:
        label = label_dict['label'].lower()

        if 'violence' or 'fight' in label:
            detection_times.append(current_time)
            last_alert_time = check_for_alert(detection_times, last_alert_time,label)
            if len(detection_times) >= DETECTION_THRESHOLD and current_time - last_alert_time >= ALERT_COOLDOWN:
                logger.info("Alert: Violence detected!")
                last_alert_time = current_time
            label_to_display = label 
        elif 'fire' in label:
            detection_times.append(current_time)
            last_alert_time = check_for_alert(detection_times, last_alert_time,label)
            if len(detection_times) >= DETECTION_THRESHOLD and current_time - last_alert_time >= ALERT_COOLDOWN:
                logger.info("Alert: Fire detected!")
                last_alert_time = current_time
            label_to_display = label 
        elif 'crash' in label:
            detection_times.append(current_time)
            last_alert_time = check_for_alert(detection_times, last_alert_time,label)
            if len(detection_times) >= DETECTION_THRESHOLD and current_time - last_alert_time >= ALERT_COOLDOWN:
                logger.info("Alert: Crash detected!")
                last_alert_time = current_time
            label_to_display = label
        else:
            detection_times.clear()


    alert_message = ""
    if label_to_display :
        if label_to_display in ['fight on a street','fire on a street','street violence','car crash','violence in office','fire in office','car on fire']:
            alert_message = f"{label_to_display} alert triggered at {latitude},{longitude} at {timestamp}!"
        
            try:
                message = client.messages.create(
                    body=alert_message,
                    to='+919880923876',  # Your phone number
                    from_='+17753306947'  # Your Twilio phone number
                )
                logger.info(f"Twilio message sent with SID: {message.sid}")
            except Exception as e:
                logger.error(f"Failed to send Twilio message: {e}")
                raise HTTPException(status_code=500, detail="Failed to send alert")
            
            try:
                await send_email_alert(
                subject=f"Alert: {label_to_display}",
                text=alert_message,
                to="navya0807@gmail.com"  # Replace with the recipient's email
                )
            except Exception as e:
                logger.error(f"Failed to send email : {e}")
                raise HTTPException(status_code=500, detail="Failed to send email")
            
            try:
                if label_to_display in ['fight on a street','fire on a street','street violence','car crash','violence in office','fire in office','car on fire']:
                    save_video_clip(label_to_display, image_np)
            except Exception as e:
                logger.error("Error saving video")


    return {"predicted_label": label_to_display, "alert_message": alert_message}
   