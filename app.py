
import eventlet
eventlet.monkey_patch()
import os
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import mediapipe as mp
import numpy as np
import base64
import joblib




app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load model
model = joblib.load("Models/svm_winner.pkl")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print("Client connected")
    emit('status', {'message': 'Connected to gesture recognition server'})

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")

@socketio.on('frame')
def handle_frame(data):
    try:
        # Initialize MediaPipe drawing utilities
        mp_drawing = mp.solutions.drawing_utils
        
        # Convert base64 image to OpenCV format
        header, encoded = data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])
                
                # Normalize landmarks
                wrist_x, wrist_y, wrist_z = landmarks[0]
                landmarks[:, 0] -= wrist_x
                landmarks[:, 1] -= wrist_y
                mid_finger_x, mid_finger_y, _ = landmarks[12]
                scale_factor = np.sqrt(mid_finger_x**2 + mid_finger_y**2)
                landmarks[:, 0] /= scale_factor
                landmarks[:, 1] /= scale_factor
                
                # Predict
                features = landmarks.flatten().reshape(1, -1)
                
                if not np.isnan(features).any():
                    prediction = model.predict(features)[0]
                    confidence = np.max(model.predict_proba(features))
                    
                    # Draw landmarks and prediction on frame
                    mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                    )
                    
                    # Display prediction text
                    cv2.putText(frame, f'Prediction: {prediction}', (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, f'Confidence: {confidence:.2f}', (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    # Convert frame back to base64
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    emit('prediction', {
                        'gesture': str(prediction),
                        'confidence': float(confidence),
                        'frame': f"data:image/jpeg;base64,{frame_base64}"
                    })
                    print(prediction)
                else:
                    emit('prediction', {
                        'gesture': 'Invalid hand data',
                        'confidence': 0.0
                    })
                    print("no hand")
        else:
            emit('prediction', {
                'gesture': 'No hand detected',
                'confidence': 0.0
            })
            
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        emit('error', {'message': str(e)})

