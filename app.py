import eventlet
eventlet.monkey_patch()
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import mediapipe as mp
import numpy as np
import joblib
import base64
import io
from PIL import Image
import time
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_key_123')
socketio = SocketIO(app, cors_allowed_origins="*")

# Load model
model = joblib.load("Models/svm_winner.pkl")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
def process_and_draw(frame, hand_landmarks, width, height):
    """Process landmarks and ensure drawings persist"""
    # Make a writable copy of the frame
    frame_copy = frame.copy()
    
    # Convert to RGB for MediaPipe drawing
    rgb_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
    
    # Draw landmarks (using the RGB frame)
    mp.solutions.drawing_utils.draw_landmarks(
        rgb_frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
        mp.solutions.drawing_styles.get_default_hand_connections_style()
    )
    
    # Convert back to BGR for OpenCV text
    frame_with_drawings = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    
    return frame_with_drawings

def process_hand_landmarks(landmarks):
    """Normalize and prepare hand landmarks for prediction"""
    landmarks = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])
    
    # Center landmarks on wrist
    wrist = landmarks[0]
    landmarks -= wrist
    
    # Normalize scale using middle finger length
    scale = np.linalg.norm(landmarks[12])
    landmarks /= scale
    
    return landmarks.flatten().reshape(1, -1)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('server_ready', {'status': 'ready'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('frame')
def handle_frame(data):
    try:
        start_time = time.time()
        frame_id = data.get('frameId', 0)
        
        # Decode the frame
        frame_bytes = base64.b64decode(data['frame'].split(',')[1])
        image = Image.open(io.BytesIO(frame_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        height, width = frame.shape[:2]
        
        # Process frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Prepare landmarks for prediction
                features = process_hand_landmarks(hand_landmarks)
                prediction = model.predict(features)[0]
                
                # Draw landmarks on frame
                
                frame = process_and_draw(frame, hand_landmarks, width, height)
                
                cv2.putText(frame, f'Prediction: {prediction}', (200, 300),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Encode processed frame
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = base64.b64encode(buffer).decode('utf-8')
                
                processing_time = (time.time() - start_time) * 1000  # ms
                
                emit('processed_frame', {
                    'frame': f'data:image/jpeg;base64,{frame_bytes}',
                    'prediction': str(prediction),
                    'hand_detected': True,
                    'frameId': frame_id,
                    'processing_time': f"{processing_time:.2f}ms"
                })
                return
        
        # No hand detected - send original frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = base64.b64encode(buffer).decode('utf-8')
        
        processing_time = (time.time() - start_time) * 1000  # ms
        
        emit('original_frame', {
            'frame': f'data:image/jpeg;base64,{frame_bytes}',
            'hand_detected': False,
            'frameId': frame_id,
            'processing_time': f"{processing_time:.2f}ms"
        })

    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        emit('processing_error', {
            'error': str(e),
            'frameId': data.get('frameId', 0)
        })

if __name__ == '__main__':
    
    socketio.run(app, host='0.0.0.0', port="5000", debug=True)
