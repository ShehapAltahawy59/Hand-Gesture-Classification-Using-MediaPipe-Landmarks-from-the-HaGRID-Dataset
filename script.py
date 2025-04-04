import cv2
import mediapipe as mp
import numpy as np
import joblib  # For loading the SVM model

# Load trained SVM model
svm_model = joblib.load("Models/svm_winner.pkl")

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Start video capture
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract (x, y, z) coordinates
            landmarks = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])

            # Normalize: Recenter based on wrist position (landmark 0)
            wrist_x, wrist_y, wrist_z = landmarks[0]
            landmarks[:, 0] -= wrist_x  # Center x-coordinates
            landmarks[:, 1] -= wrist_y  # Center y-coordinates
            # DO NOT modify the z-coordinates

            # Scale only x and y using the mid-finger tip (landmark 12)
            mid_finger_x, mid_finger_y, _ = landmarks[12]  # Ignore z for scaling
            scale_factor = np.sqrt(mid_finger_x**2 + mid_finger_y**2)
            landmarks[:, 0] /= scale_factor  # Scale x
            landmarks[:, 1] /= scale_factor  # Scale y
            # DO NOT scale z-coordinates

            # Flatten the features for SVM
            features = landmarks.flatten().reshape(1, -1)
            if not np.isnan(features).any():
            
                # Predict using SVM
                prediction = svm_model.predict(features)[0]

                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Display the prediction on the frame
                cv2.putText(frame, f'Prediction: {prediction}', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Show the frame
    cv2.imshow("Hand Gesture Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
