import cv2
import mediapipe as mp
import pandas as pd
import os


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

GESTURE_NAME = "forward"  
SAVE_DIR = "dataset"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

all_landmarks = []

cap = cv2.VideoCapture(0)
print("Collecting gesture:", GESTURE_NAME)
print("Press 's' to save a frame, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if len(landmarks) == 63:
                cv2.putText(frame, 'Ready to Save (Press S)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Collecting Gesture Data", frame)

    key = cv2.waitKey(1)
    if key == ord('s') and results.multi_hand_landmarks:
        all_landmarks.append(landmarks)
        print(f"Saved frame: {len(all_landmarks)}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


df = pd.DataFrame(all_landmarks)
df['label'] = GESTURE_NAME
df.to_csv(os.path.join(SAVE_DIR, f"{GESTURE_NAME}.csv"), index=False)
print("Data saved.")
