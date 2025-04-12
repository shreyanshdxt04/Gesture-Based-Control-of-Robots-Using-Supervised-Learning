import cv2
import mediapipe as mp
import numpy as np
import joblib
import pygame
import threading
import time

model = joblib.load('gesture_knn_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

current_gesture = "stop"

def gesture_thread():
    global current_gesture
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                if len(landmarks) == 63:
                    input_data = np.array(landmarks).reshape(1, -1)
                    prediction = model.predict(input_data)[0]
                    current_gesture = label_encoder.inverse_transform([prediction])[0]

                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(image, f'Gesture: {current_gesture}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Gesture Detection', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

threading.Thread(target=gesture_thread, daemon=True).start()

pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Gesture Controlled Robot Simulation")

clock = pygame.time.Clock()

robot = pygame.Rect(400, 300, 50, 50)
speed = 5

running = True
while running:
    screen.fill((80, 80, 80))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if current_gesture == "forward":
        robot.y -= speed
    elif current_gesture == "backward":
        robot.y += speed
    elif current_gesture == "left":
        robot.x -= speed
    elif current_gesture == "right":
        robot.x += speed
    elif current_gesture == "stop":
        pass  

    pygame.draw.rect(screen, (0, 255, 0), robot)
    pygame.display.flip()
    clock.tick(30)

pygame.quit()
