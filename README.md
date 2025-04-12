
# Gesture-Based Control of Robots Using Supervised Learning 🤖✋

This project enables **real-time hand gesture recognition** using **Mediapipe** and **supervised learning (KNN)** to control a **virtual robot in a GUI simulation**. It's an interactive, hands-free way to control motion using only your hand gestures.

---

## 🚀 Features

- Collect custom hand gesture data using your webcam
- Extract 3D hand landmarks using Mediapipe
- Train a machine learning model to classify gestures
- Use the trained model in **real-time** to control a virtual robot
- Simulate robot motion in a Pygame GUI

---

## 📂 Project Structure

```
gesture-control-robot/
│
├── dataset/                    # Collected gesture CSVs
│   ├── stop.csv
│   ├── forward.csv
│   └── ...
│
├── gesture_data_collection.py  # Script to collect gesture data
├── gesture_preprocess.py       # Preprocess data & train-test split
├── gesture_train.py            # Train KNN model
├── gesture4.py                 # Real-time webcam gesture prediction
├── gesture_simulation_gui.py   # Pygame simulation (virtual robot)
│
├── gesture_knn_model.pkl       # Trained ML model
├── label_encoder.pkl           # Label encoder for gestures
│
└── README.md
```

---

## 📦 Installation

```bash
pip install mediapipe opencv-python scikit-learn pandas numpy joblib pygame
```

---

## 🧠 How It Works

1. **Data Collection**  
   Run `gesture_data_collection.py` for each gesture (like `stop`, `forward`, `left`) to collect data samples using your webcam.

2. **Preprocessing**  
   Combine and encode gesture data using `gesture_preprocess.py`.

3. **Model Training**  
   Train a K-Nearest Neighbors (KNN) model with `gesture_train.py`. The model is saved to `gesture_knn_model.pkl`.

4. **Real-Time Prediction**  
   Use `gesture4.py` to recognize gestures live using the webcam.

5. **Robot Simulation**  
   Run `gesture_simulation_gui.py` to control a green robot in a Pygame window using your gestures.

---

## 🕹️ Default Gestures

| Gesture    | Action              |
|------------|---------------------|
| `stop`     | Stops the robot     |
| `forward`  | Moves up            |
| `backward` | Moves down          |
| `left`     | Moves left          |
| `right`    | Moves right         |

You can customize gestures in both training and logic.

---

## 📸 Screenshots

| Webcam Prediction | Robot Simulation |
|-------------------|------------------|
| ![prediction](https://via.placeholder.com/300x200.png?text=Webcam+Gesture) | ![simulation](https://via.placeholder.com/300x200.png?text=Robot+Simulation) |

---

## 📚 Technologies Used

- [Python](https://www.python.org/)
- [OpenCV](https://opencv.org/)
- [Mediapipe](https://google.github.io/mediapipe/)
- [Scikit-learn](https://scikit-learn.org/)
- [Pygame](https://www.pygame.org/)

---

## 📌 To-Do

- [ ] Add gesture recording UI
- [ ] Save model training metrics
- [ ] Arduino/Serial integration for real robot



---

## 🙋‍♂️ Author

Made with 💚 by Shreyansh Dixit  
Feel free to fork and improve!
