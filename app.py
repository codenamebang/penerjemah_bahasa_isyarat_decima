from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import pickle

# =========================
# Flask app
# =========================
app = Flask(__name__)

# =========================
# Load model + label encoder
# =========================
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']
label_encoder = model_dict['label_encoder']

print("Loaded classes:", label_encoder.classes_)

# =========================
# MediaPipe Hands
# =========================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# Camera
# =========================
cap = cv2.VideoCapture(0)

# =========================
# Feature extractor
# =========================
def extract_hand_features(hand_landmarks):
    x_ = [lm.x for lm in hand_landmarks.landmark]
    y_ = [lm.y for lm in hand_landmarks.landmark]

    min_x = min(x_)
    min_y = min(y_)

    features = []
    for lm in hand_landmarks.landmark:
        features.append(lm.x - min_x)
        features.append(lm.y - min_y)

    return features  # 42 fitur / tangan

# =========================
# Video generator (Flask)
# =========================
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        data_aux = []
        x_all, y_all = [], []

        if results.multi_hand_landmarks:

            hands_data = list(zip(
                results.multi_handedness,
                results.multi_hand_landmarks
            ))

            # Urutkan tangan: Left → Right
            hands_data.sort(
                key=lambda x: x[0].classification[0].label
            )

            # =========================
            # 2 tangan
            # =========================
            if len(hands_data) == 2:
                for _, hand_landmarks in hands_data:
                    data_aux.extend(extract_hand_features(hand_landmarks))

            # =========================
            # 1 tangan → padding
            # =========================
            elif len(hands_data) == 1:
                _, hand_landmarks = hands_data[0]
                data_aux.extend(extract_hand_features(hand_landmarks))
                data_aux.extend([0.0] * 42)

            # =========================
            # Prediction
            # =========================
            if len(data_aux) == 84:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_label = label_encoder.inverse_transform(prediction)[0]

                # Draw landmarks + bbox
                for _, hand_landmarks in hands_data:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
                    for lm in hand_landmarks.landmark:
                        x_all.append(lm.x)
                        y_all.append(lm.y)

                x1 = int(min(x_all) * W) - 10
                y1 = int(min(y_all) * H) - 10
                x2 = int(max(x_all) * W) + 10
                y2 = int(max(y_all) * H) + 10

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(
                    frame,
                    predicted_label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 0),
                    3
                )

        # =========================
        # Encode frame for Flask
        # =========================
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# =========================
# Flask routes
# =========================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# =========================
# Run app
# =========================
if __name__ == '__main__':
    app.run(debug=True)


# =========================
# RUN FLASK SERVER
# =========================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        threaded=True
    )
