import pickle
import cv2
import mediapipe as mp
import numpy as np

# =========================
# Load model + label encoder
# =========================
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
label_encoder = model_dict['label_encoder']

# =========================
# OpenCV Camera
# =========================
cap = cv2.VideoCapture(0)

# =========================
# MediaPipe Setup
# =========================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3
)

# =========================
# Helper function
# =========================
def extract_hand_features(hand_landmarks):
    """Extract 42 features (x,y) normalized"""
    x_ = [lm.x for lm in hand_landmarks.landmark]
    y_ = [lm.y for lm in hand_landmarks.landmark]

    min_x = min(x_)
    min_y = min(y_)

    features = []
    for lm in hand_landmarks.landmark:
        features.append(lm.x - min_x)
        features.append(lm.y - min_y)

    return features  # 42 features

# =========================
# Main Loop
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    data_aux = []
    x_all, y_all = [], []

    if results.multi_hand_landmarks:

        # Draw landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        # Combine handedness + landmarks
        hands_data = list(
            zip(results.multi_handedness, results.multi_hand_landmarks)
        )

        # Sort Left → Right
        hands_data.sort(
            key=lambda x: x[0].classification[0].label
        )

        # =========================
        # 2 hands
        # =========================
        if len(hands_data) == 2:
            for _, hand_landmarks in hands_data:
                data_aux.extend(extract_hand_features(hand_landmarks))

        # =========================
        # 1 hand
        # =========================
        elif len(hands_data) == 1:
            _, hand_landmarks = hands_data[0]
            data_aux.extend(extract_hand_features(hand_landmarks))
            data_aux.extend([0.0] * 42)  # padding for second hand

        # =========================
        # Prediction
        # =========================
        if len(data_aux) == 84:
            prediction = model.predict([np.asarray(data_aux)])

            # Decode label
            predicted_label = label_encoder.inverse_transform(prediction)[0]

            # Ambil huruf saja (misal: Sibi_A → A)
            predicted_character = predicted_label.replace("Sibi_", "")

            # Bounding box
            for _, hand_landmarks in hands_data:
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
                predicted_character,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3,
                cv2.LINE_AA
            )

    cv2.imshow('Sign Language Detection (XGBoost)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# Cleanup
# =========================
cap.release()
cv2.destroyAllWindows()
