import os
import pickle
import cv2
import mediapipe as mp

# =========================
# MediaPipe Setup
# =========================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.3
)

DATA_DIR = './data'

data = []
labels = []

# =========================
# Helper function
# =========================
def extract_hand_features(hand_landmarks):
    """Return 42 features (x,y) normalized"""
    x_ = [lm.x for lm in hand_landmarks.landmark]
    y_ = [lm.y for lm in hand_landmarks.landmark]

    min_x = min(x_)
    min_y = min(y_)

    features = []
    for lm in hand_landmarks.landmark:
        features.append(lm.x - min_x)
        features.append(lm.y - min_y)

    return features  # length = 42

# =========================
# Loop Dataset
# =========================
for label in os.listdir(DATA_DIR):
    label_path = os.path.join(DATA_DIR, label)

    if not os.path.isdir(label_path):
        continue

    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            continue

        data_aux = []

        # Gabungkan handedness + landmarks
        hands_data = list(
            zip(results.multi_handedness, results.multi_hand_landmarks)
        )

        # Urutkan: Left dulu, lalu Right
        hands_data.sort(
            key=lambda x: x[0].classification[0].label
        )

        # =========================
        # KASUS 2 TANGAN
        # =========================
        if len(hands_data) == 2:
            for _, hand_landmarks in hands_data:
                data_aux.extend(extract_hand_features(hand_landmarks))

        # =========================
        # KASUS 1 TANGAN
        # =========================
        elif len(hands_data) == 1:
            _, hand_landmarks = hands_data[0]

            # Anggap sebagai LEFT
            data_aux.extend(extract_hand_features(hand_landmarks))

            # Padding RIGHT hand (42 nol)
            data_aux.extend([0.0] * 42)

        # =========================
        # Safety check
        # =========================
        if len(data_aux) == 84:
            data.append(data_aux)
            labels.append(label)

# =========================
# Save Dataset
# =========================
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Dataset berhasil dibuat")
print("Total sample :", len(data))
print("Jumlah fitur per sample :", len(data[0]) if data else 0)
