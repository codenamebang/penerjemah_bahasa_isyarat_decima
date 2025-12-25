import pickle
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# =========================
# Load dataset
# =========================
data_dict = pickle.load(open('./data.pickle', 'rb'))

X = np.asarray(data_dict['data'])
y = np.asarray(data_dict['labels'])

# print(y)

print("Data shape :", X.shape)
print("Total label :", len(set(y)))

# =========================
# Encode labels (WAJIB)
# =========================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# =========================
# Train-test split
# =========================
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    shuffle=True,
    stratify=y_encoded
)

# =========================
# XGBoost Model
# =========================
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softmax',   # multiclass classification
    num_class=len(np.unique(y_encoded)),
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)

# =========================
# Training
# =========================
model.fit(x_train, y_train)

# =========================
# Evaluation
# =========================
y_pred = model.predict(x_test)
score = accuracy_score(y_test, y_pred)

print(f"{score * 100:.2f}% of samples were classified correctly!")

# =========================
# Save model + encoder
# =========================
with open('model.p', 'wb') as f:
    pickle.dump(
        {
            'model': model,
            'label_encoder': label_encoder
        },
        f
    )

print("Model XGBoost berhasil disimpan!")
