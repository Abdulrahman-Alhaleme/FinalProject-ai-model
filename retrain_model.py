import os
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
from tqdm import tqdm

# مجلدات الداتا
CLEAN_DATA = {
    'normal':   'clean_data/normal',
    'abnormal': 'clean_data/abnormal',
    'noise':    'clean_data/noise'
}

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=22050, duration=10)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        rms = np.mean(librosa.feature.rms(y=audio))
        centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
        return np.concatenate([mfcc_mean, mfcc_std, [rms, centroid, zcr]])
    except Exception as e:
        print(f"Error: {e}")
        return None

all_features = []
all_labels = []

# تحميل الداتا
print("=== Loading clean data ===")
for label, folder in CLEAN_DATA.items():
    files = [f for f in os.listdir(folder) if f.endswith('.wav')]
    print(f"{label}: {len(files)} samples")
    for file in tqdm(files, desc=label):
        features = extract_features(os.path.join(folder, file))
        if features is not None:
            all_features.append(features)
            all_labels.append(label)

X = np.array(all_features)
y = np.array(all_labels)

print(f"\nTotal: {len(y)} samples")
print(f"normal:   {np.sum(y=='normal')}")
print(f"abnormal: {np.sum(y=='abnormal')}")
print(f"noise:    {np.sum(y=='noise')}")

# توازن الداتا - نستخدم كل الداتا
TARGET = 160

print(f"\nBalancing to: {TARGET} per class")

X_bal = np.concatenate([
    resample(X[y=='normal'],   replace=False, n_samples=TARGET, random_state=42),
    resample(X[y=='abnormal'], replace=False, n_samples=TARGET, random_state=42),
    resample(X[y=='noise'],    replace=True,  n_samples=TARGET, random_state=42)
])
y_bal = np.concatenate([
    np.array(['normal']   * TARGET),
    np.array(['abnormal'] * TARGET),
    np.array(['noise']    * TARGET)
])

print(f"Total after balancing: {len(y_bal)}")

# تدريب
le = LabelEncoder()
y_encoded = le.fit_transform(y_bal)

X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"Training: {len(X_train)} | Testing: {len(X_test)}")
print("\nTraining model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n=== Results ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# حفظ
joblib.dump(model, 'sound_model_realworld.pkl')
joblib.dump(le,    'label_encoder_realworld.pkl')
print("\n✅ Model saved: sound_model_realworld.pkl")