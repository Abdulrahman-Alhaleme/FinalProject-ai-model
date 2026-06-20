import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import urllib.request

# تحميل YAMNet TFLite
YAMNET_MODEL_URL = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/audio_classification/android/lite-model_yamnet_classification_tflite_1.tflite"
YAMNET_PATH = "yamnet.tflite"

if not os.path.exists(YAMNET_PATH):
    print("Downloading YAMNet TFLite model...")
    urllib.request.urlretrieve(YAMNET_MODEL_URL, YAMNET_PATH)
    print("Downloaded!")

# تحميل الموديل
interpreter = tf.lite.Interpreter(model_path=YAMNET_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input shape: {input_details[0]['shape']}")
print(f"Output shape: {output_details[0]['shape']}")

# إعدادات
CLEAN_DATA = {
    0: 'clean_data/abnormal',
    1: 'clean_data/noise',
    2: 'clean_data/normal'
}
CLASS_NAMES = ['abnormal', 'noise', 'normal']
SAMPLE_RATE = 16000
DURATION = 10

def load_audio(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        target_length = SAMPLE_RATE * DURATION
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        return audio.astype(np.float32)
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_yamnet_features(audio):
    # YAMNet يحتاج قطع من 0.975 ثانية
    waveform_length = int(SAMPLE_RATE * 0.975)
    features = []

    for i in range(0, len(audio) - waveform_length, waveform_length // 2):
        chunk = audio[i:i + waveform_length]
        if len(chunk) < waveform_length:
            break

        interpreter.set_tensor(input_details[0]['index'], chunk)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        features.append(output[0])

    if len(features) == 0:
        return None

    return np.mean(features, axis=0)

# تحميل الداتا
print("\n=== Loading real-world data ===")
all_features = []
all_labels = []

for label_idx, folder in CLEAN_DATA.items():
    files = [f for f in os.listdir(folder) if f.endswith('.wav')]
    print(f"{CLASS_NAMES[label_idx]}: {len(files)} samples")
    for file in tqdm(files, desc=CLASS_NAMES[label_idx]):
        audio = load_audio(os.path.join(folder, file))
        if audio is not None:
            features = get_yamnet_features(audio)
            if features is not None:
                all_features.append(features)
                all_labels.append(label_idx)

X = np.array(all_features)
y = np.array(all_labels)

print(f"\nTotal: {len(y)} samples")
print(f"Feature shape: {X.shape}")

# تقسيم الداتا
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training: {len(X_train)} | Testing: {len(X_test)}")

# بناء classifier
print("\nBuilding classifier...")
classifier = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(3, activation='softmax')
])

classifier.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

classifier.summary()

# تدريب
print("\nTraining...")
history = classifier.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True
        )
    ]
)

# تقييم
print("\nEvaluating...")
y_pred = np.argmax(classifier.predict(X_test), axis=1)

print("\n=== YAMNet Results ===")
print(classification_report(
    y_test, y_pred,
    target_names=CLASS_NAMES,
    zero_division=0
))

# رسم
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Val')
ax1.set_title('YAMNet Accuracy')
ax1.legend()
ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Val')
ax2.set_title('YAMNet Loss')
ax2.legend()
plt.tight_layout()
plt.savefig('yamnet_results.png')
plt.show()

classifier.save('yamnet_classifier.keras')
print("\n✅ Saved: yamnet_classifier.keras")