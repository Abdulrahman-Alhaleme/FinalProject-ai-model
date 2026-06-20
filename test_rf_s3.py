import serial
import numpy as np
import librosa
import joblib
import time
import requests
import soundfile as sf
import noisereduce as nr

# تحميل الموديل
model = joblib.load('sound_model_realworld.pkl')
le = joblib.load('label_encoder_realworld.pkl')

# إعدادات
SERIAL_PORT = 'COM6'
BAUD_RATE = 921600
RECORD_RATE = 8000
MODEL_RATE = 22050

# إعدادات Telegram
TELEGRAM_TOKEN = "8658739506:AAF0X3oVvLkkAO77z2u8Y82eFDU6DKhjosg"
TELEGRAM_CHAT_ID = "1783822715"

# المسارات
WAV_PATH = r'C:\Users\ONE BY ONE\OneDrive\Desktop\FinalProject\temp_recording.wav'
CLEAN_WAV_PATH = r'C:\Users\ONE BY ONE\OneDrive\Desktop\FinalProject\temp_clean.wav'

def send_telegram(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message
        })
        print("📨 تم إرسال الإشعار على Telegram!")
    except Exception as e:
        print(f"❌ فشل إرسال الإشعار: {e}")

def record_from_board(ser):
    ser.write(b'S')
    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line == "RECORDING":
            print("🎤 جاري التسجيل...")
            break
    samples = []
    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line == "DONE":
            break
        if line == "SENDING":
            print("📡 جاري الإرسال...")
            continue
        try:
            samples.append(int(line))
        except:
            continue
    return np.array(samples, dtype=np.float32)

def save_and_clean(raw):
    # تطبيع الإشارة
    audio = raw - raw.mean()
    audio = audio / (np.abs(audio).max() + 1e-8)

    # رفع الـ sample rate
    audio_resampled = librosa.resample(
        audio, orig_sr=RECORD_RATE, target_sr=MODEL_RATE
    )

    # حفظ الصوت الأصلي
    sf.write(WAV_PATH, audio_resampled, MODEL_RATE)

    # إزالة الضجيج
    noise_sample = audio_resampled[:int(MODEL_RATE * 0.5)]
    audio_clean = nr.reduce_noise(
        y=audio_resampled,
        sr=MODEL_RATE,
        y_noise=noise_sample,
        prop_decrease=0.8
    )

    # حفظ الصوت النظيف
    sf.write(CLEAN_WAV_PATH, audio_clean, MODEL_RATE)
    print("💾 تم حفظ الصوت النظيف")

    return audio_clean

def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    rms = np.mean(librosa.feature.rms(y=audio))
    centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
    return np.concatenate([mfcc_mean, mfcc_std, [rms, centroid, zcr]])

print("=== نظام مراقبة الماكينات ===")
print(f"جاري الاتصال على {SERIAL_PORT}...")

last_alert_time = 0
ALERT_COOLDOWN = 60

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=60)
    time.sleep(2)
    ser.flushInput()
    print("✅ متصل! النظام يراقب الآن...\n")
    send_telegram("✅ نظام المراقبة بدأ يشتغل!")

    while True:
        print(f"🎤 [{time.strftime('%H:%M:%S')}] جاري التسجيل...")

        # سجّل الصوت
        raw = record_from_board(ser)
        print(f"استقبلنا {len(raw)} عينة")

        # احفظ ونظّف
        audio_clean = save_and_clean(raw)

        # استخرج الـ features وصنّف
        features = extract_features(audio_clean, MODEL_RATE).reshape(1, -1)
        prediction = model.predict(features)[0]
        label = le.inverse_transform([prediction])[0]
        confidence = model.predict_proba(features)[0][prediction] * 100

        print(f"🔍 النتيجة: {label.upper()} | الثقة: {confidence:.1f}%")

        # إرسال النتيجة للشاشة
        if label == 'abnormal':
            ser.write(b'A')
        elif label == 'normal':
            ser.write(b'N')
        else:
            ser.write(b'X')

        current_time = time.time()

        if label == 'abnormal':
            print("🚨 تحذير: صوت غير طبيعي!")
            if current_time - last_alert_time > ALERT_COOLDOWN:
                send_telegram(
                    f"🚨 تحذير! كشف النظام صوت غير طبيعي!\n"
                    f"🕐 الوقت: {time.strftime('%H:%M:%S')}"
                )
                last_alert_time = current_time
        elif label == 'normal':
            print("✅ الماكينة تعمل بشكل طبيعي")
        else:
            print("🔇 ضوضاء عادية")

        print("-" * 40)
        time.sleep(1)

except serial.SerialException:
    print(f"❌ ما قدرنا نتصل على {SERIAL_PORT}")
except KeyboardInterrupt:
    print("\n⛔ تم إيقاف النظام")
    send_telegram("⛔ تم إيقاف نظام المراقبة")
    ser.close()