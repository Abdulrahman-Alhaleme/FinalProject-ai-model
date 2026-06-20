import serial
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
import time
import os

# إعدادات
SERIAL_PORT = 'COM6'
BAUD_RATE = 921600
RECORD_RATE = 8000
MODEL_RATE = 22050

# مجلدات الداتا
os.makedirs('clean_data/normal', exist_ok=True)
os.makedirs('clean_data/abnormal', exist_ok=True)
os.makedirs('clean_data/noise', exist_ok=True)

def record_from_board(ser):
    ser.write(b'S')
    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line == "RECORDING":
            break
    samples = []
    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line == "DONE":
            break
        if line == "SENDING":
            continue
        try:
            samples.append(int(line))
        except:
            continue
    return np.array(samples, dtype=np.float32)

def process_and_save(raw, folder, count):
    audio = raw - raw.mean()
    audio = audio / (np.abs(audio).max() + 1e-8)
    audio_resampled = librosa.resample(
        audio, orig_sr=RECORD_RATE, target_sr=MODEL_RATE
    )
    noise_sample = audio_resampled[:int(MODEL_RATE * 0.5)]
    audio_clean = nr.reduce_noise(
        y=audio_resampled,
        sr=MODEL_RATE,
        y_noise=noise_sample,
        prop_decrease=0.8
    )
    filename = f"{folder}/sample_{count:04d}.wav"
    sf.write(filename, audio_clean, MODEL_RATE)
    return filename

print("Connecting...")
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=60)
time.sleep(2)
ser.flushInput()
print("Connected!\n")

while True:
    print("\nChoose type:")
    print("1 = normal")
    print("2 = abnormal")
    print("3 = noise")
    print("q = quit")

    choice = input("Choice: ").strip()

    if choice == 'q':
        break
    elif choice == '1':
        folder = 'clean_data/normal'
        label = 'normal'
    elif choice == '2':
        folder = 'clean_data/abnormal'
        label = 'abnormal'
    elif choice == '3':
        folder = 'clean_data/noise'
        label = 'noise'
    else:
        print("Wrong choice!")
        continue

    # كم عينة تبي؟
    try:
        num_samples = int(input(f"How many {label} samples? "))
    except:
        print("Invalid number!")
        continue

    print(f"\nReady to record {num_samples} {label} samples automatically!")
    input("Press Enter when ready and start playing the sound...")

    recorded = 0
    errors = 0

    while recorded < num_samples:
        count = len([f for f in os.listdir(folder) if f.endswith('.wav')])
        print(f"\n[{recorded+1}/{num_samples}] Recording {label}...")

        try:
            raw = record_from_board(ser)

            if len(raw) < 1000:
                print("⚠️ Short recording, skipping...")
                errors += 1
                if errors > 5:
                    print("Too many errors, stopping!")
                    break
                continue

            filename = process_and_save(raw, folder, count)
            recorded += 1
            errors = 0
            print(f"✅ Saved: {filename} ({recorded}/{num_samples})")

            # انتظر ثانية بين كل تسجيل
            time.sleep(1)

        except Exception as e:
            print(f"❌ Error: {e}")
            errors += 1
            if errors > 5:
                print("Too many errors, stopping!")
                break

    total = len([f for f in os.listdir(folder) if f.endswith('.wav')])
    print(f"\n✅ Done! Total {label}: {total} samples")

ser.close()
print("\nFinished!")