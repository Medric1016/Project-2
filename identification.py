import serial
import pandas as pd
import numpy as np
import os
import time

# --- Configuration ---
PORT = 'COM6'          # STM32 COM port
BAUD = 115200          # UART baud rate
BUFFER_SIZE = 256      # number of samples per trial
FS = 8000              # sampling rate in Hz (set to STM32 timer rate)
THRESHOLD = 1000       # threshold for duration calculation
DELAY = 1.5            # seconds to wait before next trial

# --- Classification thresholds ---
CENTROID_THRESHOLD = 950   # Hz, separates 10 cm vs 30 cm distance
RMS_THRESHOLD = 300        # separates height at 10 cm distance
DECAY_THRESHOLD = -0.034    # separates height at 30 cm distance

def classify_trial(features):
    centroid = features["SpectralCentroidHz"]
    rms = features["RMS"]
    decay = features["DecayRate"]

    # Step 1: distance
    if centroid < CENTROID_THRESHOLD:
        distance = "D10"
        # Step 2: height (use RMS)
        if rms < RMS_THRESHOLD:
            height = "H10"
        else:
            height = "H30"
    else:
        distance = "D30"
        # Step 2: height (use DecayRate)
        if decay < DECAY_THRESHOLD:
            height = "H10"
        else:
            height = "H30"

    return f"{height}_{distance}"

# --- Serial setup ---
ser = serial.Serial(port=PORT, baudrate=BAUD, timeout=1)

all_trials = []
trial_num = 1

try:
    while True:
        print(f"Waiting for trial {trial_num} data...")
        line = ser.readline().decode('utf-8').strip()

        if not line:
            continue

        samples = [int(x) for x in line.split(',') if x.strip().isdigit()]
        if len(samples) == BUFFER_SIZE:
            samples = np.array(samples)

            # --- Feature extraction ---
            peak = np.max(samples)
            rms = np.sqrt(np.mean(samples**2))
            duration = np.sum(samples > THRESHOLD)

            samples_centered = samples - np.mean(samples)
            window = np.hanning(BUFFER_SIZE)
            samples_windowed = samples_centered * window

            fft = np.fft.rfft(samples_windowed)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(BUFFER_SIZE, d=1/FS)

            dominant_bin = np.argmax(magnitude[1:]) + 1
            dominant_freq = freqs[dominant_bin]
            spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            decay_rate = np.mean(samples_centered[-50:]) / peak

            features = {
                "Peak": peak,
                "RMS": rms,
                "DurationAboveThreshold": duration,
                "DominantFreqHz": dominant_freq,
                "SpectralCentroidHz": spectral_centroid,
                "DecayRate": decay_rate
            }

            category = classify_trial(features)

            print(f"Trial {trial_num}: Peak={peak}, RMS={rms:.2f}, Duration={duration}, "
                  f"DominantFreq={dominant_freq:.1f} Hz, Centroid={spectral_centroid:.1f} Hz, "
                  f"DecayRate={decay_rate:.3f} -> Classified as {category}")

            # Save trial data into dictionary
            trial_data = {
                "Trial": trial_num,
                **features,
                **{f"Sample_{i}": samples[i] for i in range(BUFFER_SIZE)},
                "Category": category
            }
            all_trials.append(trial_data)
            trial_num += 1

            time.sleep(DELAY)

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    if all_trials:
        df = pd.DataFrame(all_trials)
        save_path = r"C:\Users\yapzy\OneDrive\Year 4 Sem 1\TRC3500\Project 2\adc_trials_two.xlsx"
        df.to_excel(save_path, index=False)
        print("Data saved to:", save_path)
    ser.close()
