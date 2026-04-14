import serial
import pandas as pd
import numpy as np
import os
import time

# --- Configuration ---
PORT = 'COM6'          # STM32 COM port
BAUD = 115200          # UART baud rate
BUFFER_SIZE = 256      # number of samples per trial
FS = 5000              # sampling rate in Hz (set to STM32 timer rate)
THRESHOLD = 1000       # threshold for duration calculation
DELAY = 1.5            # seconds to wait before next trial

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

            # Peak: maximum ADC value (strongest vibration amplitude)
            peak = np.max(samples)

            # RMS: root mean square, overall vibration energy
            rms = np.sqrt(np.mean(samples**2))

            # DurationAboveThreshold: number of samples above threshold
            # shows how long vibration stays strong
            duration = np.sum(samples > THRESHOLD)

            # Remove DC offset before FFT (center signal around zero)
            samples_centered = samples - np.mean(samples)

            # Apply window to reduce FFT leakage
            window = np.hanning(BUFFER_SIZE)
            samples_windowed = samples_centered * window

            # FFT: convert time-domain signal into frequency-domain
            fft = np.fft.rfft(samples_windowed)
            magnitude = np.abs(fft)

            # Frequency bins
            freqs = np.fft.rfftfreq(BUFFER_SIZE, d=1/FS)

            # DominantFreqHz: strongest frequency component (ignoring DC)
            dominant_bin = np.argmax(magnitude[1:]) + 1
            dominant_freq = freqs[dominant_bin]

            # SpectralCentroidHz: weighted average frequency
            # tells whether spectrum is "high-pitched" (near impact) or "low-pitched" (far impact)
            spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)

            # DecayRate: ratio of average amplitude in last 50 samples to peak
            # shows how quickly vibration dies out (near impacts decay faster)
            decay_rate = np.mean(samples_centered[-50:]) / peak

            print(f"Trial {trial_num}: Peak={peak}, RMS={rms:.2f}, Duration={duration}, "
                  f"DominantFreq={dominant_freq:.1f} Hz, Centroid={spectral_centroid:.1f} Hz, DecayRate={decay_rate:.3f}")

            # Save trial data into dictionary
            trial_data = {
                "Trial": trial_num,
                "Peak": peak,
                "RMS": rms,
                "DurationAboveThreshold": duration,
                "DominantFreqHz": dominant_freq,
                "SpectralCentroidHz": spectral_centroid,
                "DecayRate": decay_rate,
                **{f"Sample_{i}": samples[i] for i in range(BUFFER_SIZE)}
            }
            all_trials.append(trial_data)
            trial_num += 1

            # Delay to avoid capturing second bounce
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
