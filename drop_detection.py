import serial
import pandas as pd
import numpy as np
import os
import time
 
# --- Configuration ---
PORT        = 'COM7'
BAUD        = 115200
DETECT_WINDOW = 5.0   # seconds to wait for a signal per trial
DELAY         = 1.0   # seconds between trials
TRIALS_PER_CLASS = 50 # 50 drop + 50 no-drop = 100 total
SAVE_PATH   = r"C:\Users\TRC 3500\strongperformance.xlsx"
 
# --- Serial setup ---
ser = serial.Serial(port=PORT, baudrate=BAUD, timeout=1)
ser.reset_input_buffer()
 
all_trials = []
trial_num  = 1
 
print("=" * 50)
print("Binary Drop Detection — 100 Trial Session")
print(f"  Trials 1–{TRIALS_PER_CLASS}:   DROP the object (Actual = 1)")
print(f"  Trials {TRIALS_PER_CLASS+1}–{TRIALS_PER_CLASS*2}: DO NOT drop (Actual = 0)")
print(f"  Each trial: {DETECT_WINDOW}s window, then {DELAY}s gap")
print("=" * 50)
input("\nPress Enter to begin...\n")
 
try:
    for phase, actual_label in [(1, 1), (2, 0)]:
        if phase == 1:
            print("\n" + "="*50)
            print(f"PHASE 1 — DROP the coin for each trial ({TRIALS_PER_CLASS} trials)")
            print("="*50 + "\n")
        else:
            print("\n" + "="*50)
            print(f"PHASE 2 — DO NOT drop, hold still ({TRIALS_PER_CLASS} trials)")
            print("="*50 + "\n")
            input("Press Enter when ready for Phase 2...\n")
 
        for i in range(TRIALS_PER_CLASS):
            print(f"Trial {trial_num:>3} | Actual: {'Drop (1)' if actual_label else 'No Drop (0)'} | "
                  f"Waiting {DETECT_WINDOW}s for signal...", end=' ', flush=True)
 
            # Listen for DETECT_WINDOW seconds
            ser.reset_input_buffer()
            detected   = False
            start_time = time.time()
 
            while time.time() - start_time < DETECT_WINDOW:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    detected = True
                    break
 
            predicted = 1 if detected else 0
            result_str = "DETECTED (1)" if predicted else "NOT DETECTED (0)"
            correct    = "✓" if predicted == actual_label else "✗"
            print(f"→ {result_str}  {correct}")
 
            trial_data = {
                "Predicted Category": predicted,
                "True Category":      actual_label,
            }
            all_trials.append(trial_data)
            trial_num += 1
 
            time.sleep(DELAY)
 
    print("\nAll 100 trials complete!")
 
except KeyboardInterrupt:
    print("\nStopped early by user.")
 
finally:
    if all_trials:
        df = pd.DataFrame(all_trials)
        cols = ['Predicted Category', 'True Category'] + \
               [c for c in df.columns if c not in ('Predicted Category', 'True Category')]
        df = df[cols]
        df.to_excel(SAVE_PATH, index=False)
        print(f"Data saved to: {SAVE_PATH}")
        print(f"Total rows saved: {len(df)}")
 
        # Quick summary
        pred = df['Predicted Category'].values
        true = df['True Category'].values
        TP = int(((pred==1)&(true==1)).sum())
        TN = int(((pred==0)&(true==0)).sum())
        FP = int(((pred==1)&(true==0)).sum())
        FN = int(((pred==0)&(true==1)).sum())
        print(f"\nQuick summary: TP={TP}, TN={TN}, FP={FP}, FN={FN}")
        print(f"Accuracy: {(TP+TN)/len(df)*100:.1f}%")
    else:
        print("No data collected. File not saved.")
 
    ser.close()
