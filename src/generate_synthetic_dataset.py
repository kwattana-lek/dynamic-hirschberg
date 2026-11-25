import numpy as np
import pandas as pd
import os
import shutil
import random

# --- Configuration ---
DATASET_SIZE = 200  # Increased size for better robustness
DURATION_SEC = 5
FPS = 30
TOTAL_FRAMES = DURATION_SEC * FPS
OUTPUT_DIR = "synthetic_dataset_robust"

if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

print(f"Generating {DATASET_SIZE * 3} ROBUST synthetic samples...")

def add_noise(signal, level=0.005):
    """Adds high-frequency sensor noise."""
    noise = np.random.normal(0, level, signal.shape)
    return signal + noise

def add_head_drift(signal):
    """Adds low-frequency sine wave to simulate head moving closer/further."""
    t = np.linspace(0, 2*np.pi, len(signal))
    drift_amplitude = np.random.uniform(0.001, 0.005) 
    drift = np.sin(t + np.random.uniform(0, 2*np.pi)) * drift_amplitude
    return signal + drift

def add_blinks(signal):
    """Randomly sets segments to 0.0 to simulate blinks."""
    num_blinks = np.random.randint(0, 3) # 0 to 2 blinks per 5 sec
    signal_with_blinks = signal.copy()
    
    for _ in range(num_blinks):
        # Blink duration: 3 to 10 frames (100ms - 300ms)
        duration = np.random.randint(3, 10)
        start = np.random.randint(0, len(signal) - duration)
        signal_with_blinks[start : start+duration] = 0.0
        
    return signal_with_blinks

def generate_signal(condition):
    signal = np.zeros(TOTAL_FRAMES)
    kappa = np.random.uniform(-0.02, 0.02) # Angle Kappa
    signal += kappa
    
    if condition == "Healthy":
        pass # Just kappa + noise
        
    elif condition == "Constant":
        deviation = np.random.choice([1, -1]) * np.random.uniform(0.12, 0.25)
        signal += deviation
        
    elif condition == "Intermittent":
        start_drift = np.random.randint(30, 100)
        drift_duration = np.random.randint(30, 80)
        max_deviation = np.random.uniform(0.10, 0.20)
        
        for t in range(TOTAL_FRAMES):
            if t >= start_drift and t < start_drift + drift_duration:
                progress = (t - start_drift) / drift_duration
                signal[t] += max_deviation * progress
            elif t >= start_drift + drift_duration:
                signal[t] += max_deviation

    # --- Apply Real-World Artifacts ---
    signal = add_head_drift(signal)
    signal = add_noise(signal)
    signal = add_blinks(signal) # The most important new step
    
    return signal

# --- Generation Loop ---
classes = ["Healthy", "Intermittent", "Constant"]

for class_name in classes:
    class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    for i in range(DATASET_SIZE):
        L_ratio = generate_signal(class_name)
        # Right eye is healthy but also blinks/drifts
        R_ratio = generate_signal("Healthy") 
        
        # In 'Constant' case, sometimes both eyes are bad? 
        # For now, let's keep it simple: Left is the problem eye.
        
        df = pd.DataFrame({'L_ratio': L_ratio, 'R_ratio': R_ratio})
        
        filename = f"{class_name}_{i:04d}.csv"
        df.to_csv(os.path.join(class_dir, filename), index=False)

print(f"Done! New robust data saved to '{OUTPUT_DIR}'.")
print("IMPORTANT: Update your train_model.py to point to this new folder!")