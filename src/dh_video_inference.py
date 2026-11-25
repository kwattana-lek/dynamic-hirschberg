import os
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import tkinter as tk
from tkinter import filedialog
import os

# --- Configuration ---
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments", "real_video", "cnn_bilstm_seed0.keras")
SEQUENCE_LENGTH = 150
CLASSES = ["Healthy", "Intermittent", "Constant"]
# Your specific video folder
INITIAL_VIDEO_DIR = os.getcwd()  # folder shown first in the file dialog

# --- Load the Brain (Once at startup) ---
print(f"Loading model from {MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Initialize Tkinter once
root = tk.Tk()
root.withdraw() 

# --- Shared Utilities ---
mp_face_mesh = mp.solutions.face_mesh
LEFT_IRIS = [468, 469, 471]
RIGHT_IRIS = [473, 474, 476]

def find_glint(frame, iris_center_coords, threshold, search_radius=5):
    x_c, y_c = iris_center_coords
    h, w, _ = frame.shape
    x1, y1 = max(0, x_c - search_radius), max(0, y_c - search_radius)
    x2, y2 = min(w, x_c + search_radius), min(h, y_c + search_radius)
    
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0: return None
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred_roi = cv2.GaussianBlur(gray_roi, (3, 3), 0)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred_roi)
    if max_val < threshold: return None
    return (x1 + max_loc[0], y1 + max_loc[1])

# --- Continuous Processing Loop ---
while True:
    print("\n--- New Session ---")
    print("Opening file selection dialog...")
    
    if os.path.exists(INITIAL_VIDEO_DIR):
        start_dir = INITIAL_VIDEO_DIR
    else:
        start_dir = os.getcwd()

    # Ask for file
    video_path = filedialog.askopenfilename(
        initialdir=start_dir,
        title="Select Video (Cancel to Exit)",
        filetypes=(("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*"))
    )

    if not video_path:
        print("No file selected. Exiting program.")
        break

    print(f"Analyzing video: {os.path.basename(video_path)}")

    # --- Reset Buffer for New Video ---
    # CRITICAL: Must clear old data so it doesn't leak into new diagnosis
    data_buffer = deque(maxlen=SEQUENCE_LENGTH)
    for _ in range(SEQUENCE_LENGTH):
        data_buffer.append([0.0, 0.0]) 

    # --- Video Loop ---
    cap = cv2.VideoCapture(video_path)
    cv2.namedWindow('Dynamic Hirschberg 2.0 - AI DIAGNOSIS')
    # Trackbar (persistence handled by recreating window named same)
    cv2.createTrackbar('Glint Threshold', 'Dynamic Hirschberg 2.0 - AI DIAGNOSIS', 200, 255, lambda x: None)

    current_prediction = "Analyzing..."
    confidence_score = 0.0

    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success: 
                print("End of video file.")
                break

            h, w, _ = image.shape
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            thresh = cv2.getTrackbarPos('Glint Threshold', 'Dynamic Hirschberg 2.0 - AI DIAGNOSIS')
            
            # Helper text
            cv2.putText(image, "Press 'q' for next video, 'ESC' to quit", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            results = face_mesh.process(rgb_image)
            
            l_ratio_val = 0.0
            r_ratio_val = 0.0
            valid_frame = False

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mesh_points = np.array([np.multiply([p.x, p.y], [w, h]).astype(int) for p in face_landmarks.landmark])
                    
                    # Left Eye
                    l_c = tuple(mesh_points[LEFT_IRIS[0]])
                    l_dia = np.linalg.norm(mesh_points[LEFT_IRIS[1]] - mesh_points[LEFT_IRIS[2]])
                    l_glint = find_glint(image, l_c, thresh)
                    if l_glint and l_dia > 0:
                        cv2.line(image, l_c, l_glint, (0,255,0), 1)
                        l_ratio_val = np.linalg.norm(np.array(l_c) - np.array(l_glint)) / l_dia

                    # Right Eye
                    r_c = tuple(mesh_points[RIGHT_IRIS[0]])
                    r_dia = np.linalg.norm(mesh_points[RIGHT_IRIS[1]] - mesh_points[RIGHT_IRIS[2]])
                    r_glint = find_glint(image, r_c, thresh)
                    if r_glint and r_dia > 0:
                        cv2.line(image, r_c, r_glint, (0,255,0), 1)
                        r_ratio_val = np.linalg.norm(np.array(r_c) - np.array(r_glint)) / r_dia
                    
                    valid_frame = True

            # --- AI PIPELINE ---
            data_buffer.append([l_ratio_val, r_ratio_val])
            
            if len(data_buffer) == SEQUENCE_LENGTH and valid_frame:
                input_data = np.array(data_buffer).reshape(1, SEQUENCE_LENGTH, 2)
                preds = model.predict(input_data, verbose=0)
                class_idx = np.argmax(preds)
                confidence_score = preds[0][class_idx]
                current_prediction = CLASSES[class_idx]

            # --- VISUALIZATION ---
            color_map = {"Healthy": (0, 255, 0), "Intermittent": (0, 165, 255), "Constant": (0, 0, 255)}
            status_color = color_map.get(current_prediction, (255, 255, 255))
            
            cv2.rectangle(image, (0, h-100), (w, h), (0, 0, 0), -1)
            cv2.putText(image, f"AI Diagnosis: {current_prediction}", (20, h-40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            cv2.putText(image, f"Confidence: {confidence_score*100:.1f}%", (20, h-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow('Dynamic Hirschberg 2.0 - AI DIAGNOSIS', image)
            
            # --- KEYBOARD CONTROL ---
            key = cv2.waitKey(25) & 0xFF
            if key == ord('q'):
                print("Skipping to next video...")
                break # Break inner loop, goes back to 'while True'
            elif key == 27: # ESC key
                print("Exiting program.")
                cap.release()
                cv2.destroyAllWindows()
                exit() # Exit entire script

    cap.release()
    cv2.destroyAllWindows()

print("Program finished.")