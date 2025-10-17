import cv2
from ultralytics import YOLO
import time
import os

# --- IMPORTANT: VERIFY THIS PATH ---
# Use the path from your latest GPU training run
MODEL_PATH = 'Best.pt'

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    exit()

# Load your custom-trained YOLOv8 model
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Attempt to open the webcam
cap = cv2.VideoCapture(0) 
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened successfully. Press 'q' to quit.")

while True:
    start_time = time.time()
    success, frame = cap.read()

    if not success:
        print("Error: Failed to capture frame.")
        break

    # Run YOLOv8 inference on the frame
    # We lower the confidence threshold to 0.4 to see more detections
    results = model(frame, conf=0.4)

    # Get the annotated frame with bounding boxes
    annotated_frame = results[0].plot()

    # --- Pick Point Generation ---
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            pick_point_x = (x1 + x2) // 2
            pick_point_y = (y1 + y2) // 2
            
            # Draw the pick point
            cv2.drawMarker(annotated_frame, (pick_point_x, pick_point_y), 
                           color=(0, 0, 255), markerType=cv2.MARKER_CROSS, 
                           markerSize=20, thickness=2)

    # --- Latency Measurement (Bonus!) ---
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    fps = 1 / (end_time - start_time)
    
    # Add latency and FPS to the frame
    cv2.putText(annotated_frame, f"Latency: {latency_ms:.2f} ms", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Real-Time Scrap Classifier", annotated_frame)

    # This is crucial for the window to update and to catch key presses
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Simulation stopped.")