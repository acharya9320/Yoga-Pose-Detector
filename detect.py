import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import time

# ---------- Helper Functions ----------
def inFrame(lst):
    if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility > 0.6 and lst[16].visibility > 0.6:
        return True
    return False

def resize_aspect_auto(frame, target_size):
    """Resize frame keeping aspect ratio."""
    h, w = frame.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    # Center the resized frame
    result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    result[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return result

# ---------- Load Model ----------
model = load_model("model.h5")
label = np.load("labels.npy")

# ---------- Mediapipe Setup ----------
holistic = mp.solutions.pose
holis = holistic.Pose()
drawing = mp.solutions.drawing_utils

# ---------- Camera Setup ----------
cap = cv2.VideoCapture(0)
cv2.namedWindow("Yoga Pose Detector", cv2.WINDOW_NORMAL)
cv2.moveWindow("Yoga Pose Detector",100,100)

flip_h = False
flip_v = False
aspect_mode = "9:16"
prev_time = 0

print("♻ Controls:")
print("  h → flip horizontally")
print("  v → flip vertically")
print("  1 → aspect 16:9")
print("  2 → aspect 9:16")
print("  3 → aspect 4:3")
print("  q → quit")

# ---------- Main Loop ----------
while True:
    ret, frm = cap.read()
    if not ret:
        break

    # Flip options
    if flip_h:
        frm = cv2.flip(frm, 1)
    if flip_v:
        frm = cv2.flip(frm, 0)

    # Prepare main responsive window
    window_h, window_w = 900, 900
    window = np.zeros((window_h, window_w, 3), dtype="uint8")

    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
    frm = cv2.blur(frm, (4, 4))

    lst = []
    if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
        for i in res.pose_landmarks.landmark:
            lst.append(i.x - res.pose_landmarks.landmark[0].x)
            lst.append(i.y - res.pose_landmarks.landmark[0].y)

        lst = np.array(lst).reshape(1, -1)
        p = model.predict(lst)
        pred = label[np.argmax(p)]

        if p[0][np.argmax(p)] > 0.75:
            cv2.putText(window, pred, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        else:
            cv2.putText(window, "Asana is wrong or not trained", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(frm, "Make sure full body visible", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Draw pose landmarks
    drawing.draw_landmarks(
        frm, res.pose_landmarks, holistic.POSE_CONNECTIONS,
        connection_drawing_spec=drawing.DrawingSpec(color=(255, 255, 255), thickness=3),
        landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), circle_radius=2, thickness=2)
    )

    # Responsively resize the frame inside window
    frame_resized = resize_aspect_auto(frm, (window_w - 100, window_h - 150))
    window[100:100 + frame_resized.shape[0], 50:50 + frame_resized.shape[1]] = frame_resized

    # FPS counter
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    cv2.putText(window, f"FPS: {int(fps)}", (20, window_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    # Show window
    cv2.imshow("Yoga Pose Detector", window)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('h'):
        flip_h = not flip_h
    elif key == ord('v'):
        flip_v = not flip_v
    elif key == ord('1'):
        aspect_mode = "16:9"
    elif key == ord('2'):
        aspect_mode = "9:16"
    elif key == ord('3'):
        aspect_mode = "4:3"

cap.release()
cv2.destroyAllWindows()
