import mediapipe as mp 
import numpy as np 
import cv2 

def inFrame(lst):
    if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility > 0.6 and lst[16].visibility > 0.6:
        return True 
    return False
 
cap = cv2.VideoCapture(0)

name = input("🟠🟡🟢 Enter the name of the Asana : ")

holistic = mp.solutions.pose
holis = holistic.Pose()
drawing = mp.solutions.drawing_utils

X = []
data_size = 0
collecting = False  # नया variable — यह बताएगा कि collection चालू हुआ या नहीं

while True:
    lst = []
    ret, frm = cap.read()
    if not ret:
        break

    frm = cv2.flip(frm, 1)
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    # जब collecting False हो तो केवल "Press SPACE to Start" दिखाएँ
    if not collecting:
        cv2.putText(frm, "Press SPACE to start collecting pose", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    else:
        # Collection चालू है
        if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
            for i in res.pose_landmarks.landmark:
                lst.append(i.x - res.pose_landmarks.landmark[0].x)
                lst.append(i.y - res.pose_landmarks.landmark[0].y)

            X.append(lst)
            data_size += 1
        else:
            cv2.putText(frm, "Make sure full body visible", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS)
    cv2.putText(frm, f"Frames: {data_size}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Pose Collector", frm)

    key = cv2.waitKey(1)
    if key == 27:  # ESC दबाकर बंद करें
        break
    elif key == 32:  # SPACE दबाकर शुरू करें
        collecting = True
        print("Pose collection started!")

    if collecting and data_size >= 80:
        print("Pose collection complete!")
        break

cv2.destroyAllWindows()
cap.release()

np.save(f"{name}.npy", np.array(X))
print(f"✅ Saved {name}.npy with shape: {np.array(X).shape}")
