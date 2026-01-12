import os
import numpy as np
import tensorflow as tf

# Step 1: User Input
pose_name = input(" 🟡--- Enter the Name of Asana/Pose to delete: ").strip()

# Step 2: Delete the .npy file
pose_file = f"{pose_name}.npy"
if os.path.exists(pose_file):
    os.remove(pose_file)
    print(f"Deleted dataset file: {pose_file}")
else:
    print(f"File {pose_file} not found. Maybe it's already deleted.")

# Step 3: Remove pose label from labels.npy (if exists)
if os.path.exists("labels.npy"):
    labels = np.load("labels.npy")
    labels = [lbl for lbl in labels if lbl != pose_name]
    np.save("labels.npy", np.array(labels))
    print("Updated labels.npy (pose removed).")
else:
    print("labels.npy not found.")

# Step 4: Retrain model automatically
train_choice = input("Do you want to retrain model now? (y/n): ").strip().lower()
if train_choice == 'y':
    print("\nRetraining model (this will replace old model.h5)...")
    import subprocess
    subprocess.run(["python", "data_training.py"])
    print("Model retrained successfully (model.h5 updated).")
else:
    print("Model not retrained yet. Run 'python data_training.py' manually later.")

print("\nPose deletion process completed successfully.")
