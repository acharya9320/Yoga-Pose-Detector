import os  
import numpy as np 
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense 
from keras.models import Model

# -------- Step 1: Load all .npy pose files automatically --------
X, y, label = [], [], []

for file in os.listdir():
    if file.endswith(".npy") and file != "labels.npy":
        print(f"📂 Loading {file} ...")
        data = np.load(file)
        X.extend(data)
        y.extend([file.replace(".npy", "")] * len(data))
        if file.replace(".npy", "") not in label:
            label.append(file.replace(".npy", ""))

X = np.array(X)
y = np.array(y)

print(" Data Loaded Successfully!")
print(f"Total Samples: {X.shape[0]}, Features per Sample: {X.shape[1]}")
print(f"Total Classes: {len(label)} → {label}")

# -------- Step 2: Encode class labels --------
label_dict = {name: idx for idx, name in enumerate(label)}
y_encoded = np.array([label_dict[i] for i in y])
y_categorical = to_categorical(y_encoded)

# -------- Step 3: Shuffle data --------
shuffle_idx = np.arange(len(X))
np.random.shuffle(shuffle_idx)
X = X[shuffle_idx]
y_categorical = y_categorical[shuffle_idx]

# -------- Step 4: Build model --------
input_shape = (X.shape[1],)
ip = Input(shape=input_shape)

m = Dense(128, activation="tanh")(ip)
m = Dense(64, activation="tanh")(m)
op = Dense(len(label), activation="softmax")(m)

model = Model(inputs=ip, outputs=op)
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

# -------- Step 5: Train the model --------
print("\n  Training model ...\n")
model.fit(X, y_categorical, epochs=80, batch_size=16, verbose=1)

# -------- Step 6: Save model and labels --------
model.save("model.h5")
np.save("labels.npy", np.array(label))
print("\n Model retrained and saved successfully!")
print(" Labels updated and saved in labels.npy")
