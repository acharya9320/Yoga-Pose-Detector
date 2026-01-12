# ----------------------------------------------------
# TRAINING SECTION 
# ----------------------------------------------------
if train_model:
    st.subheader("Model Training Started ...")
    try:
        files = os.listdir("data")
        X, Y = [], []
        for file in files:
            if file.endswith(".npy"):
                pose_data = np.load(f"data/{file}")
                X.extend(pose_data)
                Y.extend([file.split(".")[0]] * len(pose_data))

        X = np.array(X)
        classes = sorted(list(set(Y)))

        # Manual label encoding
        label_to_index = {label: i for i, label in enumerate(classes)}
        Y = np.array([label_to_index[y] for y in Y])

        # One-hot encoding (manual)
        num_classes = len(classes)
        Y_onehot = np.zeros((len(Y), num_classes))
        for i, val in enumerate(Y):
            Y_onehot[i, val] = 1

        np.save("labels.npy", np.array(classes))
        st.write(f"Loaded {len(files)} poses with total samples: {len(X)}")

        # Manual train-test split (80-20)
        split_index = int(0.8 * len(X))
        x_train, x_test = X[:split_index], X[split_index:]
        y_train, y_test = Y_onehot[:split_index], Y_onehot[split_index:]

        # Simple Neural Network
        from keras.models import Sequential
        from keras.layers import Dense

        model = Sequential([
            Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        hist = model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test), verbose=1)

        model.save("model.h5")
        st.success("Model trained and saved successfully as model.h5")

        # Plot training accuracy graph
        st.line_chart({
            "Training Accuracy": hist.history['accuracy'],
            "Validation Accuracy": hist.history['val_accuracy']
        })

    except Exception as e:
        st.error(f"Error during training: {e}")
