import ssl
from keras.datasets import fashion_mnist
from keras import layers, models, callbacks
from sklearn.model_selection import train_test_split


ssl._create_default_https_context = ssl._create_unverified_context


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

checkpoint_cb = callbacks.ModelCheckpoint("best_model.keras", save_best_only=True)

tensorboard_cb = callbacks.TensorBoard(log_dir="./logs")

early_stopping_cb = callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5,
                                            restore_best_weights=True)

history = model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_valid, y_valid),
                    callbacks=[checkpoint_cb, tensorboard_cb, early_stopping_cb])

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

model.save("final_model.keras")
