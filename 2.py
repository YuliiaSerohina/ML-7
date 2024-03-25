import pandas as pd
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras import Sequential
from keras.layers import Dense, Dropout


dataset = pd.read_csv('kc_house_data.csv')

y = np.asarray(dataset['price'].values.tolist())
y = y.reshape(len(y), 1)
X = np.array(dataset.drop(['price', 'date'], axis=1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(1)
])

model.compile(
    optimizer='adam', loss=lambda y_true,
    y_pred: 10 * keras.losses.mean_squared_error(y_true, y_pred)
)


history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on test set:", mse)

model.save("regression_model.keras")


