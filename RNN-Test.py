import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Set random seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

# Load the data
file_path = './data/dengue_input_simple.csv'  # Update with your file path
data = pd.read_csv(file_path, delimiter=';')

# Filter data from 2017 onwards
data = data[data['ano'] >= 2017]

# Select and normalize features
features = ['dengue_diagnosis', 'precipitacao (mm)', 'temperatura (°C)', 'umidade ar (%)', 
            't-1', 't-2', 't-3', 'precipitacao (mm)-1', 'temperatura (°C)-1', 'umidade ar (%)-1']
target = 'dengue_diagnosis'
data = data[features]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Prepare data for RNN
look_back = 3  # Increase look-back period for more context
X, Y = [], []
for i in range(len(scaled_data) - look_back):
    X.append(scaled_data[i:i + look_back])
    Y.append(scaled_data[i + look_back, 0])  # target column is the first
X, Y = np.array(X), np.array(Y)

# Reshape data for [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))

# Build and train the RNN model
model = Sequential()
model.add(SimpleRNN(50, input_shape=(look_back, len(features))))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Implement early stopping
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

model.fit(X, Y, epochs=100, batch_size=1, verbose=2, callbacks=[early_stopping])

# Make predictions for all months from 2017 onwards
predictions = []
for i in range(look_back, len(scaled_data)):
    last_data = scaled_data[i-look_back:i]
    last_data = np.reshape(last_data, (1, look_back, len(features)))
    prediction = model.predict(last_data)
    predictions.append(prediction[0, 0])

# Inverse transform the predictions
predictions = np.array(predictions).reshape(-1, 1)
predictions = scaler.inverse_transform(np.concatenate([predictions, np.zeros((predictions.shape[0], scaled_data.shape[1] - 1))], axis=1))[:, 0]

# Clip the predictions to ensure non-negative values
predictions = np.clip(predictions, 0, None)

# Filter out predictions that are zero
filtered_predictions = [pred for pred in predictions if pred > 0]

# Print predictions
print("Previsões do nível de infestação de dengue desde 2017:")
for i, pred in enumerate(filtered_predictions):
    print(f"Mês {i + 1}: {pred}")