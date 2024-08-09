import pandas as pd
import numpy as np
import tensorflow as tf
import random
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout
from tensorflow.keras.models import Model

# Set random seeds for reproducibility
def set_random_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

set_random_seeds()

# Funções auxiliares para o Transformer
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({"position": self.position, "d_model": self.d_model})
        return config

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs
    x = LayerNormalization(epsilon=1e-6)(res)
    x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

# Passo 1: Leitura do CSV
file_path = './data/dengue_input_simple.csv'   # Substitua pelo caminho do seu arquivo
data = pd.read_csv(file_path, delimiter=';')

# Passo 2: Seleção e normalização das features
features = ['dengue_diagnosis', 'precipitacao (mm)', 'temperatura (°C)', 'umidade ar (%)', 
            't-1', 't-2', 't-3', 'precipitacao (mm)-1', 'temperatura (°C)-1', 'umidade ar (%)-1']
target = 'dengue_diagnosis'

# Filtrando as colunas de interesse
data = data[features]

# Normalizando os dados
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Passo 3: Preparação dos dados para o Transformer
look_back = 12  # um look_back apropriado
X, Y = [], []
for i in range(len(scaled_data) - look_back):
    X.append(scaled_data[i:i + look_back])
    Y.append(scaled_data[i + look_back, 0])  # coluna target é a primeira

X, Y = np.array(X), np.array(Y)

# Passo 4: Construção do modelo Transformer
input_shape = X.shape[1:]
inputs = Input(shape=input_shape)
x = PositionalEncoding(input_shape[0], input_shape[1])(inputs)
x = transformer_encoder(x, head_size=256, num_heads=4, ff_dim=4)
x = transformer_encoder(x, head_size=256, num_heads=4, ff_dim=4)
x = transformer_encoder(x, head_size=256, num_heads=4, ff_dim=4)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
outputs = Dense(1, activation='relu')(x)  # ReLU activation to ensure non-negative output
model = Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mean_squared_error')

# Passo 5: Treinamento do modelo
history = model.fit(X, Y, epochs=100, batch_size=32, validation_split=0.1, verbose=2)

# Passo 6: Fazendo previsões
# Print scaled data
print("Scaled Data:", scaled_data)

# Prepare last data for prediction
last_data = scaled_data[-look_back:]
print("Last Data Before Expanding Dimensions:", last_data)

last_data = np.expand_dims(last_data, axis=0)
print("Last Data After Expanding Dimensions:", last_data)

# Make prediction
prediction = model.predict(last_data)
print("Raw Model Prediction:", prediction)

# Ensure correct inverse transformation
prediction = scaler.inverse_transform(
    np.concatenate([prediction, np.zeros((prediction.shape[0], scaled_data.shape[1] - 1))], axis=1)
)[:, 0]
print("Inverse Transformed Prediction:", prediction)

print("Previsão do nível de infestação de dengue para o próximo mes:", prediction[0])
