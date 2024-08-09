import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler

# Passo 1: Leitura do CSV
file_path = './data/dengue_input_simple.csv'  # Substitua pelo caminho do seu arquivo
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

# Passo 3: Preparação dos dados para o SARIMA
# Para o SARIMA, usaremos apenas a coluna target 'dengue_diagnosis'
dengue_diagnosis = scaled_data[:, 0]

# Passo 4: Ajuste do modelo SARIMA
# Escolha dos parâmetros p, d, q, P, D, Q, m (esses parâmetros podem ser ajustados conforme necessário)
p, d, q = 1, 1, 1  # Parâmetros ARIMA
P, D, Q, m = 1, 1, 1, 12  # Parâmetros sazonais (m = 12 para sazonalidade anual)

# Ajuste do modelo SARIMA
model = SARIMAX(dengue_diagnosis, order=(p, d, q), seasonal_order=(P, D, Q, m))
model_fit = model.fit(disp=False)

# Passo 5: Fazendo previsões
# Previsão para o próximo período
forecast = model_fit.forecast(steps=1)
forecast = scaler.inverse_transform(np.concatenate([forecast.reshape(-1, 1), np.zeros((forecast.shape[0], scaled_data.shape[1] - 1))], axis=1))[:, 0]

print("Previsão do nível de infestação de dengue para o próximo mes:", forecast[0])
