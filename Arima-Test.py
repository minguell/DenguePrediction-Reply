import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler

# Passo 1: Leitura do CSV
file_path = 'path_to_your_file.csv'  # Substitua pelo caminho do seu arquivo
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

# Passo 3: Preparação dos dados para o ARIMA
# Para o ARIMA, usaremos apenas a coluna target 'dengue_diagnosis'
dengue_diagnosis = scaled_data[:, 0]

# Passo 4: Ajuste do modelo ARIMA
# Escolha dos parâmetros p, d, q (esses parâmetros podem ser ajustados conforme necessário)
p, d, q = 5, 1, 0  # Você pode ajustar esses valores conforme necessário

# Ajuste do modelo ARIMA
model = ARIMA(dengue_diagnosis, order=(p, d, q))
model_fit = model.fit()

# Passo 5: Fazendo previsões
# Previsão para o próximo período
forecast = model_fit.forecast(steps=1)
forecast = scaler.inverse_transform(np.concatenate([forecast.reshape(-1, 1), np.zeros((forecast.shape[0], scaled_data.shape[1] - 1))], axis=1))[:, 0]

print("Previsão do nível de infestação de dengue para o próximo dia:", forecast[0])
