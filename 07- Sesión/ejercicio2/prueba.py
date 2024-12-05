import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

# Leer el CSV
df = pd.read_csv('archivo.csv', sep=',', skiprows=1)
df.columns = ['Semana', 'software']
df['Semana'] = pd.to_datetime(df['Semana'])

# Estadísticas descriptivas
print("Estadísticas descriptivas:")
print(df['software'].describe())

# Graficar serie temporal
plt.figure(figsize=(15, 6))
plt.plot(df['Semana'], df['software'], marker='o')
plt.title('Evolución del software con el tiempo')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Análisis de tendencia
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df['software'], period=52)  # Periodo anual
result.plot()
plt.show()