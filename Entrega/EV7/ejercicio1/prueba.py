import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

# Leer el CSV usando la ruta completa
df = pd.read_csv(r"C:\Users\Usuario\OneDrive - SENA\Documentos\complementario-3125033\07- Sesión\ejercicio1\archivo.csv", sep=',', skiprows=1)

# Asignar nombres a las columnas
df.columns = ['Semana', 'software']

# Convertir la columna 'Semana' a formato de fecha
df['Semana'] = pd.to_datetime(df['Semana'])

# Estadísticas descriptivas
print("Estadísticas descriptivas:")
print(df['software'].describe())

# Graficar serie temporal
plt.figure(figsize=(15, 6))
plt.plot(df['Semana'], df['software'], marker='o', color='red')  # Cambiar el color a rojo
plt.title('Transformación del Software a lo largo del tiempo')
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