import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

# Cargar los datos desde un archivo CSV
df = pd.read_csv(r"C:\Users\Usuario\OneDrive - SENA\Documentos\complementario-3125033\07- Sesión\ejercicio7\archivo.csv", sep='|')

# Asegúrate de que no hay valores NaN en las columnas de interés
df_clean = df.dropna(subset=['Race Effectiveness', 'Exercise1 Effectiveness'])

# Variables
X = df_clean[['Race Effectiveness']].values
y = df_clean['Exercise1 Effectiveness'].values

# Modelo de regresión lineal
modelo = LinearRegression().fit(X, y)
y_pred = modelo.predict(X)

# Calcular coeficiente de correlación
correlacion, p_value = stats.pearsonr(X.flatten(), y)

# Hipótesis: si el valor p es menor que 0.05, rechazamos la hipótesis nula
# y concluimos que hay una relación significativa
if p_value < 0.05:
    print(f"Existe una relación significativa entre Race Effectiveness y Exercise1 Effectiveness (p-value = {p_value:.4f}).")
    
    # Graficar los puntos y la línea de regresión
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', s=100, label='Datos')
    plt.plot(X, y_pred, color='red', label='Línea de regresión')
    plt.title('Análisis y Efectividad entre Race Effectiveness y Exercise 1 Effectiveness.', fontsize=16)
    plt.xlabel('Race Effectiveness', fontsize=14)
    plt.ylabel('Exercise1 Effectiveness', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

else:
    print(f"No existe una relación significativa entre Race Effectiveness y Exercise1 Effectiveness (p-value = {p_value:.4f}).")