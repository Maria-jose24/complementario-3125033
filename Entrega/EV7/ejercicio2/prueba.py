import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

# Cargar los datos desde un archivo CSV
df = pd.read_csv(r"C:\Users\Usuario\OneDrive - SENA\Documentos\complementario-3125033\07- Sesión\ejercicio2\archivo.csv")

# Asegúrate de que no hay valores NaN en las columnas de interés
df_clean = df.dropna(subset=['Python', 'Java', 'C++'])

# Variables
X = df_clean[['Python']].values
y = df_clean['Java'].values

# Modelo de regresión lineal
modelo = LinearRegression().fit(X, y)
y_pred = modelo.predict(X)

# Calcular coeficiente de correlación
correlacion, p_value = stats.pearsonr(X.flatten(), y)

# Hipótesis: si el valor p es menor que 0.05, rechazamos la hipótesis nula
# y concluimos que hay una relación significativa
if p_value < 0.05:
    print(f"Existe una relación significativa entre Python y Java (p-value = {p_value:.4f}).")
    
    # Graficar los puntos y la línea de regresión
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', s=100, label='Datos')
    plt.plot(X, y_pred, color='red', label='Línea de regresión')
    plt.title('Análisis de Correlación entre Python y Java', fontsize=16)
    plt.xlabel('Python', fontsize=14)
    plt.ylabel('Java', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
else:
    print(f"No existe una relación significativa entre Python y Java (p-value = {p_value:.4f}).")
