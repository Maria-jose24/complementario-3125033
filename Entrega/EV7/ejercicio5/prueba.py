# Importar las librerías necesarias
import matplotlib.pyplot as plt  # Para graficar datos y líneas de regresión
import pandas as pd              # Para manipular datos tabulares (DataFrame)
import numpy as np               # Para operaciones matemáticas y manipulación de arrays
from sklearn.linear_model import LinearRegression  # Para realizar regresión lineal
from scipy import stats          # Para calcular correlaciones y valores p

# Cargar los datos desde un archivo CSV usando la ruta completa
df = pd.read_csv(r"C:\Users\Usuario\OneDrive - SENA\Documentos\complementario-3125033\07- Sesión\ejercicio5\archivo.csv")

# Limpiar los datos asegurándose de que no haya valores NaN en la columna 'Arquitectura de software'
df_clean = df.dropna(subset=['Arquitectura de software'])

# Definir las variables para el análisis
# X: Un array de índices secuenciales (una variable auxiliar para representar semanas)
# reshape(-1, 1) convierte el array en formato 2D necesario para el modelo
X = np.arange(len(df_clean)).reshape(-1, 1)
# y: Los valores de la columna 'Arquitectura de software', que es nuestra variable dependiente
y = df_clean['Arquitectura de software'].values

# Crear y entrenar el modelo de regresión lineal
modelo = LinearRegression().fit(X, y)
# Generar predicciones del modelo para los valores en X
y_pred = modelo.predict(X)

# Calcular el coeficiente de correlación de Pearson entre las variables X y y
# X.flatten() convierte X en una dimensión para calcular correctamente la correlación
correlacion, p_value = stats.pearsonr(X.flatten(), y)

# Evaluar si existe una relación significativa entre las variables
# Hipótesis: si p_value < 0.05, hay una relación significativa
if p_value < 0.05:
    # Imprimir el resultado indicando que hay una relación significativa
    print(f"Existe una relación significativa entre Semana y Arquitectura de software (p-value = {p_value:.4f}).")
    
    # Graficar los datos y la línea de regresión
    plt.figure(figsize=(14, 8))  # Crear una figura más grande para mejor visualización
    plt.scatter(df_clean['Semana'], y, color='blue', s=100, label='Datos')  # Puntos de datos originales
    plt.plot(df_clean['Semana'], y_pred, color='red', label='Línea de regresión')  # Línea de regresión predicha
    # Configurar el título y las etiquetas de los ejes
    plt.title('Evolución de las Preferencias en Arquitectura de Software', fontsize=16)
    plt.xlabel('Semana', fontsize=14)
    plt.ylabel('Arquitectura de Software', fontsize=14)
    plt.legend(loc='best')  # Agregar leyenda en la mejor posición automática
    plt.grid(True, linestyle='--', alpha=0.5)  # Mostrar cuadrícula con líneas punteadas

    # Ajustar las etiquetas del eje X para que muestren cada 8 semanas con rotación para legibilidad
    plt.xticks(ticks=np.arange(0, len(df_clean), step=8), labels=df_clean['Semana'][::8], rotation=45, fontsize=10)
    
    # Ajustar el espaciado del gráfico para que todo se visualice correctamente
    plt.tight_layout()
    plt.show()  # Mostrar el gráfico generado
else:
    # Imprimir un mensaje indicando que no existe una relación significativa
    print(f"No existe una relación significativa entre Semana y Arquitectura de software (p-value = {p_value:.4f}).")
