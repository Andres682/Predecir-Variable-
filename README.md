# Datos proporcionados
X1 = [10, 15, 20, 30, 35, 40, 45, 50, 55]
X2 = [5, 8, 12, 15, 18, 20, 22, 25, 28]
X3 = [20, 25, 30, 35, 40, 45, 50, 55, 60]
Y = [35, 48, 62, 75, 88, 100, 112, 115, 138]
```
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from data import X1, X2, X3, Y
# Preparar los datos
X = np.column_stack((X1, X2, X3))
y = np.array(Y)
# Dividir en conjuntos de entrenamiento y prueba (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Escalar características para K-NN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
def evaluar_modelo(modelo, X_train, X_test, y_train, y_test, nombre):
    """Función para entrenar y evaluar un modelo"""
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"\n=== Modelo {nombre} ===")
    print(f"MSE: {mse:.2f}")
    print("Predicciones vs Valores reales:")
    for pred, real in zip(y_pred, y_test):
        print(f"Pred: {pred:.1f} | Real: {real} | Diferencia: {abs(pred-real):.1f}")
    return mse
# 1. Modelo K-NN
knn = KNeighborsRegressor(n_neighbors=3)
mse_knn = evaluar_modelo(knn, X_train_scaled, X_test_scaled, y_train, y_test, "K-NN")
# 2. Modelo Regresión Lineal Múltiple
lr = LinearRegression()
mse_lr = evaluar_modelo(lr, X_train, X_test, y_train, y_test, "Regresión Lineal")
# Mostrar coeficientes del modelo lineal
print("\nCoeficientes del modelo de regresión lineal:")
print(f"Intercepto: {lr.intercept_:.2f}")
print(f"Coeficiente X1: {lr.coef_[0]:.2f}")
print(f"Coeficiente X2: {lr.coef_[1]:.2f}")
print(f"Coeficiente X3: {lr.coef_[2]:.2f}")
# Comparación final
print("\n=== Comparación Final ===")
print(f"Diferencia en MSE: {abs(mse_knn - mse_lr):.2f}")
if mse_knn < mse_lr:
    print("El modelo K-NN tuvo mejor desempeño (menor MSE)")
else:
    print("El modelo de Regresión Lineal tuvo mejor desempeño (menor MSE)")
```
### 3. Archivo `requirements.txt`
```
numpy==1.21.0
scikit-learn==0.24.2
```
### 4. Archivo `README.md`
```markdown
# Predicción de Y usando X1, X2 y X3
Este proyecto implementa dos métodos de aprendizaje supervisado para predecir la variable Y en función de las variables X1, X2 y X3.
## Métodos Implementados
1. **K-Nearest Neighbors (K-NN)**
   - Algoritmo basado en instancias
   - Predice el valor promediando los k vecinos más cercanos
   - Requiere escalado de características
2. **Regresión Lineal Múltiple**
   - Modelo lineal que encuentra la mejor combinación de variables independientes
   - Proporciona coeficientes interpretables
## Interpretación
- **K-NN**: Bueno para relaciones no lineales pero sensible a la escala de datos
- **Regresión Lineal**: Óptimo para relaciones lineales, proporciona ecuación interpretable

