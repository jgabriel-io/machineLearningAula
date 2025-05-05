from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix
import numpy as np

# --- 1. Regressão Linear: Preço da Pizza ---
diametros = np.array([15, 20, 25, 30, 35]).reshape(-1, 1)
precos = np.array([20, 30, 40, 50, 60])
X_train, X_test, y_train, y_test = train_test_split(diametros, precos, test_size=0.2, random_state=42)
modelo_reg = LinearRegression()
modelo_reg.fit(X_train, y_train)
pizza_28 = modelo_reg.predict([[28]])
erro_medio = mean_absolute_error(y_test, modelo_reg.predict(X_test))
print("🔵 Regressão Linear")
print(f"Preço previsto para pizza de 28cm: R$ {pizza_28[0]:.2f}")
print(f"Erro médio absoluto: R$ {erro_medio:.2f}")

# --- 2. Classificação: Frutas por Peso ---
pesos = np.array([100, 120, 150, 170]).reshape(-1, 1)
classes = ["maçã", "maçã", "laranja", "laranja"]
X_train, X_test, y_train, y_test = train_test_split(pesos, classes, test_size=0.25, random_state=42)
modelo_fruit = DecisionTreeClassifier()
modelo_fruit.fit(X_train, y_train)
fruit_140 = modelo_fruit.predict([[140]])
acuracia = accuracy_score(y_test, modelo_fruit.predict(X_test))
print("\n🍊 Classificação de Frutas")
print(f"Classe prevista para 140g: {fruit_140[0]}")
print(f"Acurácia do modelo: {acuracia:.2f}")

# --- 3. Árvore de Decisão: Jogar Tênis ---
clima = np.array([0, 1, 1, 0]).reshape(-1, 1)
joga = [0, 1, 1, 0]
X_train, X_test, y_train, y_test = train_test_split(clima, joga, test_size=0.5, random_state=42)
modelo_tenis = DecisionTreeClassifier()
modelo_tenis.fit(X_train, y_train)
tenis_sol = modelo_tenis.predict([[1]])
matriz = confusion_matrix(y_test, modelo_tenis.predict(X_test))
print("\n🎾 Decisão de Jogar Tênis")
print(f"Vai jogar com clima 'sol'? {'Sim' if tenis_sol[0] == 1 else 'Não'}")
print("Matriz de Confusão:")
print(matriz)
