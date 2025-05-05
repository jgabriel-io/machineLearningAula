# Relatório de Implementação – Algoritmos de Machine Learning

## Objetivo

Este relatório apresenta a implementação de três algoritmos fundamentais de Machine Learning com a biblioteca **scikit-learn**, aplicados a problemas simples e práticos. O objetivo é treinar modelos preditivos utilizando dados supervisionados e avaliar seus desempenhos com métricas apropriadas.

---

## 1. Regressão Linear – Preço de Pizzas

### Descrição do Problema

A tarefa consiste em prever o preço de uma pizza com base no seu diâmetro (em cm). Para isso, foi utilizado um modelo de regressão linear simples.

### Dados Utilizados

| Diâmetro (cm) | Preço (R$) |
|---------------|------------|
| 15            | 20         |
| 20            | 30         |
| 25            | 40         |
| 30            | 50         |
| 35            | 60         |

### Processo

- Os dados foram divididos em **80% para treino** e **20% para teste**, usando `train_test_split`.
- Foi aplicado o método `.fit()` com `LinearRegression`.
- Realizou-se a previsão do preço para uma pizza de **28 cm**.
- O desempenho foi avaliado usando **Erro Médio Absoluto (MAE)**.

### Resultado

- **Preço previsto para 28 cm:** R$ 46,00  
- **Erro médio absoluto:** R$ 0,00

---

## 2. Classificação – Frutas por Peso

### Descrição do Problema

O objetivo foi classificar frutas como "maçã" ou "laranja" com base no peso, utilizando um classificador de árvore de decisão.

### Dados Utilizados

| Peso (g) | Classe   |
|----------|----------|
| 100      | maçã     |
| 120      | maçã     |
| 150      | laranja  |
| 170      | laranja  |

### Processo

- Divisão dos dados em **75% treino** e **25% teste**.
- Treinamento com `DecisionTreeClassifier` e método `.fit()`.
- Previsão feita para uma fruta com **140g**.
- Acurácia calculada com `accuracy_score`.

### Resultado

- **Classe prevista para 140g:** laranja  
- **Acurácia:** 100%

---

## 3. Árvore de Decisão – Jogar Tênis?

### Descrição do Problema

Foi implementada uma árvore de decisão para prever se uma pessoa vai jogar tênis com base nas condições climáticas (sol ou chuva).

### Dados Utilizados

| Clima | Jogar |
|-------|--------|
| chuva | não    |
| sol   | sim    |
| sol   | sim    |
| chuva | não    |

(Clima: 0 = chuva, 1 = sol / Jogar: 0 = não, 1 = sim)

### Processo

- Divisão dos dados em **50% treino** e **50% teste**.
- Treinamento com `DecisionTreeClassifier` e método `.fit()`.
- Previsão para o clima **"sol"**.
- Avaliação com **matriz de confusão**.

### Resultado

- **Vai jogar com clima "sol"?** Sim  
- **Matriz de Confusão:**

