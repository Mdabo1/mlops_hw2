import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from mlflow.models.signature import infer_signature
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Загружаем датасет "Wine Quality"
data = fetch_openml(name="wine-quality-red", version=1, as_frame=True)
X = data.data
y = data.target.astype(float)  # Преобразуем целевые данные в числовой формат

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Настраиваем MLflow
mlflow.set_experiment("Wine_Quality_Regression")

# Параметры модели (изменяем для экспериментов)
n_estimators = 200
max_depth = 10

with mlflow.start_run():
    # Логируем параметры
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Обучаем модель
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Оцениваем модель
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    # Логируем метрики
    mlflow.log_metric("rmse", rmse)

    # Создаём пример входных данных и подпись
    input_example = X_test.iloc[:5]
    signature = infer_signature(X_train, model.predict(X_train))

    # Логируем модель с примером и подписью
    mlflow.sklearn.log_model(model, "random_forest_regressor", input_example=input_example, signature=signature)

    # Сохраняем метрики в файл
    metrics = [{"n_estimators": n_estimators, "max_depth": max_depth, "rmse": rmse}]
    df = pd.DataFrame(metrics)
    df.to_csv("metrics.csv", index=False)

    # Визуализация
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.title("Actual vs Predicted Wine Quality")
    plt.xlabel("Actual Quality")
    plt.ylabel("Predicted Quality")
    plt.savefig("prediction_plot_1.png")
