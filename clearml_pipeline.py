from clearml import Task
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Настраиваем ClearML Task
task = Task.init(project_name="Wine Quality Regression", task_name="Automated Pipeline")

# Загружаем датасет
data = fetch_openml(name="wine-quality-red", version=1, as_frame=True)
X = data.data
y = data.target.astype(float)

# Разделяем данные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Параметры для экспериментов
experiments = [
    {"n_estimators": 200, "max_depth": 10},
    {"n_estimators": 300, "max_depth": 20}
]

results = []

for exp in experiments:
    # Настраиваем подзадачу
    sub_task = Task.create(project_name="Wine Quality Regression", task_name=f"Experiment n_estimators={exp['n_estimators']} max_depth={exp['max_depth']}")
    sub_task.connect(exp)  # Логируем параметры
    
    # Создаем и обучаем модель
    model = RandomForestRegressor(n_estimators=exp["n_estimators"], max_depth=exp["max_depth"], random_state=42)
    model.fit(X_train, y_train)
    
    # Предсказания и метрики
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    results.append({"n_estimators": exp["n_estimators"], "max_depth": exp["max_depth"], "rmse": rmse})

    # Логируем метрики
    sub_task.get_logger().report_scalar("Metrics", "RMSE", iteration=0, value=rmse)
    
    # Визуализация
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.title(f"Actual vs Predicted (n_estimators={exp['n_estimators']}, max_depth={exp['max_depth']})")
    plt.xlabel("Actual Quality")
    plt.ylabel("Predicted Quality")
    plot_filename = f"actual_vs_predicted_{exp['n_estimators']}_{exp['max_depth']}.png"
    plt.savefig(plot_filename)
    sub_task.upload_artifact("Prediction Plot", plot_filename)
    plt.close()
    
    # Завершаем подзадачу
    sub_task.close()

# Сравнительный график RMSE
df = pd.DataFrame(results)
plt.figure(figsize=(8, 6))
plt.bar(range(len(results)), df["rmse"], tick_label=[f"{r['n_estimators']}, {r['max_depth']}" for r in results])
plt.title("Comparison of RMSE Across Experiments")
plt.xlabel("Experiment (n_estimators, max_depth)")
plt.ylabel("RMSE")
plt.savefig("rmse_comparison.png")
task.upload_artifact("RMSE Comparison Plot", "rmse_comparison.png")
plt.close()

# Завершаем основную задачу
task.close()
