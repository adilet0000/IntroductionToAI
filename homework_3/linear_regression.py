import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Загружаем переданный датасет
file_path = "Housing.csv"
df = pd.read_csv(file_path)

# Преобразуем категориальные переменные в числовые
categorical_columns = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea", "furnishingstatus"]
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Выбираем признаки и целевую переменную
X = df.drop(columns=["price"])
y = df["price"]

# Масштабируем данные
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение данных на train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Создание и обучение модели
model = LinearRegression()
model.fit(X_train, y_train)

# Предсказания
predictions = model.predict(X_test)

# Оценка модели
mse = mean_squared_error(y_test, predictions)
r2_score = model.score(X_test, y_test) * 100  # R² в процентах

print(f"Mean Squared Error: {mse:.2f}")
print(f"Точность модели (R²): {r2_score:.2f}%")

# Рассчитываем комбинированный признак (сумма преимуществ дома)
df_test = pd.DataFrame(X_test, columns=X.columns)  # Восстанавливаем имена признаков
df_test["SUM_FEATURES"] = df_test.sum(axis=1)  # Суммируем все характеристики
df_test["PREDICTED_PRICE"] = predictions
df_test["ACTUAL_PRICE"] = y_test.values

# Отображаем график предсказанных цен по сумме характеристик дома
plt.figure(figsize=(10, 6))
plt.scatter(df_test["SUM_FEATURES"], df_test["ACTUAL_PRICE"], color="blue", label="Фактические цены", edgecolors="black", s=50)
plt.scatter(df_test["SUM_FEATURES"], df_test["PREDICTED_PRICE"], color="green", marker="x", s=50, label="Предсказанные цены")
plt.plot(sorted(df_test["SUM_FEATURES"]), sorted(df_test["PREDICTED_PRICE"]), color="red", linewidth=2, label="Линейная регрессия")

# Подписи осей и заголовок
plt.xlabel("Комбинированный признак (сумма преимуществ дома)")
plt.ylabel("Цена дома ($)")
plt.title("Линейная зависимость цены от суммы преимуществ дома")
plt.legend()

# Отображение точности модели
accuracy_text = f"Точность модели (R²): {r2_score:.2f}%"
plt.figtext(0.15, 0.06, accuracy_text, wrap=True, horizontalalignment="left", fontsize=12, bbox={"facecolor": "lightgreen", "alpha": 0.8, "pad": 5})

# Показ графика
plt.show()