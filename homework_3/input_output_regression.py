# Импорт необходимых библиотек
import numpy as np
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

# Функция для предсказания цены на основе пользовательского ввода
def predict_house_price(area, bedrooms, bathrooms, stories, mainroad, guestroom,
                        basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus):
   # Создаем DataFrame с одним объектом
   input_data = pd.DataFrame([{
      "area": area,
      "bedrooms": bedrooms,
      "bathrooms": bathrooms,
      "stories": stories,
      "mainroad": mainroad,
      "guestroom": guestroom,
      "basement": basement,
      "hotwaterheating": hotwaterheating,
      "airconditioning": airconditioning,
      "parking": parking,
      "prefarea": prefarea,
      "furnishingstatus": furnishingstatus
   }])

   # Преобразуем категориальные переменные так же, как в обучающей выборке
   input_data = pd.get_dummies(input_data, columns=categorical_columns, drop_first=True)

   # Добавляем недостающие колонки (если вдруг в тесте нет каких-то категорий)
   missing_cols = set(X.columns) - set(input_data.columns)
   for col in missing_cols:
      input_data[col] = 0  # Заполняем их нулями

   # Упорядочиваем колонки так же, как в X
   input_data = input_data[X.columns]

   # Масштабируем вводные данные
   input_scaled = scaler.transform(input_data)

   # Предсказываем цену
   predicted_price = model.predict(input_scaled)[0]

   return round(predicted_price, 2)

# # Пример использования функции
# predicted_price = predict_house_price(
#    area=5000, bedrooms=4, bathrooms=3, stories=2,
#    mainroad="yes", guestroom="no", basement="no",
#    hotwaterheating="no", airconditioning="yes",
#    parking=2, prefarea="yes", furnishingstatus="semi-furnished"
# )

# predicted_price


if __name__ == "__main__":
   area = float(input("Введите площадь (area): "))
   bedrooms = int(input("Введите количество спален (bedrooms): "))
   bathrooms = int(input("Введите количество ванных комнат (bathrooms): "))
   stories = int(input("Введите количество этажей (stories): "))
   mainroad = input("Находится ли дом на главной дороге? (yes/no): ")
   guestroom = input("Есть ли гостевая комната? (yes/no): ")
   basement = input("Есть ли подвал? (yes/no): ")
   hotwaterheating = input("Есть ли горячее водоснабжение? (yes/no): ")
   airconditioning = input("Есть ли кондиционер? (yes/no): ")
   parking = int(input("Количество парковочных мест (parking): "))
   prefarea = input("Дом в престижном районе? (yes/no): ")
   furnishingstatus = input("Статус меблировки (furnished/semi-furnished/unfurnished): ")

   predicted_price = predict_house_price(area, bedrooms, bathrooms, stories, 
                                         mainroad, guestroom, basement, hotwaterheating, 
                                         airconditioning, parking, prefarea, furnishingstatus)

   print(f"Предсказанная цена дома: ${predicted_price}")
