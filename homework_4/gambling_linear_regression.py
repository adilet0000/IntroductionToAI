import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Определяем масти и ранги карт
suits = ['♠', '♥', '♦', '♣']
ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

# Формируем колоду (52 карты)
deck = [rank + suit for suit in suits for rank in ranks]

# Сопоставляем ранг числовому значению (для расчёта силы руки)
rank_to_value = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
}

def simulate_game():
    """
    Симулируем одну раздачу:
      - Перемешиваем колоду и раздаем 2 карты игроку и 5 общих карт.
      - Генерируем случайный размер ставки (от 10 до 100).
      - Вычисляем силу руки как сумму значений двух карт игрока.
      - На основе силы руки и размера ставки (с добавлением шума) рассчитываем «score».
      - Применяем сигмоиду, чтобы получить вероятность выигрыша, и случайно определяем результат.
    """
    cards = deck.copy()
    random.shuffle(cards)
    
    # Раздаем карты: 2 карты игроку и 5 общих (flop, turn, river)
    hand = cards[:2]
    board = cards[2:7]
    
    # Случайный размер ставки
    bet = random.uniform(10, 100)
    
    # Считаем силу руки (сумма значений двух карт)
    hand_strength = sum([rank_to_value[card[:-1]] for card in hand])
    
    # Параметры модели для расчёта базового score
    # Ожидаемое среднее для двух карт ~16 (при равномерном распределении)
    # Средняя ставка равна (10+100)/2 = 55
    alpha = 0.05      # влияние силы руки
    beta_coeff = 0.01 # влияние размера ставки
    avg_bet = 55
    bias = 0.5
    noise = np.random.normal(0, 0.05)
    
    # Линейная комбинация признаков с шумом
    score = bias + alpha * (hand_strength - 16) + beta_coeff * (bet - avg_bet) + noise
    
    # Преобразуем score в вероятность с помощью сигмоидальной функции
    prob = 1 / (1 + np.exp(-score))
    
    # Определяем исход раздачи (win=1, lose=0)
    win = 1 if random.random() < prob else 0
    
    return {
        'hand': hand,
        'board': board,
        'bet': bet,
        'hand_strength': hand_strength,
        'win': win
    }

# Генерируем датасет (например, 1000 раздач)
n_games = 1000
data = [simulate_game() for _ in range(n_games)]
df = pd.DataFrame(data)

# Выводим первые строки датасета для ознакомления
print("Пример данных:")
print(df.head())

# Для обучения модели используем признаки: 'hand_strength' и 'bet'
X = df[['hand_strength', 'bet']]
y = df['win']

# Делим данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучаем модель линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Делаем предсказания и оцениваем модель
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nРезультаты модели:")
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)
print("Коэффициенты:", model.coef_)
print("Свободный член (Intercept):", model.intercept_)

# Анализ зависимости выигрыша от силы руки
df_grouped = df.groupby('hand_strength')['win'].mean().reset_index()

plt.figure(figsize=(10, 6))
plt.scatter(df_grouped['hand_strength'], df_grouped['win'], alpha=0.7)
plt.title('Зависимость вероятности выигрыша от силы руки')
plt.xlabel('Сила руки (сумма значений карт)')
plt.ylabel('Вероятность выигрыша')
plt.grid(True)
plt.show()

# Анализ влияния размера ставки на исход раздачи
# Бинним размер ставки для визуализации
df['bet_bin'] = pd.cut(df['bet'], bins=10)
bet_grouped = df.groupby('bet_bin')['win'].mean().reset_index()

plt.figure(figsize=(10, 6))
plt.bar(bet_grouped['bet_bin'].astype(str), bet_grouped['win'])
plt.title('Зависимость вероятности выигрыша от размера ставки')
plt.xlabel('Размер ставки (биновое разбиение)')
plt.ylabel('Вероятность выигрыша')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Вывод основных выводов
print("\nАнализ результатов:")
print("1. Руки с большей суммой значений карт (hand_strength) выигрывают чаще.")
print("2. Увеличение размера ставки (bet) положительно коррелирует с шансом на победу, хотя влияние может быть слабее.")
print("3. Полученные зависимости могут служить базой для разработки стратегий, позволяющих повысить вероятность выигрыша.")
