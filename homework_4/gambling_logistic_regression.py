import numpy as np
import pandas as pd
import random
from itertools import combinations
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import argparse
import json

# ------------------------------
# 1. Настройка карт и базовые функции
# ------------------------------

# Масти и ранги
suits = ['♠', '♥', '♦', '♣']
suits_map = {'♠': 0, '♥': 1, '♦': 2, '♣': 3}
ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
rank_to_value = {
    '2': 2,  '3': 3,  '4': 4,  '5': 5,  '6': 6,
    '7': 7,  '8': 8,  '9': 9,  '10': 10, 'J': 11,
    'Q': 12, 'K': 13, 'A': 14
}

# Формируем колоду из 52 карт (например, "A♠", "10♦", и т.д.)
deck = [rank + suit for suit in suits for rank in ranks]

def parse_card(card_str):
    """
    Преобразует строку карты (например, "10♣" или "A♥") в кортеж (rank_value, suit_value).
    """
    suit_symbol = card_str[-1]          # Последний символ – масть
    rank_str = card_str[:-1]            # Остальная часть строки – ранг
    return (rank_to_value[rank_str], suits_map[suit_symbol])

def is_straight(ranks_sorted_desc):
    """
    Проверяет, являются ли 5 отсортированных по убыванию рангов подряд идущими (стрит).
    Учитывает стандартный стрит и особый случай Ace-low (A,5,4,3,2).
    """
    # Стандартный стрит
    for i in range(4):
        if ranks_sorted_desc[i] != ranks_sorted_desc[i+1] + 1:
            break
    else:
        return True
    # Особый случай Ace-low (колесо)
    if set(ranks_sorted_desc) == {14, 5, 4, 3, 2}:
        return True
    return False

def evaluate_5_cards(cards_5):
    """
    Оценивает 5-карточную руку (список кортежей (rank, suit)).
    Возвращает категорию руки от 0 до 8:
      8 = Straight Flush, 7 = Four of a Kind, 6 = Full House,
      5 = Flush, 4 = Straight, 3 = Three of a Kind,
      2 = Two Pair, 1 = One Pair, 0 = High Card.
    (Тайбрейки не учитываются.)
    """
    ranks_ = sorted([c[0] for c in cards_5], reverse=True)
    suits_ = [c[1] for c in cards_5]
    
    flush = (len(set(suits_)) == 1)
    straight = is_straight(ranks_)
    
    rank_counts = Counter(ranks_)
    freq = sorted(rank_counts.values(), reverse=True)
    
    if straight and flush:
        return 8
    elif 4 in freq:
        return 7
    elif 3 in freq and 2 in freq:
        return 6
    elif flush:
        return 5
    elif straight:
        return 4
    elif 3 in freq:
        return 3
    elif freq.count(2) == 2:
        return 2
    elif 2 in freq:
        return 1
    else:
        return 0

def best_5_from_7(cards_7):
    """
    Перебирает все 5-карточные комбинации из 7 и возвращает лучший рейтинг (0–8).
    """
    best_rank = 0
    for combo in combinations(cards_7, 5):
        rank_5 = evaluate_5_cards(combo)
        if rank_5 > best_rank:
            best_rank = rank_5
    return best_rank

# ------------------------------
# 2. Генерация синтетических раздач (для обучения модели)
# ------------------------------

def simulate_game():
    """
    Симулирует одну раздачу:
      - Перемешивает колоду и раздает 2 карты игроку и 5 общих.
      - Вычисляет лучший рейтинг руки (hand_rank) из 7 карт.
      - Генерирует случайный размер ставки (bet) от 10 до 100.
      - Вычисляет score как линейную комбинацию (с шумом) от hand_rank и bet.
      - Преобразует score в вероятность выигрыша через сигмоиду.
      - Определяет win (1, если выигрыш, 0, если проигрыш).
    """
    cards_copy = deck.copy()
    random.shuffle(cards_copy)
    hand = cards_copy[:2]
    board = cards_copy[2:7]
    all_7_cards = [parse_card(c) for c in (hand + board)]
    hand_rank = best_5_from_7(all_7_cards)
    bet = random.uniform(10, 100)
    
    # Параметры логистической функции (коэффициенты подобраны для симуляции)
    avg_rank = 3.5   # центр по hand_rank (0–8)
    avg_bet = 55     # средняя ставка
    bias = 0.0
    alpha = 0.8      # влияние hand_rank
    beta_coeff = 0.01  # влияние ставки
    noise = np.random.normal(0, 0.5)
    
    score = bias + alpha * (hand_rank - avg_rank) + beta_coeff * (bet - avg_bet) + noise
    prob = 1 / (1 + np.exp(-score))
    win = 1 if random.random() < prob else 0
    
    return {
        'hand': hand,
        'board': board,
        'hand_rank': hand_rank,
        'bet': bet,
        'win': win
    }

# Генерируем обучающий датасет
n_games = 1000
data = [simulate_game() for _ in range(n_games)]
df = pd.DataFrame(data)

# ------------------------------
# 3. Обучение модели логистической регрессии
# ------------------------------

X = df[['hand_rank', 'bet']]
y = df['win']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Выводим оценку модели
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Оценка точности (Accuracy): {accuracy:.2%}")
# print(f"\nОтчёт о классификации:\n{classification_report(y_test, y_pred)}")
# print(f"Коэффициенты модели: {model.coef_}")
# print(f"Свободный член (Intercept): {model.intercept_}")

# ------------------------------
# 4. Функция предсказания шанса выигрыша для пользовательского ввода
# ------------------------------

def predict_win_probability(my_hand, board, bet=None, completions=1000):
    """
    Принимает:
      my_hand: список из 2 карт (например, ["A♠", "K♠"])
      board: список открытых карт (от 0 до 5)
      bet: размер ставки (если не задан, используется среднее значение 55)
      completions: число симуляций для докачки недостающих карт, если board неполный
    Возвращает среднюю предсказанную вероятность выигрыша (от 0 до 1) по логистической регрессии.
    """
    if bet is None:
        bet = 55.0  # используем среднюю ставку, если не задано
    
    probs = []
    # Если board неполный, докачиваем недостающие карты completions раз
    missing = 5 - len(board)
    # Преобразуем ваши карты и уже известные карты в формат (rank, suit)
    my_hand_parsed = [parse_card(card) for card in my_hand]
    board_parsed = [parse_card(card) for card in board]
    # Список известных карт (чтобы не повторять)
    known = my_hand + board
    full_deck_list = [c for c in deck if c not in known]
    
    if missing > 0:
        for _ in range(completions):
            # Перемешиваем оставшуюся колоду и добираем недостающие карты
            random.shuffle(full_deck_list)
            extra = full_deck_list[:missing]
            # Формируем полный борд
            full_board = board + extra
            full_board_parsed = [parse_card(card) for card in full_board]
            all_7 = my_hand_parsed + full_board_parsed
            hand_rank = best_5_from_7(all_7)
            # Формируем DataFrame для модели
            X_input = pd.DataFrame({'hand_rank': [hand_rank], 'bet': [bet]})
            prob = model.predict_proba(X_input)[0, 1]
            probs.append(prob)
        avg_prob = np.mean(probs)
    else:
        # Если board полон (5 карт), сразу предсказываем
        board_parsed = [parse_card(card) for card in board]
        all_7 = my_hand_parsed + board_parsed
        hand_rank = best_5_from_7(all_7)
        X_input = pd.DataFrame({'hand_rank': [hand_rank], 'bet': [bet]})
        avg_prob = model.predict_proba(X_input)[0, 1]
    
    return avg_prob

# ------------------------------
# 5. Основная функция для пользовательского ввода через JSON
# ------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Расчет шансов победы (win probability) с использованием логистической регрессии.\n"
                    "Формат входного JSON-файла:\n"
                    '{"hand": ["A♠", "K♠"], "board": ["10♣", "J♦", "Q♥"], "bet": 80}'
    )
    parser.add_argument("input_file", help="Путь к входному JSON-файлу")
    parser.add_argument("--completions", type=int, default=1000,
                        help="Количество симуляций для докачки недостающих карт (по умолчанию 1000)")
    args = parser.parse_args()
    
    # Чтение входного JSON-файла
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Проверяем, что ключ 'hand' присутствует
    if "hand" not in data:
        print("Ошибка: в JSON не найден ключ 'hand'.")
        return
    
    # Получаем значения ключей напрямую из словаря
    my_hand = data["hand"]
    board = data["board"] if "board" in data else []
    bet = data["bet"] if "bet" in data else 55.0

    # Если board содержит более 5 карт, выдаем ошибку
    if len(board) > 5:
        print("Ошибка: в 'board' не может быть больше 5 карт.")
        return
    
    win_prob = predict_win_probability(my_hand, board, bet=bet, completions=args.completions)
    print(f"\nВаши карты: {my_hand}")
    print(f"Карты на столе: {board}")
    print(f"Ставка: {bet}")
    print(f"Расчетное преимущество (win probability): {win_prob:.2%}")

if __name__ == "__main__":
    main()