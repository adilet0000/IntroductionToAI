import json
import random
import sys
from itertools import combinations
from collections import Counter
import argparse

# ------------------------------
# 1. Определение констант и функций для разбора карт
# ------------------------------

suits = ['♠', '♥', '♦', '♣']
suits_map = {s: i for i, s in enumerate(suits)}
ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
rank_to_value = {
    '2': 2,  '3': 3,  '4': 4,  '5': 5,  '6': 6,
    '7': 7,  '8': 8,  '9': 9,  '10': 10, 'J': 11,
    'Q': 12, 'K': 13, 'A': 14
}

def parse_card(card_str):
    """
    Преобразует строку вида "10♣" или "A♥" в кортеж (rank_value, suit_value).
    """
    suit_symbol = card_str[-1]
    rank_str = card_str[:-1]
    return (rank_to_value[rank_str], suits_map[suit_symbol])

def card_str(card):
    """Возвращает строковое представление карты по кортежу (rank, suit)."""
    # Находим ключ по значению для масти
    suit_symbol = [s for s, val in suits_map.items() if val == card[1]][0]
    # Находим ранг по значению
    rank_str = [r for r, val in rank_to_value.items() if val == card[0]][0]
    return f"{rank_str}{suit_symbol}"

def full_deck():
    """Возвращает список строк всех 52 карт."""
    return [r + s for s in suits for r in ranks]

# ------------------------------
# 2. Функции оценки покерных рук
# ------------------------------

def evaluate_hand(cards_5):
    """
    Оценивает 5-карточную руку (список кортежей (rank, suit)).
    Возвращает кортеж (category, tiebreakers...),
    где category: 8 - Straight Flush, 7 - Four of a Kind, 6 - Full House,
      5 - Flush, 4 - Straight, 3 - Three of a Kind, 2 - Two Pair,
      1 - One Pair, 0 - High Card.
    Тiebreakers – кортеж чисел для сравнения в случае равенства категории.
    """
    # Список рангов и мастей
    ranks_list = [r for (r, s) in cards_5]
    suits_list = [s for (r, s) in cards_5]
    ranks_sorted = sorted(ranks_list, reverse=True)
    
    is_flush = len(set(suits_list)) == 1

    # Проверка стрита. Для стрита учитываем уникальные ранги.
    unique_ranks = sorted(set(ranks_list), reverse=True)
    is_straight = False
    straight_high = None
    if len(unique_ranks) >= 5:
        for i in range(len(unique_ranks) - 4):
            seq = unique_ranks[i:i+5]
            if seq[0] - seq[4] == 4:
                is_straight = True
                straight_high = seq[0]
                break
    # Проверка Ace-low straight
    if not is_straight and set([14, 5, 4, 3, 2]).issubset(set(ranks_list)):
        is_straight = True
        straight_high = 5

    # Подсчёт частот рангов
    freq = Counter(ranks_list)
    freq_sorted = sorted(freq.items(), key=lambda x: (x[1], x[0]), reverse=True)
    
    if is_straight and is_flush:
        return (8, straight_high)
    if freq_sorted[0][1] == 4:
        quad = freq_sorted[0][0]
        kicker = max([r for r in ranks_list if r != quad])
        return (7, quad, kicker)
    if freq_sorted[0][1] == 3 and freq_sorted[1][1] >= 2:
        triple = freq_sorted[0][0]
        pair = freq_sorted[1][0]
        return (6, triple, pair)
    if is_flush:
        return (5, ) + tuple(sorted(ranks_list, reverse=True))
    if is_straight:
        return (4, straight_high)
    if freq_sorted[0][1] == 3:
        triple = freq_sorted[0][0]
        kickers = sorted([r for r in ranks_list if r != triple], reverse=True)
        return (3, triple) + tuple(kickers)
    if freq_sorted[0][1] == 2 and freq_sorted[1][1] == 2:
        pair1 = freq_sorted[0][0]
        pair2 = freq_sorted[1][0]
        kicker = max([r for r in ranks_list if r != pair1 and r != pair2])
        return (2, max(pair1, pair2), min(pair1, pair2), kicker)
    if freq_sorted[0][1] == 2:
        pair = freq_sorted[0][0]
        kickers = sorted([r for r in ranks_list if r != pair], reverse=True)
        return (1, pair) + tuple(kickers)
    return (0, ) + tuple(ranks_sorted)

def best_hand_from_7(cards_7):
    """
    Перебирает все комбинации по 5 карт из 7 и возвращает максимальное значение руки.
    """
    best = None
    for combo in combinations(cards_7, 5):
        current = evaluate_hand(list(combo))
        if best is None or current > best:
            best = current
    return best

# ------------------------------
# 3. Monte Carlo симуляция equity для heads-up
# ------------------------------

def simulate_equity(my_hand_str, board_str, iterations=10000):
    """
    Расчитывает вероятность победы (equity) для ваших карт (my_hand_str) против одного оппонента.
    
    Аргументы:
      my_hand_str: список строк ваших карт (например, ["A♠", "K♠"])
      board_str: список строк открытых карт на столе (может быть от 0 до 5)
      iterations: число симуляций (по умолчанию 10000)
      
    Возвращает equity в виде доли (от 0 до 1).
    """
    # Преобразуем входные данные в формат (rank, suit)
    my_hand = [parse_card(c) for c in my_hand_str]
    board = [parse_card(c) for c in board_str]
    
    known_cards = my_hand + board
    deck_remaining = [c for c in full_deck() if c not in [card_str(card) for card in known_cards]]
    deck_remaining = [parse_card(c) for c in deck_remaining]
    
    wins = 0
    ties = 0
    total = iterations

    # Количество недостающих общих карт
    missing_board = 5 - len(board)
    
    for _ in range(iterations):
        # Копия оставшейся колоды
        deck_copy = deck_remaining.copy()
        random.shuffle(deck_copy)
        
        # Если board не полон, добираем недостающие карты
        extra_board = deck_copy[:missing_board]
        full_board = board + extra_board
        
        # Удаляем добранные карты из колоды
        deck_after_board = deck_copy[missing_board:]
        
        # Раздаем оппоненту 2 карты
        opp_hand = deck_after_board[:2]
        
        # Оцениваем руки: ваши 7 карт и оппонента 7 карт
        my_best = best_hand_from_7(my_hand + full_board)
        opp_best = best_hand_from_7(opp_hand + full_board)
        
        if my_best > opp_best:
            wins += 1
        elif my_best == opp_best:
            ties += 1
        # Иначе – поражение (ничего не добавляем)
    
    equity = (wins + 0.5 * ties) / total
    return equity

# ------------------------------
# 4. Основная функция: чтение файла и вывод результата
# ------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Расчет шансов победы (equity) в heads-up покере.\n"
                    "Формат входного JSON-файла:\n"
                    '{"hand": ["A♠", "K♠"], "board": ["10♣", "J♦", "Q♥"]}'
    )
    parser.add_argument("input_file", help="Путь к входному JSON-файлу")
    parser.add_argument("--iters", type=int, default=10000, help="Количество симуляций (по умолчанию 10000)")
    
    args = parser.parse_args()
    
    # Читаем входной JSON-файл
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    my_hand = data.get("hand", [])
    board = data.get("board", [])
    
    if len(my_hand) != 2:
        print("Ошибка: в 'hand' должно быть ровно 2 карты.")
        sys.exit(1)
    if len(board) > 5:
        print("Ошибка: в 'board' не может быть больше 5 карт.")
        sys.exit(1)
    
    equity = simulate_equity(my_hand, board, iterations=args.iters)
    print(f"\nВаши карты: {my_hand}")
    print(f"Карты на столе: {board}")
    print(f"Расчетное преимущество (equity): {equity:.2%}")

if __name__ == "__main__":
    main()
