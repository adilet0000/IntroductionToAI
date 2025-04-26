# smart_logic_plus_agent.py

import numpy as np
import pygame
import sys
import random
import matplotlib.pyplot as plt
from minesweeper import MinesweeperEnv  # Импорт твоей игры

class SmartLogicPlusAgent:
   def __init__(self, env):
      self.env = env
      self.width = env.width
      self.height = env.height

   def apply_logic(self):
      for y in range(self.height):
         for x in range(self.width):
            if not self.env.revealed[y, x]:
               continue
            number = self.env.board[y, x]
            if number <= 0:
               continue

            neighbors = self.env._get_neighbors(x, y)
            closed_neighbors = [(nx, ny) for nx, ny in neighbors if not self.env.revealed[ny, nx] and not self.env.flags[ny, nx]]
            flagged_neighbors = [(nx, ny) for nx, ny in neighbors if self.env.flags[ny, nx]]

            if len(closed_neighbors) > 0 and number - len(flagged_neighbors) == len(closed_neighbors):
               # Все закрытые соседи — это мины
               nx, ny = closed_neighbors[0]
               return (nx, ny, 1)  # Поставить флажок

            if number == len(flagged_neighbors) and len(closed_neighbors) > 0:
               # Все оставшиеся безопасны
               nx, ny = closed_neighbors[0]
               return (nx, ny, 0)  # Открыть безопасную клетку

      return None

   def estimate_safe_move(self):
      # Простая оценка вероятности: кликаем на клетку с минимальным риском
      min_risk = 1.1
      best_cell = None

      total_mines_left = self.env.n_mines - np.sum(self.env.flags)
      total_cells_left = np.sum(~self.env.revealed & ~self.env.flags)

      if total_cells_left == 0:
         return None

      base_probability = total_mines_left / total_cells_left

      for y in range(self.height):
         for x in range(self.width):
            if not self.env.revealed[y, x] and not self.env.flags[y, x]:
               # В данной базовой версии считаем, что все клетки равнозначны
               probability = base_probability

               if probability < min_risk:
                  min_risk = probability
                  best_cell = (x, y)

      if best_cell:
         return (best_cell[0], best_cell[1], 0)  # Открыть
      else:
         return None

   def get_action(self):
      logic_action = self.apply_logic()
      if logic_action is not None:
         return logic_action

      safe_action = self.estimate_safe_move()
      if safe_action is not None:
         return safe_action

      # если совсем ничего — случайный ход
      return self.random_action()

   def random_action(self):
      safe_moves = []
      for y in range(self.height):
         for x in range(self.width):
            if not self.env.revealed[y, x] and not self.env.flags[y, x]:
               safe_moves.append((x, y))

      if safe_moves:
         x, y = random.choice(safe_moves)
         return (x, y, 0)  # Открыть

      x = random.randint(0, self.width - 1)
      y = random.randint(0, self.height - 1)
      return (x, y, 0)

# --- Автотестирование 100 игр ---
def auto_test(n_games=100):
   env = MinesweeperEnv(width=9, height=9, n_mines=10)  # Можно поменять на 16x30 позже
   wins = []

   for game_idx in range(n_games):
      agent = SmartLogicPlusAgent(env)
      state = env.reset()
      done = False

      while not done:
         action = agent.get_action()
         state, reward, done = env.step(action)

      wins.append(1 if reward > 0 else 0)
      print(f"Game {game_idx + 1}/{n_games}: {'WIN' if reward > 0 else 'LOSS'}")

   # Показать статистику
   win_rate = np.mean(wins) * 100
   print(f"\n=== Results after {n_games} games ===")
   print(f"Total Wins: {np.sum(wins)} / {n_games}")
   print(f"Win Rate: {win_rate:.2f}%")

   # Построить график
   plt.plot(np.cumsum(wins))
   plt.xlabel('Игры')
   plt.ylabel('Количество побед')
   plt.title('Прогресс побед агента')
   plt.grid(True)
   plt.show()

# --- Реальный тест для визуализации ---
def visual_test():
   env = MinesweeperEnv(width=9, height=9, n_mines=10)
   agent = SmartLogicPlusAgent(env)

   state = env.reset()
   clock = pygame.time.Clock()
   running = True

   while running:
      for event in pygame.event.get():
         if event.type == pygame.QUIT:
            running = False
            pygame.quit()
            sys.exit()

      env.render()
      pygame.time.delay(300)

      action = agent.get_action()
      state, reward, done = env.step(action)

      if done:
         env.render()
         font = pygame.font.SysFont('Arial', 48)
         text = "ПОБЕДА!" if reward > 0 else "ПРОИГРЫШ"
         color = (0, 255, 0) if reward > 0 else (255, 0, 0)
         text_surface = font.render(text, True, color)
         text_rect = text_surface.get_rect(center=(env.width * env.cell_size // 2,
                                                    env.height * env.cell_size // 2))
         env.screen.blit(text_surface, text_rect)
         pygame.display.flip()
         pygame.time.delay(3000)

         state = env.reset()

      clock.tick(30)

if __name__ == "__main__":
   mode = input("Enter 'auto' for 100 games test, or 'visual' for live play: ").strip()

   if mode == 'auto':
      auto_test()
   elif mode == 'visual':
      visual_test()
   else:
      print("Unknown mode.")
