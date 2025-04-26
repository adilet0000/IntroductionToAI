# frozenlake_game_rl.py

import pygame
import random
import numpy as np
import os

# --- Параметры игры ---
WIDTH, HEIGHT = 400, 400
GRID_SIZE = 4
CELL_SIZE = WIDTH // GRID_SIZE
FPS = 30

# Цвета
ICE_COLOR = (200, 200, 255)
HOLE_COLOR = (50, 50, 50)
GOAL_COLOR = (0, 255, 0)
AGENT_COLOR = (0, 0, 255)

# Карта
MAP = [
   ['S', 'F', 'F', 'F'],
   ['F', 'H', 'F', 'H'],
   ['F', 'F', 'F', 'H'],
   ['H', 'F', 'F', 'G']
]

START_POS = (0, 0)

# --- Q-Learning параметры ---
LEARNING_RATE = 0.8
DISCOUNT = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.05
N_EPISODES = 1000

QTABLE_PATH = "frozenlake_qtable.npy"

class FrozenLakeEnv:
   def __init__(self):
      self.reset()

   def reset(self):
      self.agent_pos = list(START_POS)
      return self.get_state()

   def step(self, action):
      x, y = self.agent_pos

      if action == 0 and y > 0:  # вверх
         y -= 1
      elif action == 1 and x < GRID_SIZE - 1:  # вправо
         x += 1
      elif action == 2 and y < GRID_SIZE - 1:  # вниз
         y += 1
      elif action == 3 and x > 0:  # влево
         x -= 1

      self.agent_pos = [x, y]
      cell = MAP[y][x]

      if cell == 'H':
         return self.get_state(), -1, True  # Упал в дыру
      elif cell == 'G':
         return self.get_state(), 1, True   # Дошёл до цели
      else:
         return self.get_state(), 0, False  # Обычный лёд

   def get_state(self):
      return self.agent_pos[1] * GRID_SIZE + self.agent_pos[0]

# --- Pygame запуск ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('FrozenLake RL Game')
clock = pygame.time.Clock()

env = FrozenLakeEnv()
q_table = np.zeros((GRID_SIZE * GRID_SIZE, 4))
episode = 0
state = env.reset()
done = False

font = pygame.font.SysFont('Arial', 24)

def draw():
   screen.fill((0, 0, 0))

   for y in range(GRID_SIZE):
      for x in range(GRID_SIZE):
         rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)

         if MAP[y][x] == 'H':
            pygame.draw.rect(screen, HOLE_COLOR, rect)
         elif MAP[y][x] == 'G':
            pygame.draw.rect(screen, GOAL_COLOR, rect)
         else:
            pygame.draw.rect(screen, ICE_COLOR, rect)

         pygame.draw.rect(screen, (0, 0, 0), rect, 2)

   agent_rect = pygame.Rect(env.agent_pos[0] * CELL_SIZE + 10, env.agent_pos[1] * CELL_SIZE + 10,
                            CELL_SIZE - 20, CELL_SIZE - 20)
   pygame.draw.rect(screen, AGENT_COLOR, agent_rect)

   # Эпизод и эпсилон
   episode_text = font.render(f'Episode: {episode}', True, (255, 255, 255))
   screen.blit(episode_text, (10, HEIGHT - 30))

   epsilon_text = font.render(f'Epsilon: {EPSILON:.2f}', True, (255, 255, 255))
   screen.blit(epsilon_text, (WIDTH - 150, HEIGHT - 30))

   pygame.display.flip()

# --- Главное меню ---
mode = input("Enter 'train' to train, or 'test' to load agent: ").strip()

if mode == 'train':
   running = True
   while running and episode < N_EPISODES:
      clock.tick(FPS)

      for event in pygame.event.get():
         if event.type == pygame.QUIT:
            running = False

      # Выбираем действие
      if random.uniform(0, 1) < EPSILON:
         action = random.randint(0, 3)
      else:
         action = np.argmax(q_table[state])

      next_state, reward, done = env.step(action)

      # Обновляем Q-таблицу
      q_table[state, action] = (1 - LEARNING_RATE) * q_table[state, action] + \
                               LEARNING_RATE * (reward + DISCOUNT * np.max(q_table[next_state]))

      state = next_state

      draw()

      if done:
         episode += 1
         state = env.reset()
         EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

   # Сохраняем обученную модель
   np.save(QTABLE_PATH, q_table)
   print("Training complete. Q-table saved.")

elif mode == 'test':
   if os.path.exists(QTABLE_PATH):
      q_table = np.load(QTABLE_PATH)
      EPSILON = 0.0
      print("Loaded trained Q-table.")
   else:
      print("No trained Q-table found. Please train first.")
      pygame.quit()
      sys.exit()

   running = True
   while running:
      clock.tick(FPS)

      for event in pygame.event.get():
         if event.type == pygame.QUIT:
            running = False

      action = np.argmax(q_table[state])
      next_state, reward, done = env.step(action)
      state = next_state

      draw()

      if done:
         state = env.reset()

else:
   print("Unknown mode. Exiting.")
   pygame.quit()
   sys.exit()

pygame.quit()
