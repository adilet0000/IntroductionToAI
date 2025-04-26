import numpy as np
import pygame
import sys
import random

class MinesweeperEnv:
   def __init__(self, width=30, height=16, n_mines=99, cell_size=25):
      pygame.init()
      self.width = width
      self.height = height
      self.n_mines = n_mines
      self.cell_size = cell_size

      self.screen = pygame.display.set_mode((self.width * cell_size, self.height * cell_size))
      pygame.display.set_caption('Minesweeper')
      self.font = pygame.font.SysFont('Arial', 18)
      self.big_font = pygame.font.SysFont('Arial', 48)
      self.colors = {
         'covered': (192, 192, 192),
         'uncovered': (224, 224, 224),
         'mine': (255, 0, 0),
         'text': (0, 0, 0),
         'win': (0, 255, 0),
         'lose_bg': (255, 100, 100),
         'win_bg': (100, 255, 100)
      }
      self.number_colors = {
         1: (0, 0, 255),       # Синий
         2: (0, 128, 0),       # Зеленый
         3: (255, 0, 0),       # Красный
         4: (0, 0, 128),       # Темно-синий
         5: (128, 0, 0),       # Коричневый
         6: (0, 128, 128),     # Бирюзовый
         7: (0, 0, 0),         # Черный
         8: (128, 128, 128)    # Серый
      }

      self.reset()

   def reset(self):
      self.board = np.full((self.height, self.width), -1)
      self.revealed = np.zeros((self.height, self.width), dtype=bool)
      self.flags = np.zeros((self.height, self.width), dtype=bool)
      self.done = False
      self.win = False
      self.first_click = True
      self.mines = set()
      return self._get_observation()

   def _in_bounds(self, x, y):
      return 0 <= x < self.width and 0 <= y < self.height

   def _get_neighbors(self, x, y):
      neighbors = []
      for dx in [-1, 0, 1]:
         for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
               continue
            nx, ny = x + dx, y + dy
            if self._in_bounds(nx, ny):
               neighbors.append((nx, ny))
      return neighbors

   def _place_mines(self, safe_zone):
      all_cells = [(x, y) for x in range(self.width) for y in range(self.height)]
      available_cells = [cell for cell in all_cells if cell not in safe_zone]
      self.mines = set(random.sample(available_cells, self.n_mines))

   def _count_adjacent_mines(self, x, y):
      return sum((nx, ny) in self.mines for nx, ny in self._get_neighbors(x, y))

   def step(self, action):
      if self.done:
         return self._get_observation(), 0, self.done

      x, y, action_type = action
      reward = 0

      if action_type == 1:  # Флажок
         if not self.revealed[y, x]:
            self.flags[y, x] = not self.flags[y, x]
         return self._get_observation(), reward, self.done

      # Открытие клетки
      if self.first_click:
         safe_zone = self._get_neighbors(x, y) + [(x, y)]
         self._place_mines(safe_zone)
         self.first_click = False

      if (x, y) in self.mines:
         self.board[y, x] = -2
         self.revealed[y, x] = True
         reward = -100
         self.done = True
         self.win = False
      else:
         reward = 1
         self._reveal(x, y)

         if self._check_win():
            reward = 500
            self.done = True
            self.win = True

      return self._get_observation(), reward, self.done

   def _reveal(self, x, y):
      if not self._in_bounds(x, y) or self.revealed[y, x] or self.flags[y, x]:
         return
      self.revealed[y, x] = True
      self.board[y, x] = self._count_adjacent_mines(x, y)

      if self.board[y, x] == 0:
         for nx, ny in self._get_neighbors(x, y):
            if not self.revealed[ny, nx]:
               self._reveal(nx, ny)

   def _check_win(self):
      for y in range(self.height):
         for x in range(self.width):
            if (x, y) not in self.mines and not self.revealed[y, x]:
               return False
      return True

   def _get_observation(self):
      obs = np.copy(self.board)
      obs[~self.revealed] = -1
      return obs
   
   def _get_processed_observation(self):
      obs_opened = self.revealed.astype(np.float32)
      obs_numbers = np.clip(self.board, 0, 8) / 8.0  # Нормируем числа от 0 до 1
      obs_flags = self.flags.astype(np.float32)
      return np.stack([obs_opened, obs_numbers, obs_flags], axis=0)  # (3, height, width)


   def render(self):
      if self.done:
         bg_color = self.colors['win_bg'] if self.win else self.colors['lose_bg']
         self.screen.fill(bg_color)
      else:
         self.screen.fill((0, 0, 0))

      for y in range(self.height):
         for x in range(self.width):
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)

            if self.revealed[y, x]:
               if (x, y) in self.mines:
                  pygame.draw.rect(self.screen, self.colors['mine'], rect, border_radius=3)
               else:
                  pygame.draw.rect(self.screen, self.colors['uncovered'], rect, border_radius=3)
                  count = self.board[y, x]
                  if count > 0:
                     color = self.number_colors.get(count, self.colors['text'])
                     text_surface = self.font.render(str(count), True, color)
                     text_rect = text_surface.get_rect(center=rect.center)
                     self.screen.blit(text_surface, text_rect)
            else:
               if self.flags[y, x]:
                  pygame.draw.rect(self.screen, (255, 255, 0), rect, border_radius=3)
                  flag_center = rect.center
                  pygame.draw.polygon(self.screen, (255, 0, 0), [
                     (flag_center[0], flag_center[1] - 8),
                     (flag_center[0] + 6, flag_center[1]),
                     (flag_center[0], flag_center[1] + 8)
                  ])
               else:
                  if self.done and (x, y) in self.mines:
                     pygame.draw.rect(self.screen, self.colors['mine'], rect, border_radius=3)
                  else:
                     pygame.draw.rect(self.screen, self.colors['covered'], rect, border_radius=3)

            pygame.draw.rect(self.screen, (0, 0, 0), rect, 1, border_radius=3)

      if self.done:
         text = "You Win!" if self.win else "Game Over"
         text_surface = self.big_font.render(text, True, (0, 0, 0))
         text_rect = text_surface.get_rect(center=(self.width * self.cell_size // 2,
                                                    self.height * self.cell_size // 2))
         self.screen.blit(text_surface, text_rect)

      pygame.display.flip()

   def handle_events(self):
      for event in pygame.event.get():
         if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
         if event.type == pygame.MOUSEBUTTONDOWN and not self.done:
            x, y = pygame.mouse.get_pos()
            grid_x = x // self.cell_size
            grid_y = y // self.cell_size

            if event.button == 1:  # Левая кнопка — открыть
               return (grid_x, grid_y, 0)
            elif event.button == 3:  # Правая кнопка — флажок
               return (grid_x, grid_y, 1)
      return None

if __name__ == "__main__":
   env = MinesweeperEnv(width=9, height=9, n_mines=10)
   obs = env.reset()

   clock = pygame.time.Clock()

   while True:
      env.render()
      action = env.handle_events()
      if action:
         obs, reward, done = env.step(action)
         print(f"Action: {action}, Reward: {reward}, Done: {done}")

      if env.done:
         for _ in range(90):
            env.render()
            pygame.time.delay(33)
            env.handle_events()

         obs = env.reset()

      clock.tick(30)
