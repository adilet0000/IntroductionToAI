import numpy as np
import random
import os

import sys
sys.setrecursionlimit(10000)

# Размер лабиринта (должен быть нечётным для корректной генерации)
WIDTH = 101
HEIGHT = 101
SAVE_PATH = "saved/maze_21x21.npy"

# Направления: (dx, dy)
DIRS = [(0, -2), (2, 0), (0, 2), (-2, 0)]

def generate_maze(width, height):
   maze = np.ones((height, width), dtype=np.int8)  # 1 = стена, 0 = путь

   def in_bounds(x, y):
      return 0 < x < width - 1 and 0 < y < height - 1

   def carve_passages_from(x, y):
      maze[y, x] = 0
      random.shuffle(DIRS)
      for dx, dy in DIRS:
         nx, ny = x + dx, y + dy
         mx, my = x + dx // 2, y + dy // 2
         if in_bounds(nx, ny) and maze[ny, nx] == 1:
            maze[my, mx] = 0
            carve_passages_from(nx, ny)

   # Начинаем с левого верхнего угла
   carve_passages_from(1, 1)
   
   # Начальная и конечная точки
   maze[1, 1] = 0  # Старт
   # Генерация списка всех проходимых клеток
   paths = [(x, y) for y in range(height) for x in range(width)
            if maze[y, x] == 0 and (x, y) != (1, 1)]

   # Выбираем самую дальнюю клетку от старта
   goal = max(paths, key=lambda pos: abs(pos[0] - 1) + abs(pos[1] - 1))
   gx, gy = goal
   maze[gy, gx] = 0

   return maze

def save_maze(maze, path):
   os.makedirs(os.path.dirname(path), exist_ok=True)
   np.save(path, maze)
   print(f"Maze saved to {path}")

def main():
   maze = generate_maze(WIDTH, HEIGHT)
   save_maze(maze, SAVE_PATH)

   # Визуализация в консоли
   for row in maze:
      print(''.join(['█' if cell == 1 else ' ' for cell in row]))

if __name__ == "__main__":
   main()
