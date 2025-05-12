import pygame
import numpy as np

CELL_SIZE = 8
AGENT_COLOR = (50, 150, 255)
EXIT_COLOR = (0, 255, 100)
WALL_COLOR = (30, 30, 30)
PATH_COLOR = (230, 230, 230)
VISITED_COLOR = (180, 220, 250)

class MazeEnv:
   def __init__(self, maze_file):
      self.maze = np.load(maze_file)
      self.height, self.width = self.maze.shape
      self.start = (1, 1)
      self.goal = (self.width - 2, self.height - 2)

      pygame.init()
      self.screen = pygame.display.set_mode((self.width * CELL_SIZE, self.height * CELL_SIZE))
      pygame.display.set_caption("Maze RL Environment")
      self.clock = pygame.time.Clock()
      self.agent_pos = list(self.start)
      self.visited = set()

   def reset(self):
      self.agent_pos = list(self.start)
      self.visited = set()
      return tuple(self.agent_pos)

   def step(self, action):
      dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]  # up, right, down, left
      new_x = self.agent_pos[0] + dx
      new_y = self.agent_pos[1] + dy

      reward = -1
      done = False

      if self.maze[new_y, new_x] == 0:
         self.agent_pos = [new_x, new_y]
         self.visited.add((new_x, new_y))
         if (new_x, new_y) == self.goal:
            reward = 100
            done = True
      else:
         reward = -5  # ударился в стену

      return tuple(self.agent_pos), reward, done

   def render(self, delay=0):
      self.screen.fill((0, 0, 0))

      for y in range(self.height):
         for x in range(self.width):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if self.maze[y, x] == 1:
               pygame.draw.rect(self.screen, WALL_COLOR, rect)
            else:
               pygame.draw.rect(self.screen, PATH_COLOR, rect)

      # Отмечаем посещённые клетки (след агента)
      for vx, vy in self.visited:
         rect = pygame.Rect(vx * CELL_SIZE, vy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
         pygame.draw.rect(self.screen, VISITED_COLOR, rect)

      # Рисуем выход
      gx, gy = self.goal
      goal_rect = pygame.Rect(gx * CELL_SIZE, gy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
      pygame.draw.rect(self.screen, EXIT_COLOR, goal_rect)

      # Рисуем агента
      ax, ay = self.agent_pos
      agent_rect = pygame.Rect(ax * CELL_SIZE + 4, ay * CELL_SIZE + 4, CELL_SIZE - 8, CELL_SIZE - 8)
      pygame.draw.rect(self.screen, AGENT_COLOR, agent_rect, border_radius=6)

      pygame.display.flip()
      if delay:
         pygame.time.delay(delay)

   def close(self):
      pygame.quit()

   def handle_events(self):
      for event in pygame.event.get():
         if event.type == pygame.QUIT:
            pygame.quit()
            exit()

# Пример ручного прогона (можно удалить позже)
if __name__ == '__main__':
   env = MazeEnv("saved/maze_21x21.npy")
   env.reset()
   done = False

   while not done:
      env.render(delay=50)
      env.handle_events()
      action = np.random.choice(4)
      _, _, done = env.step(action)

   pygame.time.delay(1000)
   env.close()
