# cnn_minesweeper.py

import os
import glob
import numpy as np
import pygame
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from multiprocessing import Pool

from minesweeper import MinesweeperEnv  # Твоя красивая версия игры

# --- CNN агент ---
class CNNAgent(nn.Module):
   def __init__(self, input_channels=1, width=30, height=16):
      super(CNNAgent, self).__init__()
      self.width = width
      self.height = height
      self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
      self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
      self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
      self.pool = nn.AdaptiveAvgPool2d((8, 8))
      self.fc1 = nn.Linear(128 * 8 * 8, 256)
      self.fc2 = nn.Linear(256, 2 * width * height)

   def forward(self, x):
      x = torch.relu(self.conv1(x))
      x = torch.relu(self.conv2(x))
      x = torch.relu(self.conv3(x))
      x = self.pool(x)
      x = x.view(x.size(0), -1)
      x = torch.relu(self.fc1(x))
      x = self.fc2(x)
      return x

# --- Управление агентом ---
class MinesweeperDQN:
   def __init__(self, env, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995, batch_size=128, memory_size=100000):
      self.env = env
      self.width = env.width
      self.height = env.height
      self.device = torch.device("cpu")
      self.gamma = gamma
      self.epsilon = epsilon
      self.epsilon_min = epsilon_min
      self.epsilon_decay = epsilon_decay
      self.batch_size = batch_size
      self.memory = deque(maxlen=memory_size)

      self.model = CNNAgent(width=self.width, height=self.height).to(self.device)
      self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
      self.loss_fn = nn.MSELoss()

   def get_action(self, state):
      if random.random() < self.epsilon:
         x = random.randint(0, self.width - 1)
         y = random.randint(0, self.height - 1)
         action_type = random.randint(0, 1)
         return (x, y, action_type)

      state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
      q_values = self.model(state_tensor).detach().cpu().numpy()[0]

      q_open = q_values[:self.width * self.height].reshape(self.height, self.width)
      q_flag = q_values[self.width * self.height:].reshape(self.height, self.width)

      idx = np.unravel_index(np.argmax(np.maximum(q_open, q_flag)), q_open.shape)
      x, y = idx[1], idx[0]
      action_type = 0 if q_open[y, x] >= q_flag[y, x] else 1

      return (x, y, action_type)

   def remember(self, state, action, reward, next_state, done):
      self.memory.append((state, action, reward, next_state, done))

   def replay(self):
      if len(self.memory) < self.batch_size:
         return

      batch = random.sample(self.memory, self.batch_size)
      states, actions, rewards, next_states, dones = zip(*batch)

      states = torch.FloatTensor(np.array(states)).unsqueeze(1).to(self.device)
      next_states = torch.FloatTensor(np.array(next_states)).unsqueeze(1).to(self.device)

      q_values = self.model(states)
      next_q_values = self.model(next_states)

      targets = q_values.clone()

      for i in range(self.batch_size):
         x, y, action_type = actions[i]
         idx = action_type * self.width * self.height + (y * self.width + x)
         if dones[i]:
            targets[i, idx] = rewards[i]
         else:
            targets[i, idx] = rewards[i] + self.gamma * torch.max(next_q_values[i])

      loss = self.loss_fn(q_values, targets)

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      if self.epsilon > self.epsilon_min:
         self.epsilon *= self.epsilon_decay

   def save(self, path="cnn_minesweeper.pth"):
      torch.save(self.model.state_dict(), path)

   def load(self, path):
      self.model.load_state_dict(torch.load(path, map_location=self.device))

# --- Игровой процесс для параллельной тренировки ---
def play_one_episode(dummy):
   env = MinesweeperEnv(width=30, height=16, n_mines=99)
   agent = MinesweeperDQN(env)
   state = env.reset()
   done = False
   trajectory = []

   while not done:
      action = agent.get_action(state)
      next_state, reward, done = env.step(action)
      trajectory.append((state, action, reward, next_state, done))
      state = next_state

   return trajectory

# --- Обучение с авто-чекпоинтами ---
def train_parallel(n_processes=4, n_episodes=10000, checkpoint_interval=1000):
   env = MinesweeperEnv(width=30, height=16, n_mines=99)
   agent = MinesweeperDQN(env)

   # --- Поиск последнего чекпоинта ---
   checkpoints = sorted(glob.glob("cnn_minesweeper_checkpoint_*.pth"))
   start_episode = 0
   if checkpoints:
      last_checkpoint = checkpoints[-1]
      print(f"Loading checkpoint {last_checkpoint}")
      agent.load(last_checkpoint)
      start_episode = int(last_checkpoint.split('_')[-1].split('.')[0])
      print(f"Resuming from episode {start_episode}")

   pool = Pool(processes=n_processes)

   for episode_batch in range(start_episode, n_episodes, n_processes):
      trajectories = pool.map(play_one_episode, [None] * n_processes)

      for trajectory in trajectories:
         for state, action, reward, next_state, done in trajectory:
            agent.remember(state, action, reward, next_state, done)

      agent.replay()

      if episode_batch % (100 * n_processes) == 0:
         print(f"Episode {episode_batch}, Buffer size: {len(agent.memory)}, Epsilon: {agent.epsilon:.4f}")

      if episode_batch > 0 and episode_batch % checkpoint_interval == 0:
         checkpoint_path = f"cnn_minesweeper_checkpoint_{episode_batch}.pth"
         agent.save(checkpoint_path)
         print(f"Checkpoint saved to {checkpoint_path}")

   agent.save("cnn_minesweeper.pth")
   print("Final model saved to cnn_minesweeper.pth")

   pool.close()
   pool.join()

# --- Тестирование ---
def test():
   env = MinesweeperEnv(width=30, height=16, n_mines=99)
   agent = MinesweeperDQN(env)
   agent.load("cnn_minesweeper.pth")

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

      pygame.time.delay(300)  # <-- Делаем задержку 300 мс между действиями

      action = agent.get_action(state)
      state, reward, done = env.step(action)

      if done:
         env.render()

         # Показать сообщение ПОБЕДА / ПРОИГРЫШ
         font = pygame.font.SysFont('Arial', 48)
         text = "ПОБЕДА!" if reward > 0 else "ПРОИГРЫШ"
         color = (0, 255, 0) if reward > 0 else (255, 0, 0)
         text_surface = font.render(text, True, color)
         text_rect = text_surface.get_rect(center=(env.width * env.cell_size // 2,
                                                    env.height * env.cell_size // 2))
         env.screen.blit(text_surface, text_rect)
         pygame.display.flip()

         pygame.time.delay(3000)  # Показать результат 3 секунды

         state = env.reset()

      clock.tick(30)

# --- Главный запуск ---
if __name__ == "__main__":
   mode = input("Enter 'train' to train, or 'test' to test the agent: ").strip()

   if mode == 'train':
      train_parallel()
   elif mode == 'test':
      test()
   else:
      print("Unknown mode.")