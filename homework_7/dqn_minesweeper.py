# dqn_minesweeper.py

import numpy as np
import pygame
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from minesweeper import MinesweeperEnv

# --- DQN модель ---
class DQN(nn.Module):
   def __init__(self, input_size, output_size):
      super(DQN, self).__init__()
      self.fc1 = nn.Linear(input_size, 128)
      self.fc2 = nn.Linear(128, 128)
      self.fc3 = nn.Linear(128, output_size)

   def forward(self, x):
      x = torch.relu(self.fc1(x))
      x = torch.relu(self.fc2(x))
      x = self.fc3(x)
      return x

# --- DQN Агент ---
class DQNAgent:
   def __init__(self, env, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995, batch_size=64, memory_size=10000):
      self.env = env
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.gamma = gamma
      self.epsilon = epsilon
      self.epsilon_min = epsilon_min
      self.epsilon_decay = epsilon_decay
      self.batch_size = batch_size
      self.memory = deque(maxlen=memory_size)

      input_size = env.width * env.height
      output_size = env.width * env.height * 2

      self.model = DQN(input_size, output_size).to(self.device)
      self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
      self.loss_fn = nn.MSELoss()

   def get_action(self, state):
      if random.random() < self.epsilon:
         x = random.randint(0, self.env.width - 1)
         y = random.randint(0, self.env.height - 1)
         action_type = random.randint(0, 1)
         return (x, y, action_type)

      state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0).to(self.device)
      q_values = self.model(state_tensor)
      best_action = torch.argmax(q_values).item()

      x = (best_action // 2) % self.env.width
      y = (best_action // 2) // self.env.width
      action_type = best_action % 2
      return (x, y, action_type)

   def remember(self, state, action, reward, next_state, done):
      self.memory.append((state, action, reward, next_state, done))

   def replay(self):
      if len(self.memory) < self.batch_size:
         return

      batch = random.sample(self.memory, self.batch_size)
      states, actions, rewards, next_states, dones = zip(*batch)

      states = torch.FloatTensor([s.flatten() for s in states]).to(self.device)
      next_states = torch.FloatTensor([s.flatten() for s in next_states]).to(self.device)

      q_values = self.model(states)
      q_next = self.model(next_states)

      targets = q_values.clone()

      for i in range(self.batch_size):
         x, y, action_type = actions[i]
         idx = (y * self.env.width + x) * 2 + action_type
         if dones[i]:
            targets[i, idx] = rewards[i]
         else:
            targets[i, idx] = rewards[i] + self.gamma * torch.max(q_next[i])

      loss = self.loss_fn(q_values, targets)

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      if self.epsilon > self.epsilon_min:
         self.epsilon *= self.epsilon_decay

   def save(self, path="dqn_minesweeper.pth"):
      torch.save(self.model.state_dict(), path)

   def load(self, path="dqn_minesweeper.pth"):
      self.model.load_state_dict(torch.load(path, map_location=self.device))


# --- Обучение ---
def train():
   env = MinesweeperEnv(width=9, height=9, n_mines=10)
   agent = DQNAgent(env)

   n_episodes = 5000

   for episode in range(n_episodes):
      state = env.reset()
      total_reward = 0
      done = False

      while not done:
         action = agent.get_action(state)
         next_state, reward, done = env.step(action)
         agent.remember(state, action, reward, next_state, done)
         agent.replay()
         state = next_state
         total_reward += reward

      if episode % 100 == 0:
         print(f"Episode {episode}, Total reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

   agent.save("dqn_minesweeper.pth")
   print("Model saved to dqn_minesweeper.pth")

# --- Тестирование ---
def test():
   env = MinesweeperEnv(width=9, height=9, n_mines=10)
   agent = DQNAgent(env)
   agent.load("dqn_minesweeper.pth")

   obs = env.reset()

   clock = pygame.time.Clock()

   while True:
      env.render()
      action = agent.get_action(obs)
      obs, reward, done = env.step(action)

      if done:
         pygame.time.delay(2000)
         obs = env.reset()

      clock.tick(30)

# --- Главный запуск ---
if __name__ == "__main__":
   mode = input("Enter 'train' to train, or 'test' to test the agent: ").strip()

   if mode == 'train':
      train()
   elif mode == 'test':
      test()
   else:
      print("Unknown mode.")
