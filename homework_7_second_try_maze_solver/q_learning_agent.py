import numpy as np
import pygame
from maze_env import MazeEnv
from tqdm import trange
import os
import matplotlib.pyplot as plt

MAZE_PATH = "saved/maze_21x21.npy"
QTABLE_PATH = "saved/q_table_21x21.npy"

# Параметры обучения
EPISODES = 5000
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.98
MIN_EPSILON = 0.05

SHOW_EVERY = 500  # каждые N эпизодов показывать агента


def train_agent():
   env = MazeEnv(MAZE_PATH)
   state_shape = (env.width, env.height, 4)  # Q[x][y][action]
   q_table = np.zeros(state_shape)

   global EPSILON
   rewards = []

   for episode in trange(EPISODES, desc="Training"):
      state = env.reset()
      done = False
      visited = set()
      last_action = None
      total_reward = 0
      max_steps = env.width * env.height * 2
      steps = 0

      while not done and steps < max_steps:
         x, y = state
         visited.add(state)
         steps += 1

         if np.random.rand() < EPSILON:
            possible_actions = list(range(4))
            best_actions = []

            # Проверка направлений, чтобы не возвращаться без нужды
            for action in possible_actions:
               dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
               nx, ny = x + dx, y + dy
               if 0 <= nx < env.width and 0 <= ny < env.height:
                  if env.maze[ny, nx] == 0 and (nx, ny) not in visited:
                     best_actions.append(action)

            if best_actions:
               action = np.random.choice(best_actions)
            else:
               action = (last_action + 2) % 4 if last_action is not None else np.random.choice(possible_actions)
         else:
            action = np.argmax(q_table[x, y])

         next_state, reward, done = env.step(action)
         total_reward += reward
         nx, ny = next_state

         max_future_q = np.max(q_table[nx, ny])
         current_q = q_table[x, y, action]

         new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
         q_table[x, y, action] = new_q

         state = next_state
         last_action = action

         if episode % SHOW_EVERY == 0:
            env.render(delay=5)
            env.handle_events()

      EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
      rewards.append(total_reward)

   np.save(QTABLE_PATH, q_table)
   print(f"\nTraining complete. Q-table saved to {QTABLE_PATH}.")
   env.close()

   # Построим график наград
   plt.plot(rewards)
   plt.title("Total reward per episode")
   plt.xlabel("Episode")
   plt.ylabel("Total reward")
   plt.grid(True)
   plt.tight_layout()
   plt.savefig("saved/training_rewards_21x21.png")
   plt.show()


if __name__ == '__main__':
   train_agent()