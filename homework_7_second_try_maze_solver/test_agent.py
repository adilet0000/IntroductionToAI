import numpy as np
import time
from maze_env import MazeEnv

MAZE_PATH = "saved/maze_51x51.npy"
QTABLE_PATH = "saved/q_table_51x51.npy"

DELAY = 10  # миллисекунд между шагами


def test_agent():
   q_table = np.load(QTABLE_PATH)
   env = MazeEnv(MAZE_PATH)
   state = env.reset()
   done = False

   steps = 0
   max_steps = env.width * env.height * 2

   while not done and steps < max_steps:
      env.render(delay=DELAY)
      env.handle_events()

      x, y = state
      action = np.argmax(q_table[x, y])

      next_state, reward, done = env.step(action)
      state = next_state
      steps += 1

   print("\nAgent has reached the goal!" if done else "\nAgent failed to reach the goal.")
   time.sleep(2)
   env.close()


if __name__ == '__main__':
   test_agent()