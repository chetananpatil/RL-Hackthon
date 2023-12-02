import gym
import numpy as np
from gym import spaces


class CustomFrozenLake(gym.Env):
    def __init__(self, nrow=8, ncol=8, num_holes=8):
        self.nrow = nrow
        self.ncol = ncol
        self.n_state = nrow * ncol
        self.num_holes = num_holes

        # Initialize the grid with no holes
        self.desc = np.full((nrow, ncol), b'F', dtype='c')

        # Randomly place holes on the grid
        hole_positions = np.random.choice(self.n_state, size=num_holes, replace=False)
        for pos in hole_positions:
            row, col = divmod(pos, ncol)
            self.desc[row, col] = b'H'

        # Start from the top-left corner
        self.s = 0

        # Define actions (Left, Down, Right, Up)
        self.action_space = spaces.Discrete(4)

        # Define states
        self.observation_space = spaces.Discrete(self.n_state)

    def reset(self):
        # Reset to the starting state
        self.s = 0
        return self.s

    def step(self, a):
        i, j = divmod(self.s, self.ncol)

        if a == 0:  # Left
            j = max(0, j - 1)
        elif a == 1:  # Down
            i = min(self.nrow - 1, i + 1)
        elif a == 2:  # Right
            j = min(self.ncol - 1, j + 1)
        elif a == 3:  # Up
            i = max(0, i - 1)

        new_s = i * self.ncol + j
        self.s = new_s

        # Define rewards and termination conditions
        if self.desc[i, j] == b'G':
            return new_s, 1, True, {}
        elif self.desc[i, j] == b'H':
            return new_s, -1, True, {}
        else:
            return new_s, 0, False, {}


class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_space = env.action_space.n
        self.state_space = env.observation_space.n
        self.q_table = np.zeros((self.state_space, self.action_space))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Exploration: choose random action
        else:
            return np.argmax(self.q_table[state, :])  # Exploitation: choose action with max Q-value

    def update_q_table(self, state, action, reward, new_state):
        max_next_action = np.max(self.q_table[new_state, :])
        self.q_table[state, action] = (1 - self.learning_rate) * self.q_table[state, action] + \
                                      self.learning_rate * (reward + self.discount_factor * max_next_action)

    def train(self, num_episodes=1000, max_steps_per_episode=100):
        rewards = []
        paths = []
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            path = [state]
            for step in range(max_steps_per_episode):
                action = self.choose_action(state)
                new_state, reward, done, _ = self.env.step(action)
                self.update_q_table(state, action, reward, new_state)
                total_reward += reward
                #
                path.append(new_state)
                state = new_state
                if done:
                    break
            rewards.append(total_reward)
            #
            paths.append(path)
            # Decay epsilon for exploration-exploitation trade-off
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
        return rewards, paths

# Create the custom FrozenLake environment
env = CustomFrozenLake()

# Create Q-learning agent and train it in the environment
agent = QLearningAgent(env)
episodes = 1000
rewards_per_episode = agent.train(num_episodes=episodes)
#
rewards_per_episode, paths = agent.train(num_episodes=episodes)
# Print average rewards per episode
print(f"Average rewards over {episodes} episodes: {np.mean(rewards_per_episode)}")

# Print paths for the first few episodes
for i in range(min(5, episodes)):
    print(f"Episode {i + 1} Path: {paths[i]}")