import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from collections import deque


# ------------------------
# Hyper parameter for DQN
# ------------------------

GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON = 0.5  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 10000  # experience replay buffer size
BATCH_SIZE = 32  # size of mini-batch
ENV_NAME = 'CartPole-v0'
EPISODE = 3000  # episode limitation
STEP = 300  # step limitation in an episode
TEST = 10  # the number of experiment test every 100 episode
LR = 1e-4  # learning rate for training DQN
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -------------
# Deep Q-network
# -------------

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 20)
        self.fc2 = nn.Linear(20, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# --------------
# Replay memory
# --------------

class SumTree:

    def __init__(self, capacity):
        self.write = 0
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.entry_num = 0

    # Update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    # Find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # Update priority
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    # Store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.entry_num < self.capacity:
            self.entry_num += 1

    # Get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


# --------------
# Replay memory
# --------------

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = SumTree(capacity)
        self.max_error = 1

    def push(self, state, action, reward, next_state, done):
        """Save a transition"""
        data = (state, action, reward, next_state, done)
        self.tree.add(self.max_error, data)  # default error = 1

    def sample(self, batch_size):
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            idxs.append(idx)
            priorities.append(p)
            batch.append(data)
        return idxs, batch

    def update(self, idx, error):
        p = (np.abs(error) + 0.01) ** 0.6
        self.tree.update(idx, p)

    def __len__(self):
        return self.tree.entry_num


# -------
# Agent
# -------

class Agent:
    def __init__(self, env):
        self.replay_memory = ReplayMemory(REPLAY_SIZE)  # init experience replay

        # Init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        # Init neural network
        self.DQN = DQN(self.state_dim, self.action_dim).to(DEVICE)
        self.optimizer = torch.optim.AdamW(self.DQN.parameters(), lr=LR)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.action_dim)  # select an action randomly
        else:
            with torch.no_grad():
                state_tensor = torch.as_tensor(state, dtype=torch.float, device=DEVICE)
                q_value = self.DQN.forward(state_tensor)
                action = torch.argmax(q_value).detach().item()  # select an action by Q network

        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000  # epsilon decay
        return action

    def update(self):
        if len(self.replay_memory) < REPLAY_SIZE:
            return

        # Sample mini batch
        tree_idxs, transitions = self.replay_memory.sample(BATCH_SIZE)

        state_batch = torch.as_tensor([data[0] for data in transitions], dtype=torch.float, device=DEVICE)
        action_batch = torch.as_tensor([data[1] for data in transitions], device=DEVICE).view(BATCH_SIZE, 1)
        reward_batch = torch.as_tensor([data[2] for data in transitions], device=DEVICE)
        next_state_batch = torch.as_tensor([data[3] for data in transitions], dtype=torch.float, device=DEVICE)
        done_batch = torch.as_tensor([data[4] for data in transitions], device=DEVICE)

        # Compute q value
        q_value = self.DQN.forward(state_batch).gather(1, action_batch)
        next_q_value = self.DQN.forward(next_state_batch).max(1)[0].detach() * (~done_batch)
        expected_q_value = GAMMA * next_q_value + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(q_value, expected_q_value.unsqueeze(1))

        # Update replay memory
        error = (expected_q_value.unsqueeze(1) - q_value).cpu().detach().squeeze().numpy()
        for i in range(BATCH_SIZE):
            self.replay_memory.update(tree_idxs[i], error[i])

        # Optimize model parameters
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.DQN.parameters(), 1)
        self.optimizer.step()

        return loss


def main():
    # Init Open AI Gym env and dqn agent
    env = gym.make(ENV_NAME)
    agent = Agent(env)

    start_time = time.time()
    for episode in range(EPISODE):
        state = env.reset()  # init state

        # ------
        # Train
        # ------

        done = False
        step = 0
        while not done and step < STEP:
            action = agent.select_action(state)
            next_state, reward_, done, _ = env.step(action)
            reward = -1 if done else 0.1
            agent.replay_memory.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            step += 1

        # ---------------------------
        # Test every 100 episodes
        # ---------------------------

        agent.DQN.eval()
        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                done = False
                step = 0
                while not done and step < STEP:
                    with torch.no_grad():
                        action = agent.select_action(state)
                    next_state, reward, done, _ = env.step(action)
                    total_reward += reward
                    state = next_state
                    step += 1

            ave_reward = total_reward / TEST
            print('Episode:', episode, 'Evaluation average reward:', ave_reward)

    print('Time cost: ', time.time() - start_time)


if __name__ == '__main__':
    main()
