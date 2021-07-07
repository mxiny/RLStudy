import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
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
EPISODE = 5000  # episode limitation
STEP = 300  # step limitation in an episode
TEST = 10  # the number of experiment test every 100 episodes
LR = 1e-4  # learning rate for training DQN
TARGET_SYNC = 10  # Synchronize the target net parameter every 10 episodes
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

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Save a transition"""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


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
        self.policy_net = DQN(self.state_dim, self.action_dim).to(DEVICE)
        self.target_net = DQN(self.state_dim, self.action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=LR)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.action_dim)  # select an action randomly
        else:
            with torch.no_grad():
                state_tensor = torch.as_tensor(state, dtype=torch.float, device=DEVICE)
                q_value = self.policy_net.forward(state_tensor)
                action = torch.argmax(q_value).detach().item()  # select an action by Q network

        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000  # epsilon decay
        return action

    def update(self):
        if len(self.replay_memory) < REPLAY_SIZE:
            return

        # Sample mini batch
        transitions = self.replay_memory.sample(BATCH_SIZE)

        state_batch = torch.as_tensor([data[0] for data in transitions], dtype=torch.float, device=DEVICE)
        action_batch = torch.as_tensor([data[1] for data in transitions], device=DEVICE).view(BATCH_SIZE, 1)
        reward_batch = torch.as_tensor([data[2] for data in transitions], device=DEVICE).view(BATCH_SIZE, 1)
        next_state_batch = torch.as_tensor([data[3] for data in transitions], dtype=torch.float, device=DEVICE)
        done_batch = torch.as_tensor([data[4] for data in transitions], device=DEVICE).view(BATCH_SIZE, 1)

        # Compute q value
        q_value = self.policy_net.forward(state_batch).gather(1, action_batch)
        next_action = self.policy_net.forward(next_state_batch).argmax(axis=1).view(BATCH_SIZE, 1)
        next_q_value = self.target_net.forward(next_state_batch).gather(1, next_action).detach() * (~done_batch)
        expected_q_value = GAMMA * next_q_value + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(q_value, expected_q_value)

        # Optimize model parameters
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
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

        # synchronize target net to policy net
        if episode % TARGET_SYNC == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        # ---------------------------
        # Test every 100 episodes
        # ---------------------------

        agent.policy_net.eval()
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
