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

GAMMA = 0.95  # discount factor for target Q
REPLAY_SIZE = 10000  # experience replay buffer size
BATCH_SIZE = 32  # size of mini-batch
ENV_NAME = 'CartPole-v0'
EPISODE = 10000  # episode limitation
STEP = 300  # step limitation in an episode
TEST = 10  # the number of experiment test every 100 episode
LR = 1e-4  # learning rate for training DQN
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ----------------------
# Actor-critic network
# ----------------------

class ACNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ACNet, self).__init__()
        self.afc1 = nn.Linear(state_dim, 20)  # fully connected layer 1 of actor net
        self.afc2 = nn.Linear(20, action_dim)

        self.cfc1 = nn.Linear(state_dim, 20)  # fully connected layer 1 of critic net
        self.cfc2 = nn.Linear(20, action_dim)

    def forward(self, x):
        pi = F.relu(self.afc1(x))
        pi = self.afc2(pi)

        value = F.relu(self.cfc1(x))
        value = self.cfc2(value)

        return pi, value


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
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        # Init neural network
        self.ACNet = ACNet(self.state_dim, self.action_dim).to(DEVICE)
        self.optimizer = torch.optim.AdamW(self.ACNet.parameters(), lr=LR)

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float, device=DEVICE)
            action_prob = F.softmax(self.ACNet.forward(state_tensor)[0], dim=-1).cpu().detach().numpy()
            action = np.random.choice(range(self.action_dim), p=action_prob)  # select an action by Q network
        return action

    def update(self):
        if len(self.replay_memory) < REPLAY_SIZE:
            return

        # Sample mini batch
        transitions = self.replay_memory.sample(BATCH_SIZE)

        state_batch = torch.as_tensor([data[0] for data in transitions], dtype=torch.float, device=DEVICE)
        action_batch = torch.as_tensor([data[1] for data in transitions], device=DEVICE)
        reward_batch = torch.as_tensor([data[2] for data in transitions], device=DEVICE)
        next_state_batch = torch.as_tensor([data[3] for data in transitions], dtype=torch.float, device=DEVICE)
        done_batch = torch.as_tensor([data[4] for data in transitions], device=DEVICE)

        # -------------------------
        # Evaluate critic
        # -------------------------

        # Compute q value
        q_value = self.ACNet.forward(state_batch)[1].gather(1, action_batch.view(BATCH_SIZE, 1))
        next_q_value = self.ACNet.forward(next_state_batch)[1].max(1)[0].detach() * (~done_batch)
        expected_q_value = GAMMA * next_q_value + reward_batch

        # Compute Huber loss
        td_error = F.smooth_l1_loss(input=q_value, target=expected_q_value.unsqueeze(1), reduction='none')
        critic_loss = torch.sum(td_error)

        # -------------------------
        # Evaluate actor
        # -------------------------

        action_prob = F.softmax(self.ACNet(state_batch)[0], dim=-1)
        actor_loss = F.cross_entropy(action_prob, action_batch, reduction='none')
        actor_loss = -torch.sum(actor_loss * td_error.detach())

        total_loss = critic_loss + actor_loss

        # -------------------------
        # Optimize ACNet parameters
        # -------------------------

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.ACNet.parameters(), 1)
        self.optimizer.step()


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

        agent.ACNet.eval()
        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                done = False
                step = 0
                while not done and step < STEP:
                    # env.render()
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
