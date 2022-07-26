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
TAU = 0.1 # update target network parameters 
REPLAY_SIZE = 3000  # experience replay buffer size
BATCH_SIZE = 32  # size of mini-batch
ENV_NAME = 'Pendulum-v1'
EPISODE = 10000  # episode limitation
STEP = 300  # step limitation in an episode
TEST = 10  # the number of experiment test every 100 episode
LR = 1e-4  # learning rate for training DQN
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGET_SYNC = 10  # Synchronize the target net parameter every 10 episodes


# ----------------------
# Actor-critic network
# ----------------------

class ACNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ACNet, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 20),
            nn.ReLU(),
            nn.Linear(20, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        raise NotImplementedError


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
        self.action_dim = env.action_space.shape[0]
        self.a_bound = env.action_space.high

        # Init neural network
        self.current_net = ACNet(self.state_dim, self.action_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.current_net.parameters(), lr=LR)

        self.target_net = ACNet(self.state_dim, self.action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.current_net.state_dict())

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float, device=DEVICE)
            action = self.current_net.actor(state_tensor).cpu().detach().numpy() # select an action by Q network
            action = action * self.a_bound + np.random.randn(self.action_dim) # add exploration noise
        return action

    def update(self):
        if len(self.replay_memory) < REPLAY_SIZE:
            return

        # Sample mini batch
        transitions = self.replay_memory.sample(BATCH_SIZE)

        state_batch = torch.as_tensor([data[0] for data in transitions], dtype=torch.float, device=DEVICE)
        action_batch = torch.as_tensor([data[1] for data in transitions], device=DEVICE)
        reward_batch = torch.as_tensor([data[2] for data in transitions], device=DEVICE).view(BATCH_SIZE, 1)
        next_state_batch = torch.as_tensor([data[3] for data in transitions], dtype=torch.float, device=DEVICE)
        done_batch = torch.as_tensor([data[4] for data in transitions], device=DEVICE).view(BATCH_SIZE, 1)
        state_action_batch = torch.as_tensor([np.concatenate([data[0], data[1]]) for data in transitions], dtype=torch.float, device=DEVICE)

        # -------------------------
        # Evaluate critic
        # -------------------------

        # Compute q value
        q_value = self.current_net.critic(state_action_batch)
        next_action = self.target_net.actor(next_state_batch)
        next_state_action_batch = torch.as_tensor([np.concatenate([s.detach().numpy(), a.detach().numpy()]) for s, a in zip(next_state_batch, next_action)], dtype=torch.float, device=DEVICE)
        next_q_value = self.target_net.critic(next_state_action_batch).detach() * (~done_batch)
        expected_q_value = GAMMA * next_q_value + reward_batch

        # Compute square error loss
        td_error = expected_q_value - q_value
        critic_loss = torch.sum(torch.square(td_error))

        # -------------------------
        # Evaluate actor
        # -------------------------

        action = self.current_net.actor(state_batch)
        state_action_batch = torch.as_tensor([np.concatenate([s.detach().numpy(), a.detach().numpy()]) for s, a in zip(state_batch, action)], dtype=torch.float, device=DEVICE)
        # Cross entropy returns negative log likelihood loss
        actor_loss = -torch.mean(self.current_net.critic(state_action_batch))

        total_loss = critic_loss + actor_loss

        # -------------------------
        # Optimize ACNet parameters
        # -------------------------

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.current_net.parameters(), 1)
        self.optimizer.step()
    
    def synchronize_networks(self):
        for curr, target in zip(self.current_net.parameters(), self.target_net.parameters()):
            target = (1 - TAU) * target + TAU * curr


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
            next_state, reward, done, _ = env.step(action)
            # reward = -1 if done else 0.1
            agent.replay_memory.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            step += 1

        # synchronize target net to current net
        if episode % TARGET_SYNC == 0:
            agent.synchronize_networks()
        
        # ---------------------------
        # Test every 100 episodes
        # ---------------------------

        agent.current_net.eval()
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
