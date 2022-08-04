import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import random
import time
from collections import deque

# ------------------------
# Hyper parameter for DQN
# ------------------------

GAMMA = 0.9  # discount factor for target Q
TAU = 0.005 # update target network parameters
ALPHA = 0.2 # initial value of alpha
REPLAY_SIZE = 10000  # experience replay buffer size
BATCH_SIZE = 32  # size of mini-batch
ENV_NAME = 'Pendulum-v0'
ENTROPY_BOUND = 0.1 # the lower bound of action entropy
EPISODE = 5000  # episode limitation
STEP = 200  # step limitation in an episode
TEST = 10  # the number of experiment test every 100 episode
LR_A = 1e-3  # learning rate for training Actor network
LR_C = 5e-4  # learning rate for training Critic network
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------
# Actor-critic network
# ----------------------

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.hidden_fc = nn.Linear(state_dim, 20)
        self.mean_fc = nn.Linear(20, action_dim)
        self.log_std_fc = nn.Linear(20, action_dim)

    def forward(self, state):
        x = F.relu(self.hidden_fc(state))
        mean = self.mean_fc(x)
        std =  torch.exp(self.log_std_fc(x))
        return mean, std
    
    def sample(self, state):
        mean, std = self.forward(state)
        normal = Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x) * 2
        log_pi = normal.log_prob(x).sum(axis=-1, keepdim=True)
        return action, log_pi


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 20)
        self.fc2 = nn.Linear(20, 1)

        self.fc3 = nn.Linear(state_dim + action_dim, 20)
        self.fc4 = nn.Linear(20, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        v1 = F.relu(self.fc1(sa))
        v1 = self.fc2(v1)

        v2 = F.relu(self.fc3(sa))
        v2 = self.fc4(v2)
        return v1, v2

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
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.alpha = ALPHA # temperature parameter

        # Init neural network
        self.actor = Actor(self.state_dim, self.action_dim).to(DEVICE)
        self.critic = Critic(self.state_dim, self.action_dim).to(DEVICE)

        self.target_critic = Critic(self.state_dim, self.action_dim).to(DEVICE)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=LR_A)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=LR_C)
        

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float, device=DEVICE)
            action = self.actor.sample(state_tensor)[0].cpu().detach().numpy() # select an action by Q network
        return action

    def update(self):
        if len(self.replay_memory) < REPLAY_SIZE:
            return

        # Synchronize target net to current net
        for curr, target in zip(self.critic.parameters(), self.target_critic.parameters()):
            target.data.copy_((1 - TAU) * target.data + TAU * curr.data)

        # Sample mini batch
        transitions = self.replay_memory.sample(BATCH_SIZE)
        s = torch.as_tensor(np.array([data[0] for data in transitions]), dtype=torch.float, device=DEVICE)
        a = torch.as_tensor(np.array([data[1] for data in transitions]), dtype=torch.float, device=DEVICE)
        r = torch.as_tensor(np.array([data[2] for data in transitions]), dtype=torch.float, device=DEVICE).view(BATCH_SIZE, 1)
        s_ = torch.as_tensor(np.array([data[3] for data in transitions]), dtype=torch.float, device=DEVICE)
        dones = torch.as_tensor(np.array([data[4] for data in transitions]), device=DEVICE).view(BATCH_SIZE, 1)
        
        # -------------------------
        # Evaluate critic
        # -------------------------
        # Compute q value
        q1, q2 = self.critic(s, a)
        
        with torch.no_grad():
            a_, log_pi_ = self.actor.sample(s_)
            q1_, q2_ = self.target_critic(s_, a_)
            q_ = torch.min(q1_, q2_) * (~dones)

            # Compute square error loss
            entropy = -self.alpha * log_pi_
            expected_q = GAMMA * q_ + r + entropy

        critic_loss = F.mse_loss(q1, expected_q) + F.mse_loss(q2, expected_q)

        # -------------------------
        # Optimize Critic parameters
        # -------------------------
        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optim.step()

        # -------------------------
        # Evaluate actor
        # -------------------------
        est_a, est_log_pi = self.actor.sample(s)
        est_q1, est_q2 = self.critic(s, est_a)
        actor_loss = torch.mean(self.alpha * est_log_pi - torch.min(est_q1, est_q2) )

        # -------------------------
        # Optimize Actor parameters
        # -------------------------
        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_optim.step()

        # -------------------------
        # Optimize temperature parameter alpha
        # -------------------------
        self.alpha = self.alpha - LR_C * (-est_log_pi.detach() - ENTROPY_BOUND)


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
            agent.replay_memory.push(state, action, reward / 10, next_state, done)
            agent.update()
            state = next_state
            step += 1
        
        # ---------------------------
        # Test every 20 episodes
        # ---------------------------

        agent.actor.eval()
        agent.critic.eval()
        if episode % 20 == 0:
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
