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
TAU = 0.01 # update target network parameters 
REPLAY_SIZE = 10000  # experience replay buffer size
BATCH_SIZE = 32  # size of mini-batch
ENV_NAME = 'Pendulum-v0'
EPISODE = 500  # episode limitation
STEP = 200  # step limitation in an episode
TEST = 10  # the number of experiment test every 100 episode
LR_A = 1e-3  # learning rate for training Actor network
LR_C = 2e-3  # learning rate for training Critic network
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------
# Actor-critic network
# ----------------------

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 20)
        self.fc2 = nn.Linear(20, action_dim)

    def forward(self, state):
        action = F.relu(self.fc1(state))
        action = torch.tanh(self.fc2(action)) * 2
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc11 = nn.Linear(state_dim, 20)
        self.fc12 = nn.Linear(action_dim, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, state, action):
        value = F.relu(self.fc11(state) + self.fc12(action))
        value = self.fc2(value)
        return value

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

        # Init neural network
        self.current_actor = Actor(self.state_dim, self.action_dim).to(DEVICE)
        self.current_critic = Critic(self.state_dim, self.action_dim).to(DEVICE)
        self.target_actor = Actor(self.state_dim, self.action_dim).to(DEVICE)
        self.target_critic = Critic(self.state_dim, self.action_dim).to(DEVICE)

        self.target_actor.load_state_dict(self.current_actor.state_dict())
        self.target_critic.load_state_dict(self.current_critic.state_dict())

        self.actor_optim = torch.optim.Adam(self.current_actor.parameters(), lr=LR_A)
        self.critic_optim = torch.optim.Adam(self.current_critic.parameters(), lr=LR_C)
        
        self.var = 3 # variance of exploration noise

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float, device=DEVICE)
            action = self.current_actor(state_tensor).cpu().detach().numpy() # select an action by Q network
            action = np.clip(np.random.normal(action, self.var), -2, 2) # add exploration noise
        return action

    def update(self):
        if len(self.replay_memory) < REPLAY_SIZE:
            return

        # synchronize target net to current net
        self.synchronize_networks()

        # Sample mini batch
        transitions = self.replay_memory.sample(BATCH_SIZE)

        states = torch.as_tensor(np.array([data[0] for data in transitions]), dtype=torch.float, device=DEVICE)
        actions = torch.as_tensor(np.array([data[1] for data in transitions]), dtype=torch.float, device=DEVICE)
        rewards = torch.as_tensor(np.array([data[2] for data in transitions]), dtype=torch.float, device=DEVICE).view(BATCH_SIZE, 1)
        next_states = torch.as_tensor(np.array([data[3] for data in transitions]), dtype=torch.float, device=DEVICE)
        dones = torch.as_tensor(np.array([data[4] for data in transitions]), device=DEVICE).view(BATCH_SIZE, 1)
        
        # -------------------------
        # Evaluate critic
        # -------------------------
        # Compute q value
        q_value = self.current_critic(states, actions)
        next_actions = self.target_actor(next_states)
        next_q_value = self.target_critic(next_states, next_actions).detach() * (~dones)
        expected_q_value = GAMMA * next_q_value + rewards

        # Compute square error loss
        td_error = expected_q_value - q_value
        critic_loss = torch.sum(torch.square(td_error))

        # -------------------------
        # Optimize Critic parameters
        # -------------------------
        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.current_critic.parameters(), 1)
        self.critic_optim.step()

        # -------------------------
        # Evaluate actor
        # -------------------------
        est_actions = self.current_actor(states)
        actor_loss = -torch.mean(self.current_critic(states, est_actions))

        # -------------------------
        # Optimize Actor parameters
        # -------------------------
        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.current_actor.parameters(), 1)
        self.actor_optim.step()
    
    def synchronize_networks(self):
        for curr, target in zip(self.current_actor.parameters(), self.target_actor.parameters()):
            target.data.copy_((1 - TAU) * target.data + TAU * curr.data)
            
        for curr, target in zip(self.current_critic.parameters(), self.target_critic.parameters()):
            target.data.copy_((1 - TAU) * target.data + TAU * curr.data)


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
            
            if len(agent.replay_memory) == REPLAY_SIZE:
                agent.var *= 0.9995
            
            agent.update()
            state = next_state
            step += 1
        
        # ---------------------------
        # Test every 10 episodes
        # ---------------------------

        agent.current_actor.eval()
        agent.current_critic.eval()
        if episode % 10 == 0:
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
            print('Episode:', episode, 'Evaluation average reward:', ave_reward, 'Var:', agent.var)

    print('Time cost: ', time.time() - start_time)


if __name__ == '__main__':
    main()
