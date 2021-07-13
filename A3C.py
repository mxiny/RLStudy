import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import random
import time
from collections import deque


# ------------------------
# Hyper parameter for A3C
# ------------------------

GAMMA = 0.95  # discount factor for target Q
REPLAY_SIZE = 10000  # experience replay buffer size
BATCH_SIZE = 32  # size of mini-batch
ENV_NAME = 'CartPole-v0'
STEP = 300  # step limitation in an episode
TEST = 10  # the number of experiment test every 100 episodes
LR = 1e-4  # learning rate for training DQN
GLOBAL_EPISODE = 3000  # episode limitation
GLOBAL_BETA = 0.01
GLOBAL_UPDATE_ITER = 100  # update global agent every 100 episodes
WORKER_NUM = 3  # number of workers
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -------------
# Deep Q-network
# -------------

class Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Net, self).__init__()
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
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        # Init neural network
        self.actor_net = Net(self.state_dim, self.action_dim).to(DEVICE)
        self.critic_net = Net(self.state_dim, self.action_dim).to(DEVICE)
        self.actor_opt = torch.optim.AdamW(self.actor_net.parameters(), lr=LR)
        self.critic_opt = torch.optim.AdamW(self.critic_net.parameters(), lr=LR)

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float, device=DEVICE)
            action_prob = F.softmax(self.actor_net.forward(state_tensor), dim=-1).cpu().detach().numpy()
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
        # Update critic network
        # -------------------------

        # Compute q value
        q_value = self.critic_net.forward(state_batch).gather(1, action_batch.view(BATCH_SIZE, 1))
        next_q_value = self.critic_net.forward(next_state_batch).max(1)[0].detach() * (~done_batch)
        expected_q_value = GAMMA * next_q_value + reward_batch

        # Compute Huber loss
        td_error = F.smooth_l1_loss(input=q_value, target=expected_q_value.unsqueeze(1), reduction='none')
        critic_loss = torch.sum(td_error)

        # Optimize model parameters
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_net.parameters(), 1)
        self.critic_opt.step()

        # -------------------------
        # Update actor network
        # -------------------------

        action_prob = F.softmax(self.actor_net(state_batch), dim=-1)
        actor_loss = F.cross_entropy(action_prob, action_batch, reduction='none')
        actor_loss = -torch.sum(actor_loss * td_error.detach())

        # Optimize model parameters
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_net.parameters(), 1)
        self.actor_opt.step()


class Worker(mp.Process):
    def __init__(self, name, global_agent):
        super(Worker, self).__init__()
        self.name = name

    def run(self):
        env = gym.make(ENV_NAME).unwrapped
        local_agent = Agent(env)

        state = env.reset()  # init state
        done = False
        step = 0
        while not done and step < STEP:
            action = local_agent.select_action(state)
            next_state, reward_, done, _ = env.step(action)
            reward = -1 if done else 0.1
            local_agent.replay_memory.push(state, action, reward, next_state, done)
            local_agent.update()
            state = next_state
            step += 1


def main():
    # Init Open AI Gym env and dqn agent
    env = gym.make(ENV_NAME)
    global_agent = Agent(env)

    start_time = time.time()
    for episode in range(GLOBAL_EPISODE):
        # ------------------
        # Train on workers
        # ------------------

        # Create workers
        workers = []
        for i in range(WORKER_NUM):
            worker_name = 'worker %i' % i
            workers.append(Worker(worker_name, global_agent))

        # Start the job
        for worker in workers:
            worker.start()

        # Wait until finishing all the job
        for worker in workers:
            worker.join()

        # ---------------------------
        # Test on global agent
        # ---------------------------

        global_agent.actor_net.eval()
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            done = False
            step = 0
            while not done and step < STEP:
                # env.render()
                with torch.no_grad():
                    action = global_agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state
                step += 1

        ave_reward = total_reward / TEST
        print('Episode:', episode, 'Evaluation average reward:', ave_reward)

    print('Time cost: ', time.time() - start_time)


if __name__ == '__main__':
    main()
