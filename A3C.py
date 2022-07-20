import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import time
from collections import deque


# ------------------------
# Hyper parameter for A3C
# ------------------------

GAMMA = 0.95  # discount factor for target Q
ENV_NAME = 'CartPole-v0'
MAX_STEP = 3000  # step limitation in an episode
TEST = 10  # the number of experiment test every 50 episodes
LR = 1e-3  # learning rate for training A3C
GLOBAL_EPISODE = 3000  # episode limitation
GLOBAL_UPDATE_STEP = 300  # update global agent every 100 episodes
ENTROPY_BETA = 0.001
WORKER_NUM = 3  # number of workers
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')

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
            nn.Linear(state_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        raise NotImplementedError


# --------------
# Replay memory
# --------------

class ReplayMemory(object):

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def push(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def pop_all(self):
        return self.states, self.actions, self.rewards

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()

    def __len__(self):
        return len(self.states)

# -------
# Agent
# -------

class Agent:
    def __init__(self, env):
        self.replay_memory = ReplayMemory()  # init experience replay

        # Init some parameters
        self.time_step = 0
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        # Init neural network
        self.ACNet = ACNet(self.state_dim, self.action_dim).to(DEVICE)

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float, device=DEVICE)
            action_prob = self.ACNet.actor(state_tensor).cpu().detach().numpy()
            action = np.random.choice(range(self.action_dim), p=action_prob)  # select an action by Q network
        return action

    def update(self, global_agent, global_opt, done):
        if len(self.replay_memory) == 0:
            return
        
        states, actions, rewards = self.replay_memory.pop_all()
        state_tensor = torch.as_tensor(states, dtype=torch.float, device=DEVICE)
        action_tensor = torch.as_tensor(actions, device=DEVICE)
        # -------------------------
        # Evaluate critic
        # -------------------------

        # Compute q value
        q_value = self.ACNet.critic(state_tensor)

        # Compute expected q value for each step
        values = np.zeros(len(rewards))
        if not done:
            values[-1] = q_value[-1]

        v = 0
        for t in reversed(range(0, len(rewards))):
            v = GAMMA * v + rewards[t]
            values[t] = v

        # values -= np.mean(values)
        # values /= np.std(values)
        expected_q_value = torch.as_tensor(values, device=DEVICE).unsqueeze(-1).detach()

        # Compute square error loss
        td_error = expected_q_value - q_value
        critic_loss = torch.mean(torch.square(td_error))

        # -------------------------
        # Evaluate actor
        # -------------------------

        action_prob = self.ACNet.actor(state_tensor)
        # Cross entropy returns negative log likelihood loss
        actor_loss = F.cross_entropy(action_prob, action_tensor, reduction='none').unsqueeze(-1)
        actor_loss = torch.mean(actor_loss * td_error.detach())

        # The entropy of action prob. To encourage exploration
        entropy_loss = -torch.mean(action_prob * torch.log(action_prob))

        total_loss = critic_loss + actor_loss + ENTROPY_BETA * entropy_loss

        # -------------------------
        # Optimize ACNet parameters
        # -------------------------

        global_opt.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.ACNet.parameters(), 1)
        for lp, gp in zip(self.ACNet.parameters(), global_agent.ACNet.parameters()):
            gp._grad = lp.grad
        global_opt.step()

        self.replay_memory.clear()


class Worker(mp.Process):
    def __init__(self, name, global_agent, global_opt):
        super(Worker, self).__init__()
        self.name = name
        self.env = gym.make(ENV_NAME).unwrapped
        self.global_agent = global_agent
        self.opt = global_opt
        self.local_agent = Agent(self.env)

    def run(self):
        state = self.env.reset()  # init state
        done = False
        step = 1
        self.local_agent.ACNet.load_state_dict(self.global_agent.ACNet.state_dict())

        while not done and step < MAX_STEP:
            action = self.local_agent.select_action(state)
            next_state, reward_, done, _ = self.env.step(action)
            reward = -1 if done else 0.1
            self.local_agent.replay_memory.push(state, action, reward)

            if step % GLOBAL_UPDATE_STEP == 0 or done:
                self.local_agent.update(self.global_agent, self.opt, done)
                self.local_agent.ACNet.load_state_dict(self.global_agent.ACNet.state_dict())

            state = next_state
            step += 1


def main():
    # Torch multiprocess setting for CUDA
    # mp.set_start_method('spawn')
    # Init Open AI Gym env and dqn agent
    env = gym.make(ENV_NAME)
    global_agent = Agent(env)
    global_agent.ACNet.share_memory()
    global_opt = torch.optim.Adam(global_agent.ACNet.parameters(), lr=LR)

    start_time = time.time()
    for episode in range(GLOBAL_EPISODE):
        # ------------------
        # Train on workers
        # ------------------

        # Create workers
        workers = []
        for i in range(WORKER_NUM):
            worker_name = 'worker %i' % i
            workers.append(Worker(worker_name, global_agent, global_opt))

        # Start the job
        for worker in workers:
            worker.start()

        # Wait until finishing all the job
        for worker in workers:
            worker.join()

        # ---------------------------
        # Test on global agent
        # ---------------------------

        if episode % 50 == 0:
            global_agent.ACNet.eval()
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                done = False
                step = 1
                while not done and step < MAX_STEP:
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
