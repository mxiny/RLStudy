import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time


# ------------------------
# Hyper parameter for DQN
# ------------------------

GAMMA = 0.95  # discount factor for target Q
ENV_NAME = 'CartPole-v0'
EPISODE = 3000  # episode limitation
STEP = 300  # step limitation in an episode
TEST = 10  # the number of experiment test every 100 episode
LR = 1e-2  # learning rate for training DQN
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
        x = F.softmax(self.fc2(x), dim=-1)
        return x


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
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        # Init neural network
        self.policy_net = Net(self.state_dim, self.action_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LR)

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float, device=DEVICE)
            action_prob = self.policy_net.forward(state_tensor).cpu().detach().numpy()
            action = np.random.choice(range(self.action_dim), p=action_prob)  # select an action by Q network
        return action

    def update(self):
        # Sample a complete episode
        states, actions, rewards = self.replay_memory.pop_all()
        state_tensor = torch.as_tensor(states, dtype=torch.float, device=DEVICE)
        action_tensor = torch.as_tensor(actions, device=DEVICE)
        
        # Compute true value for each step
        values = np.zeros(len(rewards))
        value = 0
        for t in reversed(range(0, len(rewards))):
            value = GAMMA * value + rewards[t]
            values[t] = value

        values -= np.mean(values)
        values /= np.std(values)
        value_tensor = torch.as_tensor(values, device=DEVICE).detach()

        # Compute estimated prob for each action
        action_prob = self.policy_net.forward(state_tensor)

        # Compute loss
        loss = F.cross_entropy(action_prob, action_tensor, reduction='none')
        loss = torch.sum(loss * value_tensor)

        # Optimize model parameters
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step()

        # Clear replay memory
        self.replay_memory.clear()

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
            agent.replay_memory.push(state, action, reward)
            state = next_state
            step += 1
            if done:
                agent.update()

        # ---------------------------
        # Test every 100 episodes
        # ---------------------------

        # agent.policy_net.eval()
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
