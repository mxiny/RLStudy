import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time


# ------------------------
# Hyper parameter for PPO
# ------------------------

GAMMA = 0.95  # discount factor for target Q
ENV_NAME = 'CartPole-v0'
EPISODE = 1000  # episode limitation
STEP = 300  # step limitation in an episode
TEST = 100  # the number of experiment test every 100 episode
A_LR = 1e-3  # learning rate for actor
C_LR = 1e-3  # learning rate for critic
BATCH_SIZE = 32  # update parameters every 32 steps
EPS_CLIP = 0.2  # clip surrogate loss
UPDATE_EPOCH = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -------------
# Actor Network
# -------------

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 20)
        self.fc2 = nn.Linear(20, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x


# ----------------
# Critic Network
# ----------------

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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
        self.actor = Actor(self.state_dim, self.action_dim).to(DEVICE)
        self.actor_op = torch.optim.Adam(self.actor.parameters(), lr=A_LR)

        self.actor_old = Actor(self.state_dim, self.action_dim).to(DEVICE)
        self.actor_old.load_state_dict(self.actor.state_dict())

        self.critic = Critic(self.state_dim).to(DEVICE)
        self.critic_op = torch.optim.Adam(self.critic.parameters(), lr=C_LR)
 
    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float, device=DEVICE)
            action_prob = self.actor.forward(state_tensor).cpu().detach().numpy()
            action = np.random.choice(range(self.action_dim), p=action_prob)  # select an action by actor network
        return action

    def update(self):
        # Sample a complete episode
        states, actions, rewards = self.replay_memory.pop_all()
        state_tensor = torch.as_tensor(states, dtype=torch.float, device=DEVICE)
        action_tensor = torch.as_tensor(actions, dtype=torch.long, device=DEVICE)

        # Compute discounted reward value for each step
        dc_reward = np.zeros(len(rewards))  # discounted reward
        value = 0
        for t in reversed(range(0, len(rewards))):
            value = GAMMA * value + rewards[t]
            dc_reward[t] = value

        dc_reward = (dc_reward - dc_reward.mean()) / (dc_reward.std() + 1e-8)
        dc_reward = torch.as_tensor(dc_reward, device=DEVICE).unsqueeze(-1).detach()

        dist_old = self.actor_old.forward(state_tensor)
        log_pro_old = torch.log(dist_old.gather(1, action_tensor.unsqueeze(-1)) + 1e-8)
        # log_pro_old = Categorical(dist_old).log_prob(action_tensor)

        for _ in range(UPDATE_EPOCH):

            # -----------------
            # Update Critic
            # -----------------

            # Compute estimated state value
            state_value = self.critic.forward(state_tensor)

            # Compute loss
            advantage = dc_reward - state_value
            critic_loss = torch.mean(torch.square(advantage))

            # Optimize parameters
            self.critic_op.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
            self.critic_op.step()

            # -----------------
            # Update Actor
            # -----------------

            # Compute estimated prob for action

            dist = self.actor.forward(state_tensor)
            log_prob = torch.log(dist.gather(1, action_tensor.unsqueeze(-1)) + 1e-8)

            # Compute surrogate loss
            ratio = torch.exp(log_prob - log_pro_old.detach())
            clipped_ratio = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP)
            actor_loss = -torch.mean(torch.min(ratio, clipped_ratio) * advantage.detach())

            # Optimize parameters
            self.actor_op.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
            self.actor_op.step()

        # Update old actor network
        self.actor_old.load_state_dict(self.actor.state_dict())

        # Clear replay memory
        self.replay_memory.clear()


def main():
    # Init Open AI Gym env and PPO agent
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
            agent.replay_memory.push(state, action, reward)
            state = next_state
            step += 1
            if step % BATCH_SIZE == 0 or done:
                agent.update()

        # ---------------------------
        # Test every 100 episodes
        # ---------------------------

        agent.actor.eval()
        if episode % TEST == 0:
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
