"""
dqn_lunarlander_v3.py
DQN skeleton for gymnasium LunarLander-v3 (continuous observations, discrete actions).
Requires: gymnasium, torch, numpy
"""
import gymnasium as gym
import numpy as np
import random
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import time
import os

# --- hyperparameters ---
ENV_NAME = "LunarLander-v3"
SEED = 42
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 64
REPLAY_CAPACITY = 100000
MIN_REPLAY_SIZE = 2000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.9995
TARGET_UPDATE_FREQ = 1000  # steps
MAX_EPISODES = 2000
MAX_STEPS_PER_EP = 1000
SAVE_DIR = "./models"
os.makedirs(SAVE_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

env = gym.make(ENV_NAME)
obs_shape = env.observation_space.shape[0]  # typically 8
n_actions = env.action_space.n  # typically 4

# --- DQN network ---
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# --- Replay Buffer ---
Transition = namedtuple("Transition", ("state","action","reward","next_state","done"))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(Transition(*args))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = torch.tensor([b.state for b in batch], dtype=torch.float32, device=DEVICE)
        actions = torch.tensor([b.action for b in batch], dtype=torch.int64, device=DEVICE).unsqueeze(1)
        rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=DEVICE).unsqueeze(1)
        next_states = torch.tensor([b.next_state for b in batch], dtype=torch.float32, device=DEVICE)
        dones = torch.tensor([b.done for b in batch], dtype=torch.float32, device=DEVICE).unsqueeze(1)
        return states, actions, rewards, next_states, dones
    def __len__(self):
        return len(self.buffer)

# --- utilities ---
def select_action(net, state, eps):
    if random.random() < eps:
        return env.action_space.sample()
    else:
        state_v = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        qvals = net(state_v)
        return int(torch.argmax(qvals, dim=1).item())

def compute_loss(policy_net, target_net, states, actions, rewards, next_states, dones):
    q_values = policy_net(states).gather(1, actions)
    with torch.no_grad():
        next_q = target_net(next_states).max(1)[0].unsqueeze(1)
        target = rewards + (1.0 - dones) * GAMMA * next_q
    loss = F.mse_loss(q_values, target)
    return loss

# --- models, optimizer, buffer ---
policy_net = DQN(obs_shape, n_actions).to(DEVICE)
target_net = DQN(obs_shape, n_actions).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = Adam(policy_net.parameters(), lr=LR)
replay = ReplayBuffer(REPLAY_CAPACITY)

# populate initial replay
state, _ = env.reset(seed=SEED)
for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    replay.push(state, action, reward, next_state, done)
    if done:
        state, _ = env.reset()
    else:
        state = next_state

# --- training loop ---
eps = EPS_START
global_step = 0
episode_rewards = deque(maxlen=100)

for ep in range(1, MAX_EPISODES+1):
    state, _ = env.reset()
    total_reward = 0.0
    for t in range(MAX_STEPS_PER_EP):
        global_step += 1
        action = select_action(policy_net, state, eps)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # Train
        if len(replay) >= BATCH_SIZE:
            s, a, r, ns, d = replay.sample(BATCH_SIZE)
            loss = compute_loss(policy_net, target_net, s, a, r, ns, d)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if global_step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        eps = max(EPS_END, eps * EPS_DECAY)
        if done:
            break

    episode_rewards.append(total_reward)

    if ep % 10 == 0:
        avg100 = np.mean(episode_rewards) if len(episode_rewards)>0 else 0.0
        print(f"Episode {ep} | Reward: {total_reward:.2f} | AvgLast{len(episode_rewards)}: {avg100:.2f} | eps: {eps:.4f} | Replay: {len(replay)}")

    # Save checkpoint when performance looks good
    if len(episode_rewards) == 100 and np.mean(episode_rewards) >= 200:  # threshold can be tuned
        save_path = os.path.join(SAVE_DIR, f"dqn_lunar_ep{ep}_avg{int(np.mean(episode_rewards))}.pt")
        torch.save(policy_net.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        break

# Final save
torch.save(policy_net.state_dict(), os.path.join(SAVE_DIR, "dqn_lunar_final.pt"))
print("Training finished. Model saved.")
env.close()

# --- Evaluation: run 5 episodes using saved model ---
def evaluate_model(weights_path, episodes=5, render=False):
    eval_env = gym.make(ENV_NAME, render_mode="human" if render else None)
    net = DQN(obs_shape, n_actions).to(DEVICE)
    net.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    net.eval()
    successes = 0
    for ep in range(episodes):
        s, _ = eval_env.reset()
        total = 0
        done = False
        while not done:
            a = int(torch.argmax(net(torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0))).item())
            ns, r, terminated, truncated, _ = eval_env.step(a)
            done = terminated or truncated
            total += r
            s = ns
            if render:
                time.sleep(0.02)
        print(f"Eval Episode {ep+1} Reward: {total:.2f}")
    eval_env.close()

# Usage example:
# evaluate_model("./models/dqn_lunar_final.pt", episodes=5, render=False)
