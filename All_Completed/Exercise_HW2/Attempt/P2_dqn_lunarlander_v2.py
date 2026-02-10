# P2_dqn_lunarlander_v2.py

#import libraries
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import os

#set up environment
env = gym.make("LunarLander-v2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

#Q network
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 4)

    # forward
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

#hyperparameters
lr = 1e-3
gamma = 0.99
eps = 1.0
eps_min = 0.01
eps_decay = 0.995
episodes = 500
batch_size = 64
buffer_size = 100000
min_replay = 2000
update_target_every = 10

#initialize
policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

#optimizer
optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
replay_buffer = deque(maxlen=buffer_size)
os.makedirs("models", exist_ok=True)

#train
def train():
    batch = random.sample(replay_buffer, batch_size)
    states = torch.from_numpy(np.array([b[0] for b in batch], dtype=np.float32)).to(device)
    actions = torch.tensor([b[1] for b in batch], dtype=torch.int64).unsqueeze(1).to(device)
    rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32).unsqueeze(1).to(device)
    next_states = torch.tensor([b[3] for b in batch], dtype=torch.float32).to(device)
    dones = torch.tensor([b[4] for b in batch], dtype=torch.float32).unsqueeze(1).to(device)
    #calculate loss
    q_values = policy_net(states).gather(1, actions)
    with torch.no_grad():
        next_q = target_net(next_states).max(1)[0].unsqueeze(1)
        target = rewards + (1 - dones) * gamma * next_q
    #backprop
    loss = F.mse_loss(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#train
print("Training started...\n")
scores = []
best_avg = -float('inf')

#main loop
for ep in range(1, episodes + 1):
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = policy_net(state_tensor).argmax().item()

        #step
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        #store
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if len(replay_buffer) >= min_replay:
            train()

    #decay epsilon
    eps = max(eps * eps_decay, eps_min)
    scores.append(total_reward)

    #update target network
    if ep % update_target_every == 0:
        target_net.load_state_dict(policy_net.state_dict())

    #print progress
    avg_reward = np.mean(scores[-100:])
    if ep % 10 == 0:
        print(f"Episode {ep:3d} | Reward: {total_reward:7.2f} | Avg100: {avg_reward:7.2f} | Eps: {eps:.3f}")

    #save best
    if len(scores) >= 100 and avg_reward > best_avg:
        best_avg = avg_reward
        torch.save(policy_net.state_dict(), "models/best_model.pth")
        print(f"  ✓ New best avg reward: {avg_reward:.2f}")

    #solved
    if avg_reward >= 200 and len(scores) >= 100:
        print(f"\n🎉 SOLVED at episode {ep}! Avg100: {avg_reward:.2f}")
        torch.save(policy_net.state_dict(), "models/solved_model.pth")
        break

#save final
torch.save(policy_net.state_dict(), "models/final_model.pth")
print("\nTraining complete!")

#Evaluate
print("\nEvaluating trained model...\n")
eval_env = gym.make("LunarLander-v2", render_mode="human")
policy_net.load_state_dict(torch.load("models/best_model.pth"))
policy_net.eval()

successes = 0
for i in range(5):
    state, _ = eval_env.reset()
    total = 0
    done = False

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action = policy_net(state_tensor).argmax().item()

        state, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        total += reward

    status = "SUCCESS ✓" if total >= 200 else "FAILED ✗"
    if total >= 200:
        successes += 1
    print(f"Eval Episode {i+1}: {total:7.2f} - {status}")

#close
eval_env.close()
print(f"\nSuccessful landings: {successes}/5")

