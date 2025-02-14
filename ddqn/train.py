import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from ddqn.env_wrapper import CartPoleImageWrapper
from ddqn.model import DQN
from ddqn.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt

# 超参数
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 10000
TARGET_UPDATE = 10
LEARNING_RATE = 1e-4
MEMORY_SIZE = 10000
EPISODES = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 创建环境
    env = CartPoleImageWrapper(gym.make('CartPole-v1', render_mode="rgb_array"))
    
    # 创建网络
    policy_net = DQN(env.observation_space.shape[:-1], env.action_space.n).to(device)
    target_net = DQN(env.observation_space.shape[:-1], env.action_space.n).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayBuffer(MEMORY_SIZE)
    
    steps_done = 0
    
    def select_action(state, eps_threshold):
        if random.random() > eps_threshold:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                return policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(env.action_space.n)]], 
                              device=device, dtype=torch.long)
    
    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        
        states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
        
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        # Double DQN
        next_actions = policy_net(next_states).max(1)[1].unsqueeze(1)
        next_state_values = target_net(next_states).gather(1, next_actions)
        expected_state_action_values = rewards.unsqueeze(1) + GAMMA * next_state_values * (1 - dones.unsqueeze(1))
        
        state_action_values = policy_net(states).gather(1, actions.unsqueeze(1))
        
        loss = nn.MSELoss()(state_action_values, expected_state_action_values.detach())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 添加记录列表
    episode_rewards = []
    
    # 训练循环
    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        
        while True:
            epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                     np.exp(-1. * steps_done / EPSILON_DECAY)
            steps_done += 1
            
            action = select_action(state, epsilon)
            next_state, reward, done, info = env.step(action.item())
            
            memory.push(state, action.item(), reward, next_state, done)
            state = next_state
            total_reward += reward
            
            optimize_model()
            
            if done:
                print(f"Episode {episode}, Total Reward: {total_reward}")
                episode_rewards.append(total_reward)
                break
        
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
    
    # 训练结束后绘制图表
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig('training_progress.png')
    plt.close()
    
    env.close()

if __name__ == "__main__":
    main()