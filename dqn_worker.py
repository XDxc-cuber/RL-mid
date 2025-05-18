import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from tqdm import tqdm
import wandb

class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim. hidden_dim=128, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim  # 0: 拒绝, 1: 接受
        
        self.policy_net = DQN(state_dim, hidden_dim, self.action_dim).to(self.device)
        self.target_net = DQN(state_dim, hidden_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(10000)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def train(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        
        batch = self.memory.sample(batch_size)
        state_batch = torch.FloatTensor(np.array([x[0] for x in batch])).to(self.device)
        action_batch = torch.LongTensor(np.array([x[1] for x in batch])).to(self.device)
        reward_batch = torch.FloatTensor(np.array([x[2] for x in batch])).to(self.device)
        next_state_batch = torch.FloatTensor(np.array([x[3] for x in batch])).to(self.device)
        
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)
    
    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def evaluate(self, valid_data):
        """在验证集上评估模型"""
        total_reward = 0
        self.policy_net.eval()  # 设置为评估模式
        
        with torch.no_grad():
            for i in range(len(valid_data['s2'])):
                state = valid_data['s2'][i].numpy()
                action = self.select_action(state)
                reward = valid_data['r'][i]
                total_reward += reward
        
        self.policy_net.train()  # 恢复训练模式
        return total_reward / len(valid_data['s2'])

def train_worker_dqn(worker_data, valid_data=None, num_episodes=1000, batch_size=64, target_update=10, log_interval=1000, eval_interval=5):
    state_dim = worker_data['s2'][0].shape[0]
    action_dim = worker_data['a'][0].shape[0]
    agent = DQNAgent(state_dim, action_dim)
    
    # 初始化wandb
    wandb.init(
        project="worker-dqn",
        config={
            "learning_rate": agent.optimizer.param_groups[0]['lr'],
            "gamma": agent.gamma,
            "epsilon_start": agent.epsilon,
            "epsilon_min": agent.epsilon_min,
            "epsilon_decay": agent.epsilon_decay,
            "batch_size": batch_size,
            "target_update": target_update,
            "state_dim": state_dim,
            "log_interval": log_interval,
            "eval_interval": eval_interval
        }
    )
    
    global_step = 0
    best_valid_reward = float('-inf')
    
    for episode in range(num_episodes):
        print(f"Episode {episode} of {num_episodes}")
        total_reward = 0
        total_loss = 0
        train_steps = 0
        
        for i in tqdm(range(len(worker_data['s2']))):
            state = worker_data['s2'][i].numpy()
            action = worker_data['a'][i]
            reward = worker_data['r'][i]
            
            if i < len(worker_data['s2']) - 1:
                next_state = worker_data['s2'][i + 1].numpy()
            else:
                next_state = state
            
            agent.memory.push(state, action, reward, next_state)
            
            # 训练并获取loss
            if len(agent.memory) >= batch_size:
                loss = agent.train(batch_size)
                if loss is not None:
                    total_loss += loss
                    train_steps += 1
                    global_step += 1
                    
                    # 每log_interval个batch记录一次
                    if global_step % log_interval == 0:
                        avg_loss = total_loss / train_steps if train_steps > 0 else 0
                        avg_reward = total_reward / (i + 1) if i > 0 else 0
                        
                        wandb.log({
                            "global_step": global_step,
                            "episode": episode,
                            "avg_reward": avg_reward,
                            "avg_loss": avg_loss,
                            "epsilon": agent.epsilon
                        })
                        
                        # 重置计数器
                        total_loss = 0
                        total_reward = 0
                        train_steps = 0
            
            total_reward += reward
        
        if episode % target_update == 0:
            agent.update_target_network()
        
        # 每个episode结束时打印一次
        avg_reward = total_reward / len(worker_data['s2'])
        avg_loss = total_loss / train_steps if train_steps > 0 else 0
        print(f"Episode {episode}, Avg Reward: {avg_reward:.4f}, Avg Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.2f}")
        
        # 每eval_interval个episode验证一次
        if valid_data is not None and episode % eval_interval == 0:
            valid_reward = agent.evaluate(valid_data)
            print(f"Validation at Episode {episode}, Avg Reward: {valid_reward:.4f}")
            
            # 记录验证结果
            wandb.log({
                "episode": episode,
                "valid_reward": valid_reward
            })
            
            # 保存最佳模型
            if valid_reward > best_valid_reward:
                best_valid_reward = valid_reward
                agent.save_model("models/worker_dqn_best.pth")
                print(f"New best model saved with validation reward: {valid_reward:.4f}")
    
    wandb.finish()
    return agent 