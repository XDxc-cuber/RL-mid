import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from tqdm import tqdm
import wandb
action_dim = 1653


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
    # method 1: 3600->1800
    def forward(self, x):        
        return self.network(x)
    # method 2  1800*d
    # def forward(self, x):
    #     self.dqn_x = self.network(x)
        
    #     return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, a_emb, next_a_embed):
        self.buffer.append((state, action, reward, next_state,  a_emb, next_a_embed))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim + 2*1653 # method 1, 2*worker_num denotes action_emb dim
        
        self.action_dim = action_dim 
        
        self.policy_net = DQN(self.state_dim, hidden_dim, self.action_dim).to(self.device)
        self.target_net = DQN(self.state_dim, hidden_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(10000)   # remove, worrying about some data don't be sampled.
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
    def select_action(self, state): # TODO
        # if random.random() < self.epsilon:   #TODO: use distribution
        #     return random.randrange(self.action_dim)

        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return nn.functional.softmax(q_values, dim=1)
            # return q_values.argmax().item()
    
    def train(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        
        batch = self.memory.sample(batch_size)
        state_batch = torch.FloatTensor(np.array([x[0] for x in batch])).to(self.device)
        action_batch = torch.LongTensor(np.array([x[1] for x in batch])).to(self.device)
        reward_batch = torch.FloatTensor(np.array([x[2] for x in batch])).to(self.device)
        next_state_batch = torch.FloatTensor(np.array([x[3] for x in batch])).to(self.device)
        a_emb_batch = torch.FloatTensor(np.array([x[4] for x in batch])).to(self.device)
        next_a_emb_batch = torch.FloatTensor(np.array([x[5] for x in batch])).to(self.device)  # TODO:得加，batch中的尾巴取不到
        
        if torch.isnan(state_batch).any() or torch.isinf(state_batch).any():
            print("警告: state_batch包含NaN或Inf值")
            # 可以选择记录或调试这些值
        # print("NaN索引:", torch.nonzero(torch.isnan(state_batch)))
        current_q_values_distribution = self.policy_net(torch.cat((state_batch, a_emb_batch), dim=1))
        current_q_values = current_q_values_distribution.gather(1, action_batch.unsqueeze(1))
        # 计算训练集奖励
        # print(current_q_values.size())
        action_probs = torch.softmax(current_q_values_distribution, dim=1)  # 对第二维应用Softmax，得到概率分布
        # print(action_probs.size())
        # 提取每个样本对应真实动作的概率，并与奖励相乘
        batch_indices = torch.arange(batch_size)
        selected_probs = action_probs[batch_indices, action_batch]  # 形状: [batch_size]
        # assert selected_probs.size() == batch_size
        # print(selected_probs, reward_batch)
        weighted_reward = selected_probs * reward_batch  # 概率加权奖励
        # print(weighted_reward)
        
        
        is_terminal = (state_batch.sum(dim=1) == 0).float()  # 根据实际终止条件调整
           # 确保is_terminal不包含非法值
        if torch.isnan(is_terminal).any() or torch.isinf(is_terminal).any():
            print("警告: is_terminal包含NaN或Inf值")
            
            
        next_q_values = self.target_net(torch.cat((next_state_batch, next_a_emb_batch), dim=1)).max(1)[0].detach()   # TODO:why is [0]?             
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - is_terminal)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss.item(), weighted_reward.sum().item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)
    
    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    
    def evaluate(self, valid_data, batch_size):
        """在验证集上批量评估模型，使用动作概率分布计算加权奖励"""
        self.policy_net.eval()  # 设置为评估模式
        total_weighted_reward = 0.0
        
        # 提取验证数据并转换为张量
        state = torch.FloatTensor(valid_data['s1'].numpy()).to(self.device)
        # entry_num = state.shape[0]
        action = torch.LongTensor(valid_data['a'].numpy()).to(self.device)  # 真实动作标签（用于索引概率）
        a_emb = torch.FloatTensor(valid_data['a_emb'].numpy()).to(self.device)
        reward = torch.FloatTensor(valid_data['r'].numpy()).to(self.device)
        for start_idx in range(0, len(state[0]), batch_size):
            
            end_idx = min(start_idx + batch_size, len(state[0]))
            batch_data = {
                's1': state[start_idx:end_idx],
                'a': action[start_idx:end_idx],
                'r': reward[start_idx:end_idx],
                'a_emb': a_emb[start_idx:end_idx]
            }
        with torch.no_grad():
            # 批量计算Q值和动作概率分布
            q_values = self.policy_net(batch_data['s1'])  # 形状: [batch_size, action_dim]
            action_probs = torch.softmax(q_values, dim=1)  # 对第二维应用Softmax，得到概率分布
            
            # 提取每个样本对应真实动作的概率，并与奖励相乘
            batch_indices = torch.arange(batch_data['s'].shape(0))
            selected_probs = action_probs[batch_indices, batch_data['a']]  # 形状: [batch_size]
            weighted_reward = selected_probs * batch_data['r']  # 概率加权奖励
            
            total_weighted_reward = weighted_reward.sum().item()
        
        self.policy_net.train()  # 恢复训练模式
        avg_weighted_reward = total_weighted_reward / len(valid_data['s2'])
        return avg_weighted_reward

def train_worker_dqn(worker_data, valid_data=None, num_episodes=1000, batch_size=64, target_update=10, log_interval=1000, eval_interval=5):
    state_dim = worker_data['s2'][0].shape[0]
    action_dim = 1653 # 
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
    
    state = worker_data['s1'].numpy()
    action = worker_data['a'].numpy()  #TODO: convert to tensor
    reward = worker_data['r']
    next_state = worker_data['s2'].numpy()
    a_emb = worker_data['a_space_emb'].numpy()    
    next_a_emb = a_emb.copy()
    n = a_emb.shape[2]
    next_a_emb[:, :, :n-1] = a_emb[:, :, 1:n]
    next_a_emb[:, :, n-1] = 0
    
    a_emb = transform_a_emb(state, a_emb)
    next_a_emb = transform_a_emb(state, next_a_emb)
    
    assert a_emb.ndim == 2
    assert state.ndim == 2
    for episode in range(num_episodes):
        print(f"Episode {episode} of {num_episodes}")
        total_reward = 0
        total_loss = 0
        train_steps = 0
        
        for i in tqdm(range(len(worker_data['s2']))):                  
            agent.memory.push(state[i], action[i], reward[i], next_state[i], a_emb[i], next_a_emb[i])  
            
            # 训练并获取loss
            if len(agent.memory) >= batch_size:
                loss, reward_train = agent.train(batch_size)
                if loss is not None:
                    total_loss += loss
                    total_reward += reward_train
                    train_steps += 1
                    global_step += 1
                    
                    # 每log_interval个batch记录一次
                    if global_step % log_interval == 0:
                        avg_loss = total_loss / train_steps if train_steps > 0 else 0
                        
                        
                        
                        avg_reward = total_reward / train_steps if train_steps > 0 else 0
                        
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
            
            
        if episode % target_update == 0:
            agent.update_target_network()
        
        # 每个episode结束时打印一次
        # avg_reward = total_reward / len(worker_data['s2'])
        # avg_loss = total_loss / train_steps if train_steps > 0 else 0
        # print(f"Episode {episode}, Avg Reward: {avg_reward:.4f}, Avg Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.2f}")
        
        # 每eval_interval个episode验证一次
        if valid_data is not None and episode % eval_interval == 0:
            valid_reward = agent.evaluate(valid_data, batch_size)
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


def transform_a_emb(state, a_emb):
    
    onehot_indices = state[:, :7].argmax(axis=1) # 7 denotes number of category
    # valid_mask = (state[:, :7].sum(axis=1) > 0)
    # onehot_indices = np.where(valid_mask, onehot_indices, np.zeros_like(onehot_indices))
    # 提取第一个元素和one-hot对应元素
    entry_num, m, n = a_emb.shape
    first_elements = a_emb[:, :, 0]  # [batch_size, m]
    # 使用向量化方式提取one-hot对应元素
    batch_indices = np.arange(entry_num).reshape(-1, 1)
    onehot_elements = a_emb[batch_indices, np.arange(m), 1+onehot_indices[:, np.newaxis]]
    # 合并特征：将[m, 2]展平为[m*2]
    a_emb = np.hstack([first_elements, onehot_elements])  # [batch_size, m*2]
    # print(1+onehot_indices[:, np.newaxis])
    return a_emb


if __name__ == "__main__":
    a = transform_a_emb(np.zeros((5,9)), np.zeros((5,6,8)))
    print(a)
