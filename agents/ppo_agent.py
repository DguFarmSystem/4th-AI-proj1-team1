import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import os
from collections import deque
import random

# CUDA 완전 비활성화
os.environ['CUDA_VISIBLE_DEVICES'] = ''

class PolicyNetwork(nn.Module):
    """
    PPO Policy Network (Actor)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        action_logits = self.fc3(x)
        return F.softmax(action_logits, dim=-1)
    
    def get_action_and_log_prob(self, state):
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

class ValueNetwork(nn.Module):
    """
    PPO Value Network (Critic)
    """
    def __init__(self, state_dim, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        value = self.fc3(x)
        return value

class PPOAgent:
    """
    진짜 Proximal Policy Optimization (PPO) 에이전트
    """
    def __init__(self, n_users, n_items, context_dim=64, 
                 learning_rate=3e-4, gamma=0.99, eps_clip=0.2,
                 k_epochs=4, hidden_dim=128):
        
        self.n_users = n_users
        self.n_items = n_items
        self.context_dim = context_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        # 상태 차원: user_id + context (히스토리 제외하여 차원 축소)
        self.state_dim = n_users + context_dim
        
        # 네트워크 초기화
        self.policy_net = PolicyNetwork(self.state_dim, n_items, hidden_dim)
        self.value_net = ValueNetwork(self.state_dim, hidden_dim)
        
        # 옵티마이저
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        # 경험 버퍼
        self.memory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'values': [],
            'dones': []
        }
        
        # 사용자별 상호작용 히스토리
        self.user_histories = {uid: np.zeros(n_items) for uid in range(n_users)}
        
        # 훈련 통계
        self.training_stats = {
            'rewards': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'episode_lengths': []
        }
        
        # 학습 카운터
        self.learning_count = 0
        
    def get_state(self, user_id, context):
        """
        상태 벡터 생성
        """
        # 사용자 원핫 인코딩
        user_onehot = np.zeros(self.n_users)
        user_onehot[user_id] = 1.0
        
        # 컨텍스트 정규화
        if context is not None and len(context) > 0:
            context_vec = np.array(context[:self.context_dim])
            if len(context_vec) < self.context_dim:
                context_vec = np.pad(context_vec, (0, self.context_dim - len(context_vec)))
        else:
            context_vec = np.zeros(self.context_dim)
        
        # 상태 벡터 결합
        state = np.concatenate([user_onehot, context_vec])
        return torch.FloatTensor(state)
    
    def select_action(self, user_id, context=None, deterministic=False):
        """
        액션 선택
        """
        state = self.get_state(user_id, context)
        
        if deterministic:
            # 결정적 정책 (평가용)
            with torch.no_grad():
                action_probs = self.policy_net(state)
                action = torch.argmax(action_probs).item()
                return action
        else:
            # 확률적 정책 (학습용)
            action, log_prob = self.policy_net.get_action_and_log_prob(state)
            
            # 메모리에 저장
            with torch.no_grad():
                value = self.value_net(state)
            
            self.memory['states'].append(state)
            self.memory['actions'].append(action)
            self.memory['log_probs'].append(log_prob)
            self.memory['values'].append(value)
            
            return action
    
    def store_experience(self, user_id, action, reward, context=None, done=False):
        """
        경험 저장
        """
        self.memory['rewards'].append(reward)
        self.memory['dones'].append(done)
        
        # 사용자 히스토리 업데이트
        self.user_histories[user_id][action] += 1
        
        # 통계 업데이트
        self.training_stats['rewards'].append(reward)
    
    def compute_advantages(self, rewards, values, dones):
        """
        Generalized Advantage Estimation (GAE) 계산
        """
        advantages = []
        returns = []
        
        # 마지막 값 (다음 상태가 없으면 0)
        next_value = 0
        
        # 역순으로 계산
        for i in reversed(range(len(rewards))):
            if dones[i]:
                next_value = 0
            
            # TD error 계산
            td_target = rewards[i] + self.gamma * next_value
            advantage = td_target - values[i].item()
            
            advantages.insert(0, advantage)
            returns.insert(0, td_target)
            
            next_value = values[i].item()
        
        return torch.FloatTensor(advantages), torch.FloatTensor(returns)
    
    def update(self):
        """
        PPO 업데이트
        """
        if len(self.memory['states']) < 16:  # 최소 배치 크기
            return
        
        # 데이터 준비
        states = torch.stack(self.memory['states'])
        actions = torch.LongTensor(self.memory['actions'])
        old_log_probs = torch.stack(self.memory['log_probs'])
        values = torch.stack(self.memory['values']).squeeze()
        rewards = self.memory['rewards']
        dones = self.memory['dones']
        
        # Advantage와 Return 계산
        advantages, returns = self.compute_advantages(rewards, values, dones)
        
        # Advantage 정규화
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # K 에폭 동안 학습
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for epoch in range(self.k_epochs):
            # Policy 업데이트
            self.policy_optimizer.zero_grad()
            
            # 현재 정책으로 액션 확률 계산
            action_probs = self.policy_net(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # PPO Loss 계산
            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            
            # Clipped Surrogate Objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
            
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_optimizer.step()
            
            # Value 업데이트 (별도 forward pass)
            self.value_optimizer.zero_grad()
            
            current_values = self.value_net(states).squeeze()
            value_loss = F.mse_loss(current_values, returns)
            
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()
            
            # 통계 누적
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
        
        # 통계 저장
        self.training_stats['policy_loss'].append(total_policy_loss / self.k_epochs)
        self.training_stats['value_loss'].append(total_value_loss / self.k_epochs)
        self.training_stats['entropy'].append(total_entropy / self.k_epochs)
        
        # 메모리 초기화
        self.clear_memory()
        self.learning_count += 1
        
        print(f"PPO 업데이트 완료 (#{self.learning_count})")
        print(f"  Policy Loss: {total_policy_loss/self.k_epochs:.4f}")
        print(f"  Value Loss: {total_value_loss/self.k_epochs:.4f}")
        print(f"  Entropy: {total_entropy/self.k_epochs:.4f}")
    
    def clear_memory(self):
        """
        메모리 초기화
        """
        for key in self.memory:
            self.memory[key] = []
    
    def get_top_k_recommendations(self, user_id, k=10, context=None):
        """
        상위 K개 추천 생성
        """
        state = self.get_state(user_id, context)
        
        with torch.no_grad():
            action_probs = self.policy_net(state)
            
        # 확률 기준으로 정렬
        sorted_items = torch.argsort(action_probs, descending=True)
        
        recommendations = []
        for i in range(min(k, len(sorted_items))):
            item_id = sorted_items[i].item()
            score = action_probs[item_id].item()
            recommendations.append((item_id, score))
        
        return recommendations
    
    def save_model(self, path):
        """
        모델 저장
        """
        try:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
            
            torch.save({
                'policy_net': self.policy_net.state_dict(),
                'value_net': self.value_net.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'value_optimizer': self.value_optimizer.state_dict(),
                'user_histories': self.user_histories,
                'training_stats': self.training_stats
            }, f"{path}.pth")
            
            print(f"PPO 모델 저장 완료: {path}.pth")
        except Exception as e:
            print(f"모델 저장 중 오류: {e}")
    
    def load_model(self, path):
        """
        모델 로드
        """
        try:
            checkpoint = torch.load(f"{path}.pth", map_location='cpu')
            
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.value_net.load_state_dict(checkpoint['value_net'])
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
            self.user_histories = checkpoint['user_histories']
            self.training_stats = checkpoint['training_stats']
            
            print(f"PPO 모델 로드 완료: {path}.pth")
        except Exception as e:
            print(f"모델 로드 중 오류: {e}")
    
    def evaluate(self, n_episodes=5):
        """
        모델 평가
        """
        recent_rewards = self.training_stats['rewards'][-100:] if self.training_stats['rewards'] else [0.0]
        
        return {
            'mean_reward': np.mean(recent_rewards),
            'std_reward': np.std(recent_rewards),
            'min_reward': np.min(recent_rewards),
            'max_reward': np.max(recent_rewards),
            'policy_loss': np.mean(self.training_stats['policy_loss'][-10:]) if self.training_stats['policy_loss'] else 0.0,
            'value_loss': np.mean(self.training_stats['value_loss'][-10:]) if self.training_stats['value_loss'] else 0.0
        }
    
    @property
    def experience_buffer(self):
        """
        호환성을 위한 experience_buffer 속성
        """
        return self.memory['states']

# 호환성을 위한 별칭
SimplePPOAgent = PPOAgent 