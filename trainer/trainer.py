import numpy as np
import torch
from env.cmg.matrix import MatrixGame
from policy import Policy, PolicyPool

class Trainer:
    def __init__(self, env, policy_pool, n_episodes):
        self.env = env
        self.policy_pool = policy_pool
        self.n_episodes = n_episodes
        
        if policy_pool is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.returns = None
        else:
            n_policies = len(policy_pool)
            self.device = self.policy_pool[0].device if n_policies > 0 else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.returns = torch.zeros((n_policies, n_episodes), device=self.device)
        
        self.payoff_matrix = self.env.payoff_matrix
    
    def entropy(self, joint_policy):
        eps = 1e-10
        return -(torch.log(joint_policy + eps) * joint_policy).sum()
    
    def evaluate(self, policy_C, policy_R, return_losses=False, return_probs=False):
        action_C = policy_C.get_action('column')
        action_R = policy_R.get_action('row')
        
        p_r, _ = policy_R.get_policy()
        _, p_c = policy_C.get_policy()
        
        joint_policy = torch.outer(p_r, p_c)
        expected_reward = torch.sum(joint_policy * self.payoff_matrix)
        
        actions = {
            self.env.agents[0]: action_C,
            self.env.agents[1]: action_R
        }
        
        _, reward, _, _, info = self.env.step(actions)
        
        if return_probs:
            return expected_reward, None, joint_policy
        
        if return_losses:
            return expected_reward, None
        
        return expected_reward
    
    def train(self, policy):
        policy.zero_grad()

        action_r = policy.get_action('row')
        action_c = policy.get_action('column')
        
        actions = {
            self.env.agents[0]: action_c,
            self.env.agents[1]: action_r}
        
        _, reward, _, _, _ = self.env.step(actions)
    
        loss = policy.compute_loss(self.payoff_matrix)
        loss.backward()
        policy.step()
        
        return reward
    
    def evaluate_batch(self, policy_pool_C=None, policy_pool_R=None, n_eval_episodes=10):
        policy_pool_C = policy_pool_C or self.policy_pool
        policy_pool_R = policy_pool_R or self.policy_pool

        n_policies_C = len(policy_pool_C)
        n_policies_R = len(policy_pool_R)
        
        reward_matrix = torch.zeros((n_policies_R, n_policies_C), device=self.device)
        joint_policies = []
        
        for i, policy_R in enumerate(policy_pool_R):
            row_joint_policies = []
            
            for j, policy_C in enumerate(policy_pool_C):
                episode_rewards = torch.zeros(n_eval_episodes, device=self.device)
                
                for ep in range(n_eval_episodes):
                    self.env.reset()
                    
                    if ep == 0:
                        reward, _, joint_policy = self.evaluate(policy_C, policy_R, 
                                                              return_losses=False, 
                                                              return_probs=True)
                        episode_rewards[ep] = reward
                        row_joint_policies.append(joint_policy)
                    else:
                        reward = self.evaluate(policy_C, policy_R, 
                                             return_losses=False, 
                                             return_probs=False)
                        episode_rewards[ep] = reward
                
                avg_reward = episode_rewards.mean()
                reward_matrix[i, j] = avg_reward
            
            joint_policies.append(row_joint_policies)
        
        n_policies = min(n_policies_R, n_policies_C)
        selfplay_returns = torch.diagonal(reward_matrix[:n_policies, :n_policies])
        
        mask = ~torch.eye(n_policies, dtype=bool, device=self.device)
        crossplay_returns = reward_matrix[:n_policies, :n_policies][mask]
        
        returns = {
            'reward_matrix': reward_matrix,
            'joint_policies': joint_policies,
            'selfplay_returns': selfplay_returns,
            'crossplay_returns': crossplay_returns
        }
        
        return returns
    
    def train_policies(self, payoff_matrix=None):
        if payoff_matrix is None:
            payoff_matrix = self.payoff_matrix
        
        self.policy_pool.zero_grad_all()
        
        joint_policies = self.policy_pool.get_stacked_policies()
        n_policies = len(self.policy_pool)
        
        expected_rewards = torch.sum(joint_policies * payoff_matrix.unsqueeze(0), dim=(1, 2))
        losses = -expected_rewards
        
        for i, policy in enumerate(self.policy_pool.policies):
            policy.zero_grad()
            losses[i].backward(retain_graph=(i < n_policies - 1))
        
        self.policy_pool.step_all()
        
        batch_actions, actions_r, actions_c = self.policy_pool.get_actions_dict(self.env.agents)
        
        batch_rewards = torch.zeros(n_policies, device=self.device)
        for i, actions in enumerate(batch_actions):
            self.env.reset()
            _, reward, _, _, _ = self.env.step(actions)
            batch_rewards[i] = reward
        
        return batch_rewards
    
    def train_custom_policies(self, loss_fn, payoff_matrix=None):
        if payoff_matrix is None:
            payoff_matrix = self.payoff_matrix
        
        self.policy_pool.zero_grad_all()
        
        joint_policies = self.policy_pool.get_stacked_policies()
        n_policies = len(self.policy_pool)
        
        losses = loss_fn(joint_policies, payoff_matrix)
        
        for i, policy in enumerate(self.policy_pool.policies):
            policy.zero_grad()
            losses[i].backward(retain_graph=(i < n_policies - 1))
        
        self.policy_pool.step_all()
        
        batch_actions, actions_r, actions_c = self.policy_pool.get_actions_dict(self.env.agents)
        
        batch_rewards = torch.zeros(n_policies, device=self.device)
        for i, actions in enumerate(batch_actions):
            self.env.reset()
            _, reward, _, _, _ = self.env.step(actions)
            batch_rewards[i] = reward
        
        return batch_rewards
    
    def train_population(self, n_episodes):
        n_policies = len(self.policy_pool)
        self_play_returns = torch.zeros((n_policies, n_episodes), device=self.device)
        cross_play_returns = torch.zeros((n_policies*(n_policies-1), n_episodes), device=self.device)
        
        for episode in range(n_episodes):
            self.env.reset()
            self_play_rewards = self.train_policies()
            self_play_returns[:, episode] = self_play_rewards
            
            eval_results = self.evaluate_batch()
            
            cross_play_returns[:, episode] = eval_results['crossplay_returns']
            
            avg_self_play = self_play_rewards.mean().item()
            avg_cross_play = eval_results['crossplay_returns'].mean().item()
            avg_overall = eval_results['reward_matrix'].mean().item()
            print(f"Episode {episode+1}/{n_episodes}, Avg SP: {avg_self_play:.4f}, Avg CP: {avg_cross_play:.4f}, Avg Overall: {avg_overall:.4f}")
        
        return self_play_returns, cross_play_returns
    
    def train_custom_population(self, loss_fn, n_episodes):
        n_policies = len(self.policy_pool)
        self_play_returns = torch.zeros((n_policies, n_episodes), device=self.device)
        cross_play_returns = torch.zeros((n_policies*(n_policies-1), n_episodes), device=self.device)
        
        for episode in range(n_episodes):
            self.env.reset()
            self_play_rewards = self.train_custom_policies(loss_fn)
            self_play_returns[:, episode] = self_play_rewards
            
            eval_results = self.evaluate_batch()
            
            cross_play_returns[:, episode] = eval_results['crossplay_returns']
            
            avg_self_play = self_play_rewards.mean().item()
            avg_cross_play = eval_results['crossplay_returns'].mean().item()
            avg_overall = eval_results['reward_matrix'].mean().item()
            print(f"Episode {episode+1}/{n_episodes}, Avg SP: {avg_self_play:.4f}, Avg CP: {avg_cross_play:.4f}, Avg Overall: {avg_overall:.4f}")
        
        return self_play_returns, cross_play_returns



