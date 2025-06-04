import torch
import numpy as np

class Policy:
    def __init__(self, n_dim, lr=0.05, std_val=0.01, device=None):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        theta_r_init = torch.randn(n_dim, 1, device="cpu") * std_val
        theta_c_init = torch.randn(n_dim, 1, device="cpu") * std_val
        self.theta_r = torch.nn.Parameter(theta_r_init.to(self.device))
        self.theta_c = torch.nn.Parameter(theta_c_init.to(self.device))
        self.optimizer = torch.optim.Adam([self.theta_r, self.theta_c], lr=lr)
        self.n_dim = n_dim
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def step(self):
        self.optimizer.step()
    
    def get_policy(self):
        p_r = torch.softmax(self.theta_r.squeeze(), dim=0)
        p_c = torch.softmax(self.theta_c.squeeze(), dim=0)
        return p_r, p_c
    
    def get_joint_policy(self):
        p_r, p_c = self.get_policy()
        return torch.outer(p_r, p_c)
    
    def get_action(self, player_role='row'):
        if player_role == 'row':
            p_r = torch.softmax(self.theta_r.squeeze(), dim=0)
            _ = torch.sum(p_r * torch.ones_like(p_r))
            return torch.argmax(self.theta_r).item()
        elif player_role == 'column':
            p_c = torch.softmax(self.theta_c.squeeze(), dim=0)
            _ = torch.sum(p_c * torch.ones_like(p_c))
            return torch.argmax(self.theta_c).item()
        else:
            raise ValueError("player_role must be 'row' or 'column'")
    
    def sample_action(self, player_role='row'):
        p_r, p_c = self.get_policy()
        if player_role == 'row':
            return torch.multinomial(p_r, 1).item()
        elif player_role == 'column':
            return torch.multinomial(p_c, 1).item()
        else:
            raise ValueError("player_role must be 'row' or 'column'")
    
    def compute_loss(self, reward):
        if not torch.is_tensor(reward):
            reward = torch.tensor(reward, device=self.device)
        else:
            reward = reward.to(self.device)
        
        joint_policy = self.get_joint_policy()
        loss = -torch.sum(joint_policy * reward)
        return loss
    
    def save(self, path):
        torch.save({
            'theta_r': self.theta_r,
            'theta_c': self.theta_c,
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.theta_r = torch.nn.Parameter(checkpoint['theta_r'].to(self.device))
        self.theta_c = torch.nn.Parameter(checkpoint['theta_c'].to(self.device))
        self.optimizer = torch.optim.Adam([self.theta_r, self.theta_c], lr=self.optimizer.param_groups[0]['lr'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def clone(self):
        """Create a deep copy of the policy."""
        new_policy = Policy(self.n_dim, lr=self.optimizer.param_groups[0]['lr'], 
                          std_val=0.0, device=self.device)
        new_policy.theta_r.data.copy_(self.theta_r.data)
        new_policy.theta_c.data.copy_(self.theta_c.data)
        return new_policy

class PolicyPool:
    def __init__(self, n_policies, n_dim, lr=0.05, std_val=0.01, device=None):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policies = [Policy(n_dim, lr, std_val, device=self.device) for _ in range(n_policies)]
        self.size = len(self.policies)
        self.lr = lr
        self.std_val = std_val
        self.n_dim = n_dim
    
    def add_policy(self, policy):
        if policy.device != self.device:
            policy.theta_r = policy.theta_r.to(self.device)
            policy.theta_c = policy.theta_c.to(self.device)
        self.policies.append(policy)
        self.size = len(self.policies)
    
    def remove_policy(self, idx):
        policy = self.policies.pop(idx)
        self.size = len(self.policies)
        return policy
    
    def zero_grad_all(self):
        for policy in self.policies:
            policy.zero_grad()
    
    def step_all(self):
        for policy in self.policies:
            policy.step()
    
    def __getitem__(self, idx):
        return self.policies[idx]
    
    def __len__(self):
        return self.size
    
    def get_thetas(self):
        thetas_r = torch.stack([policy.theta_r for policy in self.policies])
        thetas_c = torch.stack([policy.theta_c for policy in self.policies])
        return thetas_r, thetas_c
    
    def get_actions(self, player_role='row'):
        if player_role == 'row':
            actions = torch.tensor([policy.get_action('row') for policy in self.policies], 
                                device=self.device)
        elif player_role == 'column':
            actions = torch.tensor([policy.get_action('column') for policy in self.policies], 
                                device=self.device)
        else:
            raise ValueError("player_role must be 'row' or 'column'")
        return actions
    
    def get_actions_dict(self, agents):
        actions_r = self.get_actions('row')
        actions_c = self.get_actions('column')
        
        batch_actions = [{
            agents[0]: actions_r[i].item(),
            agents[1]: actions_c[i].item()
        } for i in range(self.size)]
        
        return batch_actions, actions_r, actions_c
    
    def get_policies(self):
        return [policy.get_joint_policy() for policy in self.policies]
    
    def get_stacked_policies(self):
        joint_policies = [policy.get_joint_policy() for policy in self.policies]
        return torch.stack(joint_policies)
    
    def batch_compute_losses(self, payoff_matrix):
        if not torch.is_tensor(payoff_matrix):
            payoff_matrix = torch.tensor(payoff_matrix, device=self.device)
        else:
            payoff_matrix = payoff_matrix.to(self.device)
            
        joint_policies = self.get_stacked_policies()
        expected_rewards = torch.sum(joint_policies * payoff_matrix, dim=(1, 2))
        losses = -expected_rewards
        
        return losses
    
    def batch_compute_expected_rewards(self, payoff_matrix):
        if not torch.is_tensor(payoff_matrix):
            payoff_matrix = torch.tensor(payoff_matrix, device=self.device)
        else:
            payoff_matrix = payoff_matrix.to(self.device)
            
        joint_policies = self.get_stacked_policies()
        expected_rewards = torch.sum(joint_policies * payoff_matrix, dim=(1, 2))
        
        return expected_rewards
    
    def save(self, path):
        for i, policy in enumerate(self.policies):
            policy.save(f"{path}_policy_{i}.pt")
        
    def load(self, path):
        for i in range(self.size):
            self.policies[i].load(f"{path}_policy_{i}.pt")

    def clone(self):
        new_pool = PolicyPool(self.size, self.n_dim, lr=self.lr, 
                            std_val=self.std_val, device=self.device)
        for i, policy in enumerate(self.policies):
            new_pool.policies[i] = policy.clone()
        return new_pool