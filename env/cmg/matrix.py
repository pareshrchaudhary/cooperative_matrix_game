import torch
import numpy as np
import gymnasium as gym
from env.cmg._cmg_utils.layout_generator import LayoutGenerator
from env.cmg._cmg_utils.vizualize import CMGVisualizer

class MatrixGame(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, layout="diagonal", device=None):
        self.layout = layout
        self.layout_generator = LayoutGenerator()
        self.payoff_matrix, self.n_dim, self.reward_centers = self.layout_generator.get_payoff_matrix(layout)
        
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.payoff_matrix = torch.as_tensor(self.payoff_matrix, device=self.device)
        
        self.agents = ['agent_C', 'agent_R']
        self.current_actions = {agent: 0 for agent in self.agents}
        self._setup_spaces()
    
    def _setup_spaces(self):
        single_action_space = gym.spaces.Discrete(self.n_dim)
        
        self.action_space = gym.spaces.Dict({agent: single_action_space for agent in self.agents})
        
        self.observation_space = gym.spaces.Dict({
            'payoff_matrix': gym.spaces.Box(
                low=0, high=float('inf'), shape=(self.n_dim, self.n_dim), dtype=np.float32
            )
        })
    
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.current_actions = {agent: 0 for agent in self.agents}
        
        return self._get_observation(), {}
    
    def step(self, actions):      
        self.current_actions = actions
        
        actions_tensor = torch.tensor([actions[self.agents[0]], actions[self.agents[1]]], device=self.device)
        policies = torch.zeros(2, self.n_dim, device=self.device)
        policies.scatter_(1, actions_tensor.unsqueeze(1), 1.0)
        
        p_col, p_row = policies[0], policies[1]
        
        joint_policy = torch.outer(p_row, p_col)
        reward = torch.sum(joint_policy * self.payoff_matrix)
        
        info = {
            'joint_policy': joint_policy.detach().cpu().numpy(),
            'agent_actions': self.current_actions
        }
        
        return self._get_observation(), reward.item(), True, False, info
    
    def _convert_action_to_policy(self, action):
        policy = torch.zeros(self.n_dim, device=self.device)
        policy[action] = 1.0
        return policy
    
    def _get_observation(self):
        return {
            'payoff_matrix': self.payoff_matrix.cpu().numpy()
        }
    
    def render(self, mode='human', show_layout_only=False):
        if mode == 'human':
            agent_policies = {
                agent: self._convert_action_to_policy(action) 
                for agent, action in self.current_actions.items()
            }
            
            CMGVisualizer.render_matrix_game(
                payoff_matrix=self.payoff_matrix, 
                n_dim=self.n_dim, 
                thetas=agent_policies[self.agents[0]], 
                opponent_thetas=agent_policies[self.agents[1]],
                layout=self.layout,
                show_layout_only=show_layout_only,
                agents=self.agents
            )
    
    def sample_action(self):
        actions = torch.randint(0, self.n_dim, (len(self.agents),), device=self.device)
        return {agent: action.item() for agent, action in zip(self.agents, actions)}

if __name__ == "__main__":
    env = MatrixGame(layout="cmg_s_suboptimal")
    env.reset()
    env.render(show_layout_only=True)