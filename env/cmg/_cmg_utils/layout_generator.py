import numpy as np
import torch

class LayoutGenerator:
    """Class for generating and managing different layouts for matrix games."""
    
    def __init__(self):
        """Initialize with default layout registry"""
        self.layout_registry = {
            "diagonal": self.create_diagonal,
            "cmg_s": self.create_cmg_s,
            "cmg_s_negative": self.create_cmg_s_negative,
            "cmg_s_suboptimal": self.create_cmg_s_suboptimal,
            "cmg_needle": self.create_cmg_needle,
            "cmg_needle_h": self.create_cmg_needle_h,
            "cmg_h": self.create_cmg_h,
            "cmg_ht": self.create_cmg_ht
        }
    
    # === Layout Generation Helpers ===
    
    def _create_layout(self, M, k_m, r_m):
        """Helper function to create layout with consistent format"""
        if isinstance(k_m, (int, float)):
            k_m = [k_m] * M
        if isinstance(r_m, (int, float)):
            r_m = [r_m] * M
        n_dim = sum(k_m)
        return (n_dim, M, k_m, r_m)
    
    # === Layout Registry and Utilities ===
    
    def register_layout(self, name, layout_function):
        """Register a new layout function to be used by the matrix game."""
        name = name.lower()
        if name in self.layout_registry:
            print(f"Warning: Layout '{name}' already exists and will be overwritten.")
        self.layout_registry[name] = layout_function
        print(f"Layout '{name}' registered successfully.")
    
    def create_custom_layout(self, n_dim=None, blocks=None):
        """Helper function to create custom layouts from dimensions or block specifications."""
        if blocks:
            M = len(blocks)
            k_m = [block[0] for block in blocks]
            r_m = [block[1] for block in blocks]
            total_dim = sum(k_m)
            return (total_dim, M, k_m, r_m)
        elif n_dim:
            return (n_dim, None, None, None)
        else:
            raise ValueError("Either n_dim or blocks must be specified")
    
    def get_available_layouts(self):
        """Returns list of all available layout names"""
        return list(self.layout_registry.keys())
    
    def get_reward_centers(self, layout):
        """Get reward centers for the specified layout"""
        layout = layout.lower()
        
        if layout not in self.layout_registry:
            raise ValueError(f"Unknown layout: {layout}. Available layouts: {', '.join(self.layout_registry.keys())}")
        
        n_dim, M, k_m, r_m = self.layout_registry[layout]()
        
        if M is None or k_m is None or r_m is None:
            return [(i, i, 1.0) for i in range(n_dim) if i != 1] + [(1, 0, 1.0)]
            
        centers = []
        action_idx = 0
        
        for m in range(M):
            k, reward = k_m[m], r_m[m]
            centers.extend([
                (action_idx + i, action_idx + j, reward) 
                for i in range(k) for j in range(k)
                if action_idx + i < n_dim and action_idx + j < n_dim
            ])
            action_idx += k
        
        return centers
    
    def get_payoff_matrix(self, layout):
        """Generate payoff matrix for the specified layout"""
        layout = layout.lower()
        
        if layout not in self.layout_registry:
            raise ValueError(f"Unknown layout: {layout}. Available layouts: {', '.join(self.layout_registry.keys())}")
        
        n_dim, _, _, _ = self.layout_registry[layout]()
        reward_centers = self.get_reward_centers(layout)
        
        payoff_matrix = torch.zeros((n_dim, n_dim))
        for row, col, reward in reward_centers:
            payoff_matrix[row, col] = reward
        
        return payoff_matrix, n_dim, reward_centers
    
    # === Default Layouts ===
    
    def create_diagonal(self):
        """Simple diagonal layout with fixed dimensionality"""
        n_dim = 20
        return (n_dim, None, None, None)
    
    # === Standard Block Layouts ===
    
    def create_cmg_s(self):
        """CMG with equal block sizes, rewards scaling from 0.5 to 1.0"""
        M = 8
        k_m = 8
        r_m = [round(0.5 + 0.5 * (m-1)/(M-1), 3) for m in range(1, M+1)]
        return self._create_layout(M, k_m, r_m)
    
    def create_cmg_needle(self):
        """CMG with equal blocks, last block has high reward (needle in haystack)"""
        M = 8
        k_m = 8
        r_m = [0.5] * (M-1) + [10.0]
        return self._create_layout(M, k_m, r_m)
    
    def create_cmg_needle_h(self):
        """CMG needle with smaller high-reward final block"""
        M = 8
        k_m = [8] * (M-1) + [2]
        r_m = [0.5] * (M-1) + [5.0]
        return self._create_layout(M, k_m, r_m)
    
    # === Hierarchical Layouts ===
    
    def create_cmg_h(self):
        """Hierarchical CMG with increasing block sizes, equal rewards"""
        M = 32
        k_m = [m for m in range(1, M+1)]
        r_m = 1.0
        return self._create_layout(M, k_m, r_m)
    
    def create_cmg_ht(self):
        """Hierarchical CMG with increasing sizes, rewards inversely proportional"""
        M = 32
        k_m = [m for m in range(1, M+1)]
        r_m = [round(1.0/k, 3) for k in k_m]
        return self._create_layout(M, k_m, r_m)
    
    # === Add more layouts as needed ===

    def create_cmg_s_negative(self):
        """CMG with equal block sizes, rewards ranging from -1.0 to 1.0"""
        M = 8
        k_m = 8
        r_m = [round(-1.0 + 2.0 * (m-1)/(M-1), 3) for m in range(1, M+1)]
        return self._create_layout(M, k_m, r_m)

    def create_cmg_s_suboptimal(self):
        """CMG with equal block sizes and multiple suboptimal rewards"""
        M = 8
        k_m = 8
        # Create a pattern with multiple peaks - main peak at 1.0, secondary at 0.8
        r_m = [1.0, 1.0, 0.1, 0.1, 0.3, 1.2, 0.2, 0.0]
        return self._create_layout(M, k_m, r_m)


