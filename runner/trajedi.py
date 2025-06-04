import torch
import numpy as np
import pathlib
import pickle
from env import MatrixGame
from policy import Policy, PolicyPool
from trainer import Trainer
from utils.visualize import Visualizer
import matplotlib.pyplot as plt

def entropy(pi):
    return -(torch.log(pi + 1e-8) * pi).sum()

class TrajediRunner:
    def __init__(self,
                 env_layout="cmg_s", 
                 n_policies=8, 
                 n_episodes=100, 
                 policy_lr=0.05, 
                 policy_std=0.01,
                 div_factor=5.0,
                 device=None):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env_layout = env_layout
        self.n_policies = n_policies
        self.n_episodes = n_episodes
        self.policy_lr = policy_lr
        self.policy_std = policy_std
        self.div_factor = div_factor
        
        self.training_results = {}
        self.evaluation_results = {}
    
        # Get the project root directory
        self.project_root = pathlib.Path(__file__).resolve().parent.parent
        self.results_dir = self.project_root / "results"
        self.trajedi_dir = self.results_dir / "trajedi"
        self.env_layout_dir = self.trajedi_dir / self.env_layout
        self.policy_dir = self.env_layout_dir / str(self.n_policies)
        self.policy_dir.mkdir(parents=True, exist_ok=True)
        
        self.base_filename = f"trajedi_{self.env_layout}_{self.n_policies}"
    
    def _custom_loss(self, pool, trainer, payoff_matrix):
        loss_vector = pool.batch_compute_losses(payoff_matrix)
        
        for idx_policy in range(1, len(pool.policies)):
            r1 = trainer.evaluate(pool.policies[0], pool.policies[idx_policy])
            r2 = trainer.evaluate(pool.policies[idx_policy], pool.policies[0])
            
            loss_vector[0] += -r1
            loss_vector[0] += -r2

        if len(pool.policies) > 1 and self.div_factor > 0:
            probs = [p.get_joint_policy().reshape(-1) for p in pool.policies[1:]]
            n = len(probs)

            avg_entropy = sum(entropy(p) for p in probs) / n

            avg_policy = sum(probs) / n
            entropy_of_avg = entropy(avg_policy)
            jsd = entropy_of_avg - avg_entropy

            for i in range(1, len(pool.policies)):
                loss_vector[i] -= self.div_factor * jsd
        
        return loss_vector

    def train_trajedi(self, n_seeds):
        pools_trainers = []
        BR_SP_records = []
        BR_XP_records = []
        
        initial_pools = []
        
        for seed in range(n_seeds):
            self._set_seed(seed)
            env = MatrixGame(layout=self.env_layout, device=self.device)
            local_payoff = env.payoff_matrix
            pool = PolicyPool(n_policies=self.n_policies+1, 
                              n_dim=env.n_dim,
                              lr=self.policy_lr,
                              std_val=self.policy_std,
                              device=self.device)
            trainer = Trainer(env, pool, n_episodes=self.n_episodes)
            pools_trainers.append((pool, trainer, local_payoff))
            BR_SP_records.append([])
            BR_XP_records.append([])
            initial_pools.append(pool.clone())

        middle_pools = []
        middle_step = self.n_episodes // 2

        for step in range(self.n_episodes):
            for idx, (pool, trainer, payoff) in enumerate(pools_trainers):
                loss_fn = lambda joint_policies, pm=payoff, p=pool, t=trainer: self._custom_loss(p, t, pm)
                _ = trainer.train_custom_policies(loss_fn)
                br_sp_reward = trainer.evaluate(pool.policies[0], pool.policies[0])
                BR_SP_records[idx].append(br_sp_reward.detach())
                
                if step == middle_step:
                    middle_pools.append(pool.clone())
            
            for i in range(n_seeds):
                xp_rewards = []
                for j in range(n_seeds):
                    if i != j:
                        pool_i, trainer_i, _ = pools_trainers[i]
                        pool_j, _, _ = pools_trainers[j]
                        xp_reward = trainer_i.evaluate(pool_i.policies[0], pool_j.policies[0])
                        xp_rewards.append(xp_reward.detach())
                
                if n_seeds == 1:
                    avg_xp_reward = torch.tensor(0.0, device=self.device)
                else:
                    avg_xp_reward = sum(xp_rewards) / len(xp_rewards) if xp_rewards else torch.tensor(0.0, device=self.device)
                BR_XP_records[i].append(avg_xp_reward)
            
            avg_br = sum(record[-1].item() for record in BR_SP_records) / n_seeds
            avg_xp = sum(record[-1].item() for record in BR_XP_records) / n_seeds
            if (step + 1) % 50 == 0:
                print(f"Step {step+1}/{self.n_episodes}, Avg BR Self-play: {avg_br:.4f}, Avg BR Cross-play: {avg_xp:.4f}")
        
        multi_seed_results = {}
        for seed in range(n_seeds):
            br_policy = pools_trainers[seed][0].policies[0].get_joint_policy().detach().cpu().numpy()
            multi_seed_results[seed] = {
                'BR_SP': [r.detach().cpu() for r in BR_SP_records[seed]],
                'BR_XP': [r.detach().cpu() for r in BR_XP_records[seed]],
                'pool_start': initial_pools[seed],
                'pool_mid': middle_pools[seed],
                'pool_end': pools_trainers[seed][0],
                'BR': br_policy
            }
        
        return multi_seed_results
    
    def _set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def save_results(self, results, pool_fig, training_fig):
        pickle_path = self.policy_dir / f"{self.base_filename}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        
        training_plot_path = self.policy_dir / f"train_{self.base_filename}.png"
        training_fig.savefig(training_plot_path)
        plt.close(training_fig)
        
        pool_plot_path = self.policy_dir / f"{self.base_filename}.png"
        pool_fig.savefig(pool_plot_path)
        plt.show()
        
        print(f"Results saved to {self.policy_dir}")

if __name__ == "__main__":
    env_layout = "cmg_s_suboptimal"
    n_episodes = 200
    n_seeds = 4
    policy_lr = 0.05
    policy_std = 0.01
    div_factor = 1.0
    
    # Run for different policy sizes
    policy_sizes = [8]
    for n_policies in policy_sizes:
        print(f"\nTraining Trajedi with {n_policies} policies for {n_episodes} episodes and {n_seeds} seeds.")
        env = MatrixGame(layout=env_layout)
        visualizer = Visualizer(env)
        
        runner = TrajediRunner(env_layout=env_layout,
                             n_policies=n_policies,
                             n_episodes=n_episodes,
                             policy_lr=policy_lr,
                             policy_std=policy_std,
                             div_factor=div_factor)
        
        results = runner.train_trajedi(n_seeds=n_seeds)
        
        # Calculate average BR-SP & BR-XP
        BR_SP_values = [results[seed]['BR_SP'][-1].detach().cpu().numpy() for seed in range(n_seeds)]
        avg_br_sp = np.mean(BR_SP_values)
        print(f"Average BR-SP reward: {avg_br_sp:.4f}")
        
        BR_XP_values = [results[seed]['BR_XP'][-1].detach().cpu().numpy() for seed in range(n_seeds)]
        avg_br_xp = np.mean(BR_XP_values)
        print(f"Average BR-XP reward: {avg_br_xp:.4f}")
        
        BR_SP_records = [results[seed]['BR_SP'] for seed in range(n_seeds)]
        BR_XP_records = [results[seed]['BR_XP'] for seed in range(n_seeds)]
        
        fig = visualizer.visualize_training_progress(SP_rewards=BR_SP_records, 
                                                   XP_rewards=BR_XP_records,
                                                   n_seeds=n_seeds,
                                                   labels=["BR Self-play", "BR Cross-play"])
        
        pool_fig = visualizer.visualize_pool(results[0]['pool_end'])
        runner.save_results(results, pool_fig, fig)