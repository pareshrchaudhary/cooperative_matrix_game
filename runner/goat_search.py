import torch
import numpy as np
import os
import pathlib
import pickle
import shutil
from env import MatrixGame
from policy.vae_policy import VAEPolicy
from utils.visualize import Visualizer
from runner.goat import GOATRunner

class VAEGOATOptimizer:
    def __init__(self,
                env_layout="cmg_h",
                algorithm="comedi",
                n_policies=6,
                goat_episodes=500,
                n_seeds=1,
                vae_epochs=2000,
                vae_batch_size=64,
                vae_lr=0.01,
                kl_coeff=0.01,
                ent_coeff=0.01,
                device=None):
        
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env_layout = env_layout
        self.algorithm = algorithm
        self.n_policies = n_policies
        self.goat_episodes = goat_episodes
        self.n_seeds = n_seeds
        self.vae_epochs = vae_epochs
        self.vae_batch_size = vae_batch_size
        self.vae_lr = vae_lr
        self.kl_coeff = kl_coeff
        self.ent_coeff = ent_coeff
        
        self.env = MatrixGame(layout=self.env_layout, device=self.device)
        self.visualizer = Visualizer(self.env)
        
        # Get the project root directory
        self.project_root = pathlib.Path(__file__).resolve().parent.parent
        self.results_dir = self.project_root / "results"
        self.vae_dir = self.results_dir / "vae" / self.algorithm / self.env_layout / str(self.n_policies)
        self.vae_dir.mkdir(parents=True, exist_ok=True)
        self.goat_dir = self.results_dir / "goat" / self.algorithm / self.env_layout / str(self.n_policies)
        self.goat_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_path = self.results_dir / self.algorithm / self.env_layout / str(self.n_policies) / f"{self.algorithm}_{self.env_layout}_{self.n_policies}.pkl"
        
        self.comedi_data_path = self.results_dir / "comedi" / self.env_layout / str(self.n_policies) / f"comedi_{self.env_layout}_{self.n_policies}.pkl"

    def calculate_reward_from_joint_policy(self, joint_policy):
        payoff_matrix = self.env.payoff_matrix
        expected_reward = torch.sum(joint_policy * payoff_matrix)
        return expected_reward.item()
        
    def load_training_data(self):
        print(f"Loading training data from {self.data_path}")
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        
        row_policies = []
        col_policies = []
        
        if isinstance(data, dict) and all(isinstance(v, dict) and 'pool_start' in v and 'pool_mid' in v and 'pool_end' in v for v in data.values()):
            seeds = list(data.keys())
            seeds = seeds[:self.n_seeds]
            print(f"Using {len(seeds)} seeds out of {len(data.keys())} available seeds")
            
            for seed in seeds:
                for stage in ['pool_start', 'pool_mid', 'pool_end']:
                    pool = data[seed][stage]
                    for policy in pool.policies:
                        p_r, p_c = policy.get_policy()
                        row_policies.append(p_r.detach())
                        col_policies.append(p_c.detach())
        else:
            raise ValueError("Unsupported data format. Expected dict with pool_start, pool_mid, and pool_end stages.")
        
        row_data = torch.stack(row_policies)
        col_data = torch.stack(col_policies)
        
        print(f"Loaded {len(row_policies)} policies for training (from start, mid, and end stages)")
        return row_data, col_data
    
    def train_vae_model(self, latent_dim, hidden_dim, beta):
        vae_policy = VAEPolicy(
            n_dim=self.env.n_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            beta=beta,
            lr=self.vae_lr,
            device=self.device
        )

        model_name = f"vae_policy_l{latent_dim}_h{hidden_dim}_b{beta}"
        model_path = self.vae_dir / f"{model_name}.pt"
        
        if os.path.exists(model_path):
            vae_policy.load(model_path)
            return vae_policy
        
        row_data, col_data = self.load_training_data()
        
        for epoch in range(self.vae_epochs):
            epoch_total_loss = 0
            epoch_recon_loss = 0
            epoch_kl_loss = 0
            num_batches = 0
            
            indices = torch.randperm(len(row_data))
            for i in range(0, len(row_data), self.vae_batch_size):
                batch_indices = indices[i:i+self.vae_batch_size]
                row_batch = row_data[batch_indices]
                col_batch = col_data[batch_indices]
            
                total_loss, recon_loss, kl_loss = vae_policy.train(row_batch, col_batch, row_batch)
                
                epoch_total_loss += total_loss
                epoch_recon_loss += recon_loss
                epoch_kl_loss += kl_loss
                num_batches += 1
        
        vae_policy.save(model_path)
        return vae_policy
    
    def run_goat_with_vae(self, vae_policy, latent_dim, hidden_dim, beta):
        expected_path = self.results_dir / "vae" / self.algorithm / self.env_layout / str(self.n_policies) / "vae_policy.pt"
        model_path = self.vae_dir / f"vae_policy_l{latent_dim}_h{hidden_dim}_b{beta}.pt"
        
        backup_path = None
        
        if os.path.exists(expected_path):
            backup_path = expected_path.with_suffix('.pt.bak')
            shutil.copy2(expected_path, backup_path)
            print(f"Backed up existing model to {backup_path}")
        
        shutil.copy2(model_path, expected_path)
        print(f"Copied our model to {expected_path} for GOAT")
        
        goat_runner = GOATRunner(
            env_layout=self.env_layout,
            n_episodes=self.goat_episodes,
            policy_lr=0.05,
            policy_std=0.01,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_policies_vae=self.n_policies,
            algorithm=self.algorithm,
            kl_coeff=self.kl_coeff,
            entropy_coeff=self.ent_coeff,
            adversary_lr=0.05,
            device=self.device
        )
        
        print(f"Training GOAT with VAE (latent_dim={latent_dim}, beta={beta}, hidden_dim={hidden_dim})...")
        results = goat_runner.train_goat(n_seeds=self.n_seeds)
        
        print("Generating true joint policy...")
        true_joint = goat_runner.generate_true_joint(vae_policy)
        
        if backup_path and os.path.exists(backup_path):
            shutil.copy2(backup_path, expected_path)
            os.remove(backup_path)
            print(f"Restored original model from backup")
        
        return results, true_joint, goat_runner
    
    def check_evaluation_exists(self, latent_dim, hidden_dim, beta):    
        model_name = f"n{self.n_policies}_l{latent_dim}_h{hidden_dim}_b{beta}_kl{self.kl_coeff}_ent{self.ent_coeff}"
        results_path = self.goat_dir / f"goat_eval_{model_name}.pkl"
        
        if os.path.exists(results_path):
            print(f"Evaluation already exists for hyperparameters: n_policies={self.n_policies}, latent_dim={latent_dim}, hidden_dim={hidden_dim}, beta={beta}")
            try:
                with open(results_path, 'rb') as f:
                    existing_eval = pickle.load(f)
                return True, existing_eval
            except Exception as e:
                print(f"Error loading existing evaluation: {e}")
                return False, None
        
        return False, None
        
    def evaluate_vae_with_goat(self, latent_dim, hidden_dim, beta):
        exists, existing_eval = self.check_evaluation_exists(latent_dim, hidden_dim, beta)
        if exists and existing_eval is not None:
            if 'goat_reward' in existing_eval:
                return existing_eval
        
        vae_policy = self.train_vae_model(latent_dim, hidden_dim, beta)
        results, true_joint, goat_runner = self.run_goat_with_vae(vae_policy, latent_dim, hidden_dim, beta)
        goat_reward = self.calculate_reward_from_joint_policy(true_joint)
        
        true_joint_fig = self.visualizer.visualize_pool(true_joint.unsqueeze(0), aggregation='avg', normalize=True)
        model_name = f"n{self.n_policies}_l{latent_dim}_h{hidden_dim}_b{beta}_kl{self.kl_coeff}_ent{self.ent_coeff}"
        true_joint_path = self.goat_dir / f"goat_true_joint_{model_name}.png"
        true_joint_fig.savefig(true_joint_path)
        
        evaluation_results = {
            'n_policies': self.n_policies,
            'latent_dim': latent_dim,
            'hidden_dim': hidden_dim,
            'beta': beta,
            'kl_coeff': self.kl_coeff,
            'ent_coeff': self.ent_coeff,
            'goat_reward': goat_reward,
            'true_joint': true_joint,
            'results': results
        }

        results_path = self.goat_dir / f"goat_eval_{model_name}.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(evaluation_results, f)
        
        return evaluation_results

def optimize_across_policies(betas, kl_coeffs, ent_coeffs,
                            goat_episodes=200, vae_epochs=1500, n_seeds=1):
    overall_best_reward = -float('inf')
    overall_best_params = None
    overall_best_eval = None
    all_evaluations = {}
    
    data_path = pathlib.Path("./results/comedi/cmg_s_suboptimal/8/comedi_cmg_s_suboptimal_8.pkl")
    
    if not data_path.exists():
        print(f"Error: No training data found for n_policies=8 at {data_path}")
        return None, None
    
    n_policies = 8
    hidden_dim = 128
    latent_dim = 4
    
    best_reward = -float('inf')
    
    for beta in betas:
        for kl_coeff in kl_coeffs:
            for ent_coeff in ent_coeffs:
                optimizer = VAEGOATOptimizer(
                    env_layout="cmg_s_suboptimal",
                    algorithm="comedi",
                    n_policies=n_policies,
                    goat_episodes=goat_episodes,
                    n_seeds=n_seeds,
                    vae_epochs=vae_epochs,
                    vae_batch_size=64,
                    vae_lr=0.01,
                    kl_coeff=kl_coeff,
                    ent_coeff=ent_coeff
                )
                
                exists, existing_eval = optimizer.check_evaluation_exists(latent_dim, hidden_dim, beta)
                
                try:
                    if exists and existing_eval is not None and 'goat_reward' in existing_eval:
                        evaluation = existing_eval
                    else:
                        evaluation = optimizer.evaluate_vae_with_goat(latent_dim, hidden_dim, beta)
                    
                    if evaluation is None:
                        continue
                    
                    param_key = (n_policies, latent_dim, hidden_dim, beta, kl_coeff, ent_coeff)
                    all_evaluations[param_key] = evaluation
                    
                    current_reward = evaluation['goat_reward']
                    
                    if current_reward > best_reward:
                        best_reward = current_reward
                    
                    if current_reward > (overall_best_eval['goat_reward'] if overall_best_eval else -float('inf')):
                        overall_best_reward = current_reward
                        overall_best_params = (n_policies, latent_dim, hidden_dim, beta, kl_coeff, ent_coeff)
                        overall_best_eval = evaluation
                
                except Exception as e:
                    print(f"Error evaluating n_policies={n_policies}, latent_dim={latent_dim}, hidden_dim={hidden_dim}, beta={beta}, kl_coeff={kl_coeff}, ent_coeff={ent_coeff}: {e}")
    
    results_dir = pathlib.Path("./results/goat") / "comedi" / "cmg_s_suboptimal"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    return overall_best_params, overall_best_eval


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    betas = [0.01, 0.05, 0.1, 0.2]
    kl_coeffs = [0.001, 0.01, 0.05, 0.1, 0.2]
    ent_coeffs = [0.001, 0.01, 0.05, 0.1, 0.2]
    
    best_params, best_evaluation = optimize_across_policies(betas=betas,
                                                            kl_coeffs=kl_coeffs,
                                                            ent_coeffs=ent_coeffs,
                                                            goat_episodes=200,
                                                            vae_epochs=1500,
                                                            n_seeds=1)