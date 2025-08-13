import torch
import numpy as np
import pathlib
import pickle
import argparse
from env import MatrixGame
from policy import Policy
from policy.vae_policy import VAEPolicy
from trainer import Trainer
from utils.visualize import Visualizer
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class RandomZ:
    def __init__(self, z_dim, batch_size=16, device=None):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.z_dim = z_dim
        self.batch_size = batch_size
    
    def get_z(self):
        z_prior = torch.distributions.Normal(
            torch.zeros((self.batch_size, self.z_dim), device=self.device),
            torch.ones((self.batch_size, self.z_dim), device=self.device)
        )
        
        z_sample = z_prior.sample()
        return z_sample

class GammaRunner:
    def __init__(self,
                 env_layout="cmg_s", 
                 n_episodes=100, 
                 policy_lr=0.05, 
                 policy_std=0.01,
                 latent_dim=16,
                 hidden_dim=128,
                 n_policies_vae=32,
                 algorithm="mep",
                 device=None):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env_layout = env_layout
        self.n_episodes = n_episodes
        self.policy_lr = policy_lr
        self.policy_std = policy_std
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_policies_vae = n_policies_vae
        self.algorithm = algorithm
        
        self.training_results = {}
        self.evaluation_results = {}
        
        self.regret_history = []
        self.vae_vs_vae_history = []

    def _custom_loss(self, br_policy, vae_policy, payoff_matrix, z):
        p_br_r, p_br_c = br_policy.get_policy()
        policy = vae_policy.get_policy(z)
        
        joint_br_r_vs_vae = torch.outer(p_br_r, policy)
        joint_vae_vs_br_c = torch.outer(policy, p_br_c)
        
        r1 = torch.sum(joint_br_r_vs_vae * payoff_matrix)
        r2 = torch.sum(joint_vae_vs_br_c * payoff_matrix)
        
        br_loss = -(r1 + r2)
        
        return br_loss
    
    def train_gamma(self, n_seeds):
        br_trainers = []
        BR_VAE_records = []
        VAE_embeddings = []
        random_zs = []
        
        env = MatrixGame(layout=self.env_layout, device=self.device)
        vae_policy = VAEPolicy(n_dim=env.n_dim,
                               latent_dim=self.latent_dim,
                               hidden_dim=self.hidden_dim,
                               device=self.device)
        
        # Get the project root directory
        project_root = pathlib.Path(__file__).resolve().parent.parent
        vae_model_path = project_root / "results" / "vae" / self.algorithm / self.env_layout / str(self.n_policies_vae) / "vae_policy.pt"
        if not vae_model_path.exists():
            raise FileNotFoundError(f"Pretrained VAE model not found at {vae_model_path}")
        vae_policy.load(vae_model_path)

        for seed in range(n_seeds):
            self._set_seed(seed)
            env = MatrixGame(layout=self.env_layout, device=self.device)
            local_payoff = env.payoff_matrix
            
            br_policy = Policy(n_dim=env.n_dim, lr=self.policy_lr, std_val=self.policy_std, device=self.device)
            
            random_z = RandomZ(
                z_dim=self.latent_dim,
                batch_size=16,
                device=self.device
            )
            
            trainer = Trainer(env, None, n_episodes=self.n_episodes)
            br_trainers.append((br_policy, vae_policy, trainer, local_payoff))
            random_zs.append(random_z)
            BR_VAE_records.append([])
            VAE_embeddings.append([])

        for step in range(self.n_episodes):
            vae_vs_vae_rewards = torch.zeros(1, device=self.device)
            vae_vs_br_rewards = torch.zeros(1, device=self.device)
            br_vs_vae_rewards = torch.zeros(1, device=self.device)
            
            for idx, ((br_policy, vae_policy, trainer, payoff), random_z) in enumerate(zip(br_trainers, random_zs)):
                z_sample = random_z.get_z()
                
                with torch.no_grad():
                    policies = torch.stack([vae_policy.get_policy(z_i.unsqueeze(0)) for z_i in z_sample])
                
                total_br_loss = torch.zeros(1, device=self.device)
                for z_i, policy_i in zip(z_sample, policies):
                    br_loss = self._custom_loss(br_policy, vae_policy, payoff, z_i.unsqueeze(0))
                    total_br_loss += br_loss

                br_loss = total_br_loss / z_sample.shape[0]
                
                br_policy.zero_grad()
                br_loss.backward()
                br_policy.step()
                
                p_br_r, p_br_c = br_policy.get_policy()
                
                with torch.no_grad():
                    batch_vae_vs_vae = torch.zeros(1, device=self.device)
                    batch_vae_vs_br = torch.zeros(1, device=self.device)
                    batch_br_vs_vae = torch.zeros(1, device=self.device)
                    
                    for policy in policies:
                        joint_policy = torch.outer(policy, policy)
                        batch_vae_vs_vae += torch.sum(joint_policy * payoff)
                        
                        vae_vs_br_joint = torch.outer(policy, p_br_c)
                        batch_vae_vs_br += torch.sum(vae_vs_br_joint * payoff)
                        
                        br_vs_vae_joint = torch.outer(p_br_r, policy)
                        batch_br_vs_vae += torch.sum(br_vs_vae_joint * payoff)
                    
                    vae_vs_vae_reward = batch_vae_vs_vae / z_sample.shape[0]
                    vae_vs_br = batch_vae_vs_br / z_sample.shape[0]
                    br_vs_vae = batch_br_vs_vae / z_sample.shape[0]
                    
                    total_br_vae_reward = br_vs_vae + vae_vs_br
                    
                    BR_VAE_records[idx].append(total_br_vae_reward.clone())
                    
                    z = vae_policy.get_current_embedding()
                    if z is not None:
                        VAE_embeddings[idx].append(z.clone())
                    
                    vae_vs_vae_rewards = vae_vs_vae_rewards + vae_vs_vae_reward
                    vae_vs_br_rewards = vae_vs_br_rewards + vae_vs_br
                    br_vs_vae_rewards = br_vs_vae_rewards + br_vs_vae
            
            vae_vs_vae_rewards = vae_vs_vae_rewards / n_seeds
            vae_vs_br_rewards = vae_vs_br_rewards / n_seeds
            br_vs_vae_rewards = br_vs_vae_rewards / n_seeds
            
            raw_regret = vae_vs_vae_rewards - (br_vs_vae_rewards)
            
            avg_br_vae = sum(record[-1].item() for record in BR_VAE_records) / n_seeds
            
            if step % 50 == 0 or step == self.n_episodes - 1:
                print(f"Step {step+1}/{self.n_episodes}, Avg BR vs VAE: {avg_br_vae:.4f}, Avg VAE vs VAE: {vae_vs_vae_rewards.item():.4f}")
                print(f"  VAE vs BR: {vae_vs_br_rewards.item():.4f}")
                print(f"  BR vs VAE: {br_vs_vae_rewards.item():.4f}")
                print(f"  Regret: {raw_regret.item():.4f}")
            
            self.regret_history.append(raw_regret.item())
            self.vae_vs_vae_history.append(vae_vs_vae_rewards.item())
        
        multi_seed_results = {}
        for seed in range(n_seeds):
            br_policy = br_trainers[seed][0].get_joint_policy().detach().cpu().numpy()
            seed_embeddings = VAE_embeddings[seed]
            multi_seed_results[seed] = {
                'BR_vs_VAE': [r.detach().cpu() for r in BR_VAE_records[seed]],
                'BR': br_policy,
                'VAE': [z.clone() for z in seed_embeddings]
            }
        
        return multi_seed_results
    
    def _set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def generate_true_joint(self, vae_policy, n_samples=10000):
        random_z = RandomZ(z_dim=self.latent_dim, batch_size=16, device=self.device)
        batch_size = random_z.batch_size
        joint = torch.zeros((vae_policy.n_dim, vae_policy.n_dim), device=self.device)
        for _ in range(n_samples // batch_size):
            z = random_z.get_z()
            for i in range(z.shape[0]):
                z_i = z[i].unsqueeze(0)
                with torch.no_grad():
                    policy = vae_policy.get_policy(z_i)
                    joint += torch.outer(policy, policy)
        
        return (joint / n_samples).cpu()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Gamma Policy')
    parser.add_argument('--env_layout', type=str, default="cmg_s_suboptimal",
                      help='Environment layout')
    parser.add_argument('--algorithm', type=str, default="comedi",
                      choices=['mep', 'comedi'],
                      help='Algorithm to use')
    parser.add_argument('--n_policies_vae', type=int, default=8,
                      help='Number of policies')
    parser.add_argument('--latent_dim', type=int, default=4,
                      help='Latent dimension size')
    parser.add_argument('--hidden_dim', type=int, default=128,
                      help='Hidden dimension size')
    parser.add_argument('--n_episodes', type=int, default=200,
                      help='Number of episodes')
    parser.add_argument('--n_seeds', type=int, default=1,
                      help='Number of seeds')
    parser.add_argument('--policy_lr', type=float, default=0.05,
                      help='Policy learning rate')
    parser.add_argument('--policy_std', type=float, default=0.01,
                      help='Policy standard deviation')
    
    args = parser.parse_args()
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    env_layout = args.env_layout
    algorithm = args.algorithm
    n_policies_vae = args.n_policies_vae
    latent_dim = args.latent_dim
    hidden_dim = args.hidden_dim
    n_episodes = args.n_episodes
    n_seeds = args.n_seeds
    policy_lr = args.policy_lr
    policy_std = args.policy_std
    print(f"Training Gamma with BR policy for {n_episodes} episodes, and {n_seeds} seeds.")
    env = MatrixGame(layout=env_layout)
    visualizer = Visualizer(env)

    runner = GammaRunner(env_layout=env_layout,
                        n_episodes=n_episodes,
                        policy_lr=policy_lr,
                        policy_std=policy_std,
                        latent_dim=latent_dim,
                        hidden_dim=hidden_dim,
                        n_policies_vae=n_policies_vae,
                        algorithm=algorithm)
    results = runner.train_gamma(n_seeds=n_seeds)
    
    BR_VAE_values = [results[seed]['BR_vs_VAE'][-1].detach().cpu().numpy() for seed in range(n_seeds)]
    avg_br_vae = np.mean(BR_VAE_values)
    print(f"Average BR vs VAE reward: {avg_br_vae:.4f}")
    
    vae_model_path = pathlib.Path("./results/vae") / runner.algorithm / env_layout / str(runner.n_policies_vae) / "vae_policy.pt"
    vae_policy = VAEPolicy(
        n_dim=env.n_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        device=runner.device
    )
    vae_policy.load(vae_model_path)
    
    # Generate and visualize the true joint policy
    true_joint = runner.generate_true_joint(vae_policy)
    true_joint_fig = visualizer.visualize_pool(true_joint.unsqueeze(0), aggregation='avg', normalize=True)
    plt.show()
    
    # Save results
    project_root = pathlib.Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    output_dir = results_dir / "gamma" / algorithm / env_layout / str(n_policies_vae)
    output_dir.mkdir(parents=True, exist_ok=True)

    pickle_path = output_dir / f"gamma.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)
    
    # Save true joint policy visualization
    true_joint_path = output_dir / f"gamma_true_joint.png"
    true_joint_fig.savefig(true_joint_path)
    print(f"Results saved to {output_dir}")
