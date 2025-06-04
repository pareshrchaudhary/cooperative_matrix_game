import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import pathlib
import pickle
import random
from sklearn.manifold import TSNE
import umap
from env import MatrixGame
from policy.vae_policy import VAEPolicy
from utils.visualize import Visualizer
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch.nn.functional as F

class VAEPolicyTrainer:
    SUPPORTED_ALGORITHMS = ['mep', 'comedi', 'sp', 'trajedi']
    
    def __init__(self,
                 env_layout="cmg_s",
                 algorithm="comedi",
                 n_policies=8,
                 latent_dim=10,
                 hidden_dim=64,
                 beta=0.1,
                 vae_lr=0.001,
                 n_epochs=200,
                 batch_size=8,
                 n_seeds=None,
                 augment_data=False,
                 device=None):
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(f"Algorithm must be one of {self.SUPPORTED_ALGORITHMS}")
            
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env_layout = env_layout
        self.algorithm = algorithm
        self.n_policies = n_policies
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.vae_lr = vae_lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_seeds = n_seeds
        self.augment_data = augment_data
        
        self.env = MatrixGame(layout=self.env_layout, device=self.device)
        self.n_dim = self.env.n_dim
        
        self.vae_policy = VAEPolicy(
            n_dim=self.n_dim, 
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            beta=self.beta,
            lr=self.vae_lr,
            device=self.device
        )
        
        self.visualizer = Visualizer(self.env)
        
        self.project_root = pathlib.Path(__file__).resolve().parent.parent
        self.results_dir = self.project_root / "results"
        self.algorithm_dir = self.results_dir / self.algorithm
        self.env_layout_dir = self.algorithm_dir / self.env_layout
        self.policy_dir = self.env_layout_dir / str(self.n_policies)
        self.vae_dir = self.results_dir / "vae"
        self.vae_algorithm_dir = self.vae_dir / self.algorithm
        self.vae_env_layout_dir = self.vae_algorithm_dir / self.env_layout
        self.output_dir = self.vae_env_layout_dir / str(self.n_policies)
        
        self._ensure_directories()
        
        self.base_filename = f"{self.algorithm}_{self.env_layout}_{self.n_policies}"
        self.data_path = self.policy_dir / f"{self.base_filename}.pkl"
        self.model_path = self.output_dir / f"vae_policy.pt"
        
    def _ensure_directories(self):
        directories = [
            self.policy_dir,
            self.output_dir
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
    def load_training_data(self):
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        
        row_policies = []
        col_policies = []
        
        if isinstance(data, dict) and all(isinstance(v, dict) and 'pool_start' in v and 'pool_mid' in v and 'pool_end' in v for v in data.values()):
            seeds = list(data.keys())
            if self.n_seeds is not None:
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
        
        self.row_data = torch.stack(row_policies).to(self.device)
        self.col_data = torch.stack(col_policies).to(self.device)
        
        if self.augment_data:
            original_count = len(row_policies)
            row_policies_aug = []
            col_policies_aug = []
            
            indices = torch.randperm(len(self.row_data))
            for i in range(len(self.row_data)):
                j = (i + 1 + torch.randint(0, len(self.row_data) - 1, (1,)).item()) % len(self.row_data)
                
                row_policies_aug.append(self.row_data[i])
                col_policies_aug.append(self.col_data[j])
            
            self.row_data = torch.cat([self.row_data, torch.stack(row_policies_aug)])
            self.col_data = torch.cat([self.col_data, torch.stack(col_policies_aug)])
            
            print(f"Data augmentation: Added {len(row_policies_aug)} mismatched pairs (total: {len(self.row_data)})")
        
        print(f"Loaded {len(row_policies)} policies for training (from start, mid, and end stages)")
        return self.row_data, self.col_data
    
    def train_vae_policy(self, row_data, col_data):
        for epoch in range(self.n_epochs):
            epoch_total_loss = 0
            epoch_recon_loss = 0
            epoch_kl_loss = 0
            num_batches = 0
            
            indices = torch.randperm(len(row_data))
            for i in range(0, len(row_data), self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                row_batch = row_data[batch_indices]
                col_batch = col_data[batch_indices]
            
                total_loss, recon_loss, kl_loss = self.vae_policy.train(row_batch, col_batch, row_batch)
                
                epoch_total_loss += total_loss
                epoch_recon_loss += recon_loss
                epoch_kl_loss += kl_loss
                num_batches += 1
            
            avg_total_loss = epoch_total_loss / num_batches
            avg_recon_loss = epoch_recon_loss / num_batches
            avg_kl_loss = epoch_kl_loss / num_batches
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs}:")
                print(f"  Total: {avg_total_loss:.6f}, Recon: {avg_recon_loss:.6f}, KL: {avg_kl_loss:.6f}")
    
    def evaluate_vae(self, row_data, col_data):
        with torch.no_grad():
            n_batches = (len(row_data) + self.batch_size - 1) // self.batch_size
            
            all_policies = []
            all_mus = []
            all_log_vars = []
            recon_errors = []
            
            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(row_data))
                
                row_batch = row_data[start_idx:end_idx]
                col_batch = col_data[start_idx:end_idx]
                
                x_hat, mu, log_var = self.vae_policy.vae.forward(row_batch, col_batch)
                
                recon_error = torch.mean((row_batch - x_hat) ** 2, dim=1)
                
                kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
                
                all_policies.append(x_hat)
                all_mus.append(mu)
                all_log_vars.append(log_var)
                recon_errors.append(recon_error)
            
            policies = torch.cat(all_policies, dim=0)
            mus = torch.cat(all_mus, dim=0)
            log_vars = torch.cat(all_log_vars, dim=0)
            recon_errors = torch.cat(recon_errors, dim=0)
            kl_divs = -0.5 * torch.sum(1 + log_vars - mus.pow(2) - log_vars.exp(), dim=1)
            
            print("\nReconstruction Statistics:")
            print(f"Mean Reconstruction Error: {recon_errors.mean():.4f} ± {recon_errors.std():.4f}")
            print(f"Max Reconstruction Error: {recon_errors.max():.4f}")
            print(f"Min Reconstruction Error: {recon_errors.min():.4f}")
            
            print("\nLatent Space Statistics:")
            print(f"Mean KL Divergence: {kl_divs.mean():.4f} ± {kl_divs.std():.4f}")
            print(f"Mean of mu: {mus.mean():.4f} ± {mus.std():.4f}")
            print(f"Mean of log_var: {log_vars.mean():.4f} ± {log_vars.std():.4f}")
            
            return (policies, mus, log_vars, recon_errors, kl_divs)
    
    def visualize_sample_policies(self, row_data, col_data, indices):
        for i, idx in enumerate(indices):
            p_r = row_data[idx]
            p_c = col_data[idx]
            
            with torch.no_grad():
                policy, mu, log_var = self.vae_policy.vae.forward(p_r.unsqueeze(0), p_c.unsqueeze(0))
                
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                z = mu + eps * std
                
                policy_recon = self.vae_policy.get_policy(z)
            
            self.visualizer.visualize_policy_distributions(p_r, policy_recon, idx, p_c, self.n_epochs)
    
    def visualize_dataset_policies(self, n_samples=None):
        all_policies = []
        with torch.no_grad():
            n_batches = (len(self.row_data) + self.batch_size - 1) // self.batch_size
            
            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(self.row_data))
                
                row_batch = self.row_data[start_idx:end_idx]
                col_batch = self.col_data[start_idx:end_idx]
                
                x_hat, _, _ = self.vae_policy.vae.forward(row_batch, col_batch)
                
                for policy in x_hat:
                    policy = F.softmax(policy, dim=0)
                    joint_policy = torch.outer(policy, policy)
                    all_policies.append(joint_policy)
            
        policies = torch.stack(all_policies)
        
        fig = self.visualizer.visualize_pool(policies)
        
        return fig
    
    def visualize_latent_space(self, method='UMAP', num_samples=1000):
        latents = []
        indices = torch.randperm(len(self.row_data))[:num_samples]
        
        with torch.no_grad():
            for idx in indices:
                p_r = self.row_data[idx].unsqueeze(0)
                p_c = self.col_data[idx].unsqueeze(0)
                
                _, mu, _ = self.vae_policy.vae.forward(p_r, p_c)
                
                latents.append(mu.cpu())
        
        latents = torch.cat(latents, dim=0).numpy()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if method == 'TSNE':
            n_samples = len(latents)
            perplexity = min(30, max(5, n_samples - 1))
            
            if n_samples < 10:
                print(f"Warning: Small dataset size ({n_samples}). Using perplexity={perplexity} for TSNE.")
            
            tsne = TSNE(n_components=2, perplexity=perplexity, verbose=1)
            results = tsne.fit_transform(latents)
            
            scatter = ax.scatter(results[:, 0], results[:, 1], c=indices, cmap='viridis')
            ax.set_title('VAE Latent Space with TSNE')
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            
        elif method == 'UMAP':
            reducer = umap.UMAP()
            results = reducer.fit_transform(latents)
            
            scatter = ax.scatter(results[:, 0], results[:, 1], c=indices, cmap='viridis')
            ax.set_title('VAE Latent Space with UMAP')
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
        else:
            raise ValueError("method should be 'TSNE' or 'UMAP'")
        
        plt.colorbar(scatter, ax=ax, label='Sample Index')
        
        plt.tight_layout()
        return fig
        
    def visualize_joint_policy_manifold(self, scale=1.0, n_points=15, figsize=8):
        grid_x = np.linspace(-scale, scale, n_points)
        grid_y = np.linspace(-scale, scale, n_points)[::-1]

        with torch.no_grad():
            policy_size = self.n_dim
            pad = 2
            canvas_size = policy_size * n_points + pad * (n_points - 1)
            figure = np.ones((canvas_size, canvas_size)) * np.nan

            for i, var_val in enumerate(grid_y):
                for j, mean_val in enumerate(grid_x):
                    z = torch.zeros(1, self.latent_dim, device=self.device)
                    z[0, 0] = mean_val
                    z[0, 1] = var_val

                    policy = self.vae_policy.vae.decoder(z)
                    policy = F.softmax(policy, dim=1).squeeze()
                    joint_policy = torch.outer(policy, policy)
                    joint_policy_np = joint_policy.cpu().numpy()

                    row_start = i * (policy_size + pad)
                    col_start = j * (policy_size + pad)
                    figure[
                        row_start : row_start + policy_size,
                        col_start : col_start + policy_size
                    ] = joint_policy_np

        extent = [-scale, scale, -scale, scale]
        fig, ax = plt.subplots(figsize=(figsize, figsize))
        im = ax.imshow(figure, cmap='viridis', extent=extent, interpolation='none', origin='upper')
        ax.set_xlabel('mean, z[0]')
        ax.set_ylabel('var, z[1]')
        ax.set_xticks(np.linspace(-scale, scale, 9))
        ax.set_yticks(np.linspace(-scale, scale, 9))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label('Joint Probability')

        plt.tight_layout()
        return fig
    
    def visualize_average_random_policy(self, n_samples=10000):
        accumulated_policy = torch.zeros((self.n_dim, self.n_dim), device=self.device)
        
        batch_size = 100
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                current_batch_size = end_idx - start_idx
                
                z_batch = torch.randn((current_batch_size, self.latent_dim), device=self.device)
                
                policies = self.vae_policy.vae.decoder(z_batch)
                policies = F.softmax(policies, dim=1)
                
                for policy in policies:
                    joint_policy = torch.outer(policy, policy)
                    accumulated_policy += joint_policy
        
        average_policy = accumulated_policy / n_samples
        
        average_policy_fig = self.visualizer.visualize_pool(average_policy.unsqueeze(0))
        
        return average_policy, average_policy_fig
    
    def save_model(self):
        self.vae_policy.save(self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No model found at {self.model_path}")
        self.vae_policy.load(self.model_path)
        print(f"Model loaded from {self.model_path}")

if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    parser = argparse.ArgumentParser(description='Train VAE Policy')
    parser.add_argument('--env_layout', type=str, default="cmg_s_suboptimal",
                      help='Environment layout')
    parser.add_argument('--algorithm', type=str, default="comedi",
                      choices=['mep', 'comedi'],
                      help='Algorithm to use')
    parser.add_argument('--n_policies', type=int, default=8,
                      help='Number of policies')
    parser.add_argument('--latent_dim', type=int, default=4,
                      help='Latent dimension size')
    parser.add_argument('--beta', type=float, default=0.2,
                      help='Beta parameter for VAE')
    parser.add_argument('--hidden_dim', type=int, default=128,
                      help='Hidden dimension size')
    parser.add_argument('--vae_lr', type=float, default=0.01,
                      help='Learning rate for VAE')
    parser.add_argument('--n_epochs', type=int, default=2000,
                      help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch size')
    parser.add_argument('--n_seeds', type=int, default=1,
                      help='Number of seeds')
    parser.add_argument('--augment_data', action='store_true', default=True,
                      help='Whether to augment data')
    args = parser.parse_args()
    
    # Step 1: Train VAEPolicy
    vae_trainer = VAEPolicyTrainer(
        env_layout=args.env_layout,
        algorithm=args.algorithm,
        n_policies=args.n_policies,
        latent_dim=args.latent_dim,
        beta=args.beta,
        hidden_dim=args.hidden_dim,
        vae_lr=args.vae_lr,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        n_seeds=args.n_seeds,
        augment_data=args.augment_data
    )
    row_data, col_data = vae_trainer.load_training_data()
    vae_trainer.train_vae_policy(row_data, col_data)
    
    # Step 2.1: Collct VAE statistics
    (total_policies, total_mus, total_log_vars, total_recon_errors, total_kl_divs) = vae_trainer.evaluate_vae(row_data, col_data)
    
    # Step 2.2: Visualize reconstruction of sample policies
    num_samples = 3
    indices = random.sample(range(len(row_data)), min(num_samples, len(row_data)))
    vae_trainer.visualize_sample_policies(row_data, col_data, indices)
    plt.close("all")

    # Step 2.3: Visualize aggregated reconstruction
    data_reconstruction = vae_trainer.visualize_dataset_policies()
    data_reconstruction.savefig(vae_trainer.output_dir / "data_recon.png")
    plt.close(data_reconstruction)
    
    # Step 2.4: Visualize latent space
    vae_latent_space = vae_trainer.visualize_latent_space(method='TSNE', num_samples=1000)
    vae_latent_space.savefig(vae_trainer.output_dir / "vae_latent_space.png")
    plt.close(vae_latent_space)
    
    # Step 2.5: Visualize joint policy manifold
    manifold = vae_trainer.visualize_joint_policy_manifold(scale=2.0, n_points=15, figsize=8)
    manifold.savefig(vae_trainer.output_dir / "joint_policy_manifold.png")
    plt.close(manifold)
    vae_trainer.save_model()

    _, average_random_policy = vae_trainer.visualize_average_random_policy(n_samples=10000)
    average_random_policy.savefig(vae_trainer.output_dir / "average_random_policy.png")
    plt.close(average_random_policy)