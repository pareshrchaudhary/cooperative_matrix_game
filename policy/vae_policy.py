import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from policy.vae import VAE

class VAEPolicy(nn.Module):
    def __init__(self, n_dim, latent_dim=16, hidden_dim=64, beta=0.01, lr=0.001, device=None):
        super(VAEPolicy, self).__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_dim = n_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.lr = lr
        self.current_z = None
        self.vae = VAE(self.n_dim, self.latent_dim, self.hidden_dim, self.beta, self.lr, self.device)

    def get_policy(self, z=None):
        if z is None:
            z = torch.randn(1, self.latent_dim, device=self.device)
            
        self.current_z = z
        policy = self.vae.decoder(z)
        policy = F.softmax(policy, dim=1).squeeze()
        return policy
    
    def get_action(self, z=None):
        policy = self.get_policy(z)
        return torch.argmax(policy).item()
    
    def get_current_embedding(self):
        return self.current_z
    
    def train(self, x_row, x_col, target):    
        total_loss, recon_loss, kl_loss = self.vae.train(x_row, x_col, target)
        return total_loss, recon_loss, kl_loss
    
    def save(self, path):
        self.vae.save(path)

    def load(self, path):
        self.vae.load(path)
