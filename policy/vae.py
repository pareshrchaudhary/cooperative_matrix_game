import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dim=64):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_log_var = nn.Linear(hidden_dim, z_dim)

    def forward(self, x_row, x_col):
        x = torch.cat([x_row, x_col], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, z_dim, output_dim, hidden_dim=64):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim) 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class VAE(nn.Module):
    def __init__(self, n_dim, z_dim=4, hidden_dim=64, beta=0.1, lr=0.001, device=None):
        super(VAE, self).__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = n_dim  
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.beta = beta
        
        self.encoder = Encoder(self.input_dim, self.z_dim, self.hidden_dim).to(self.device)
        self.decoder = Decoder(self.z_dim, self.input_dim, self.hidden_dim).to(self.device)
        
        self.parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(self.parameters, lr=lr)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x_row, x_col):
        x_row = x_row.to(self.device)
        x_col = x_col.to(self.device)
        mu, log_var = self.encoder(x_row, x_col)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var

    def loss_function(self, x, x_hat, mu, log_var):
        x = x.to(self.device)
        batch_size = x.size(0)
        recon_loss = -torch.sum(x * F.log_softmax(x_hat, dim=1)) / batch_size
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size
        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, recon_loss, kl_loss

    def sample(self, mu, std, num_samples=1):
        z = mu + torch.randn(num_samples, self.z_dim, device=self.device) * std
        return self.decoder(z)
    
    def train(self, x_row_batch, x_col_batch, target_batch):
        self.optimizer.zero_grad()
        x_hat, mu, log_var = self.forward(x_row_batch, x_col_batch)
        total_loss, recon_loss, kl_loss = self.loss_function(target_batch, x_hat, mu, log_var)
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item(), recon_loss.item(), kl_loss.item()
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))

    