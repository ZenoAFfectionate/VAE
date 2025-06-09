import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """ Implementation of Variational Auto-Encoder """

    def __init__(self, input_dim=784, hidden_dims=[512, 256, 128],
                 latent_dim=32, beta=1.0, capacity=0.0, capacity_iters=0):
        ''' initalize variational auto encoder '''
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.capacity = capacity
        self.capacity_iters = capacity_iters
        self.current_iter = 0
        
        # construct encoder (data to feature)
        encoder_layers = []
        prev_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.LayerNorm(h_dim))
            encoder_layers.append(nn.ELU(inplace=True))
            prev_dim = h_dim
        # create sequential
        self.encoder = nn.Sequential(*encoder_layers)
        
        # mean and variance of latent space
        self.fc_miu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # construct decoder (feature to data)
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.LayerNorm(h_dim))
            decoder_layers.append(nn.ELU(inplace=True))
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        decoder_layers.append(nn.Sigmoid())
        # create sequential
        self.decoder = nn.Sequential(*decoder_layers)
        
        # initialize the weight of the entire model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


    def encode(self, x):
        ''' encode data x into feature z ''' 
        h = self.encoder(x)
        miu = self.fc_miu(h)
        var = self.fc_var(h)
        return miu, var
    
    def reparameterize(self, miu, var):
        ''' reparameterization trick '''
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        return miu + eps * std

    def decode(self, z):
        ''' decode feature z into data x '''
        return self.decoder(z)


    def forward(self, x):
        miu, var = self.encode(x)
        z = self.reparameterize(miu, var)
        recon_x = self.decode(z)
        return recon_x, miu, var


    def loss_function(self, recon_x, x, miu, var):
        ''' calculate VAE loss '''
        if self.training: self.current_iter += 1

        # calculate reconstruct loss
        rc_loss = F.mse_loss(recon_x, x, reduction='sum')

        # calculate KL-divergence
        kl_loss = -0.5 * torch.sum(1 + var - miu.pow(2) - var.exp())

        # capacity constraint (lift the upper bound of KL gradually)
        if self.capacity > 0 and self.capacity_iters > 0:
            capacity = min(self.capacity * self.current_iter / self.capacity_iters, 
                          self.capacity)
            kl_loss = torch.abs(kl_loss - capacity)

        total_loss = rc_loss + self.beta * kl_loss

        loss_dict = {
            'total_loss': total_loss.item(),
            'rc_loss': rc_loss.item(),  # reconstruction loss
            'kl_loss': kl_loss.item(),  # kl-divergence  loss
            'beta': self.beta
        }

        return total_loss, loss_dict


    def sample(self, num_samples, device):
        ''' generate sample from latent space '''
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z)
    
    def reconstruct(self, x):
        ''' reconstruct data from latent feature '''
        with torch.no_grad():
            recon_x, _, _ = self.forward(x)
        return recon_x
