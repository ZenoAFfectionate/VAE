import os
import time
import torch
import warnings
import argparse

import torch.optim as optim

from model.vae import VAE
from utils import train, valid
from data.load_data import get_dataset


warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"==> Using device: {device}")

parser = argparse.ArgumentParser(description='Variational AutoEncoder')

# model parameters
parser.add_argument('--hidden-dims', nargs='+', type=int, default=[512, 256])
parser.add_argument('--latent-dim',  type=int,  default=32, help='latent space dimension')
parser.add_argument('--beta', type=float, default=1.0, help='KL-divergence weight')
parser.add_argument('--capacity', type=float, default=0.0,   help='capacity upper bound')
parser.add_argument('--capacity-iters', type=int, default=0, help='bound reach iter limit')

# training parameters
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--weight-decay', type=float, default=1e-5)

args = parser.parse_args()


# loading data
print(f'==> Loading training and testing data ...', end=' ')
train_loader, valid_loader = get_dataset(args.dataset, args.batch_size)
print(f'Finish')


# initialize model
print(f'==> Initialize Variation AutoEncoder Model ...', end=' ')
model = VAE(
        input_dim=...,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        beta=args.beta,
        capacity=args.capacity,
        capacity_iters=args.capacity_iters
    ).to(device)
print(f'Finish')

# initialize optimizer
print(f'==> Initialize AdamW Optimizer for VAE ...', end=' ')
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
print(f'Finish')

# initialize scheduler
print(f'==> Initialize Plateau Scheduler for VAE ...', end=' ')
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.gamma, patience=args.patience,
            verbose=True, min_lr=1e-6)
print(f'Finish')


history = {  # record loss during running
    'train_loss': [], 'train_rc': [], 'train_kl': [],
    'valid_loss': [], 'valid_rc': [], 'valid_kl': [],
    'lr': []
}
best_loss = float('inf')

print(f'==> Start training the VAE model')
for epoch in range(1, args.epochs + 1):
    start_time = time.time()

    # train the model
    train_loss, train_rc, train_kl = train(
        model, device, train_loader, optimizer, epoch, args.epochs
    )
    history['train_loss'].append(train_loss)
    history['train_rc'].append(train_rc)
    history['train_kl'].append(train_kl)

    # valid the model
    valid_loss, valid_rc, valid_kl = valid(
        model, device, valid_loader, epoch, args.epochs
    )
    history['valid_loss'].append(valid_loss)
    history['valid_rc'].append(valid_rc)
    history['valid_kl'].append(valid_kl)

    # get current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    history['lr'].append(current_lr)

    scheduler.step(valid_loss)  # update learning rate

    # store best model
    if valid_loss < best_loss:
        print(f'Best result (loss = {valid_loss:.4f}), saving the model ...', end=' ')
        best_loss = valid_loss
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_loss": best_loss,
            "epoch": epoch
        }
        os.makedirs(f'checkpoint/VAE', exist_ok=True)
        torch.save(state, './checkpoint/VAE/{args.dataset}_best')
        print('Finish')
    
    # record
    epoch_time = time.time() - start_time
    content = time.ctime() + ' ' + f"Epoch {epoch}/{args.epochs} | Time: {epoch_time:.2f}s |\n" \
              f"Train Loss: {train_loss:.6f} (Recon: {train_rc:.6f}, KL: {train_kl:.6f}) | \n"  \
              f"Test  Loss: {valid_loss:.6f} (Recon: {valid_rc:.6f}, KL: {valid_kl:.6f}) | \n"
    print(content)
    os.makedirs(f'log/VAE', exist_ok=True)
    with open(f'log/VAE/{args.dataset}.txt', 'a') as appender:
        appender.write(content + "\n")

print(f'==> Training finish')


# visualize the training result

