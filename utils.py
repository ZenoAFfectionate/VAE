import os
import torch
from tqdm import tqdm
from torchvision.utils import save_image

from torchvision.utils import save_image
from torch.nn.utils import clip_grad_norm_


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def train(model, device, train_loader, optimizer, epoch, total_epochs):
    model.train()
    loss_meter  = AverageMeter('TotalLoss')
    rc_meter = AverageMeter('RCloss')
    kl_meter = AverageMeter('KLloss')
    
    train_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{total_epochs} [Train]')
    
    for data, _ in train_bar:
        data = data.to(device)
        batch_size = data.size(0)
        
        recon_batch, miu, var = model(data)
        
        loss, loss_dict = model.loss_function(recon_batch, data, miu, var)
        
        optimizer.zero_grad()
        loss.backward()
        
        # gradient clip 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # update statistic information
        loss_meter.update(loss_dict['total_loss'], batch_size)
        rc_meter.update(loss_dict['rc_loss'], batch_size)
        kl_meter.update(loss_dict['kl_loss'], batch_size)
        
        # update progress bar
        train_bar.set_postfix({
            'loss': loss_meter.avg,
            'rc': rc_meter.avg,
            'kl': kl_meter.avg
        })
    
    return loss_meter.avg, rc_meter.avg, kl_meter.avg


def valid(model, device, valid_loader, epoch, total_epochs):
    model.eval()
    loss_meter  = AverageMeter('TotalLoss')
    rc_meter = AverageMeter('RCloss')
    kl_meter = AverageMeter('KLloss')
    
    with torch.no_grad():

        valid_bar = tqdm(valid_loader, desc=f'Epoch {epoch}/{total_epochs} [Valid]')
        
        for data, _ in valid_bar:
            data = data.to(device)
            batch_size = data.size(0)

            recon, miu, var = model(data)
            
            loss, loss_dict = model.loss_function(recon, data, miu, var)
            
            # update statistic information
            loss_meter.update(loss_dict['total_loss'], batch_size)
            rc_meter.update(loss_dict['rc_loss'], batch_size)
            kl_meter.update(loss_dict['kl_loss'], batch_size)
            
            # update progress bar
            valid_bar.set_postfix({
                'loss': loss_meter.avg,
                'rc': rc_meter.avg,
                'kl': kl_meter.avg
            })
    
    return loss_meter.avg, rc_meter.avg, kl_meter.avg



def save_comparison(original, reconstructed, epoch, output_dir, prefix='recon'):
    """ store origin image and reconstruct image comparison """
    comparison = torch.cat([
        original.view(-1, 1, 28, 28)[:8],
        reconstructed.view(-1, 1, 28, 28)[:8]
    ])
    
    save_path = os.path.join(output_dir, f'{prefix}_comparison_epoch_{epoch}.png')
    save_image(comparison.cpu(), save_path, nrow=8)


def save_samples(samples, epoch, output_dir, prefix='gen'):
    """store generated sample from VAE """
    save_path = os.path.join(output_dir, f'{prefix}_samples_epoch_{epoch}.png')
    save_image(samples.view(-1, 1, 28, 28).cpu(), save_path, nrow=8)



