import os
import torch
from src.utils.logger import logging

class Checkpointer:
    def __init__(self, save_dir, save_period=1, resume=False, **kwargs):
        self.save_dir = save_dir
        self.save_period = save_period
        self.resume = resume

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
    def save(self, epoch, model, optimizer, lr_scheduler=None, history=None):
        # save necessary configs
        checkpoint_data = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        # save plugin configs
        if lr_scheduler:
            checkpoint_data['scheduler'] = lr_scheduler.state_dict()
        
        # save remaining kwargs
        if history:
            checkpoint_data["history"] = history

        # save checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'epoch_{epoch}.pth')
        torch.save(checkpoint_data, checkpoint_path)
        logging.info(f'Saved checkpoint to {checkpoint_path}')
    
    def load(self, checkpoint_path, model, optimizer, lr_scheduler=None, history=None):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if lr_scheduler:
            lr_scheduler.load_state_dict(checkpoint['scheduler'])
        if history:
            history = checkpoint["history"]
        logging.info(f'Loaded checkpoint from {checkpoint_path}')
        return checkpoint['epoch']
    