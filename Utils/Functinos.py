import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

from Utils.Options import Param
# from Utils.ErrorLogger import error_logger


class CkpHandler(Param):
    def __init__(self):
        super(CkpHandler, self).__init__()

    def init_weight(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
            if module.bias is not None:
                nn.init.constant(module.bias.data, 0.0)

        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            if module.bias is not None:
                nn.init.constant(module.bias.data, 0.0)

        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
            if module.bias is not None:
                nn.init.constant(module.bias.data, 0.0)

    def load_ckp(self, model, optimizer, scheduler, imagenet=False):
        if imagenet:
            print(f'BackBone Pretrained [IMAGE NET] Loading..')
            return model, optimizer, scheduler, 0

        else:
            if self.do_ckp_load:
                print(f'Check Point [{self.load_ckp_epoch}] Loading...')
                ckp = torch.load(f'{self.output_ckp}/{self.load_ckp_epoch}.pth')
                model.load_state_dict(ckp['model_state_dict'])
                optimizer.load_state_dict(ckp['optimizer_state_dict'])
                scheduler.load_state_dict(ckp['scheduler_state_dict'])
                epoch = ckp['epoch'] + 1

            else:
                print(f'Initialize Model Weight...')
                model.apply(self.init_weight)
                epoch = 0

            return model, optimizer, scheduler, epoch

    def save_ckp(self, model, optimizer, scheduler, epoch):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
            },
            os.path.join(f'{self.output_ckp}', f'{epoch}.pth')
        )


class TransformBuilder(Param):
    def __init__(self):
        super(TransformBuilder, self).__init__()

    def set_train_transform(self, do_aug=False):
        assert type(do_aug) is bool, 'Only boolean type is available for self.AUG.'

        if do_aug:
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop((self.input_size, self.input_size), scale=(0.7, 1), ratio=(0.5, 2)),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

        return transform