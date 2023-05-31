import os
import re
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from Utils.Options import Param
from Utils.Displayer import LossDisPlayer
from Utils.Functinos import CkpHandler, TransformBuilder
from Utils.CustomDataset import MakeDataset

from Model.ConvMixer import ConvMixer

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Trainer(Param):
    def __init__(self):
        super(Trainer, self).__init__()

        os.makedirs(self.output_ckp, exist_ok=True)
        os.makedirs(self.output_log, exist_ok=True)

        self.tr_disp = LossDisPlayer(['tr_acc', 'tr_loss'])
        self.te_disp = LossDisPlayer(['te_acc', 'te_loss'])

        self.CkpHandler = CkpHandler()
        self.TransformBuilder = TransformBuilder()

    def run(self):
        print('--------------------------------------------------')
        print(f'[DEVICE] : {self.device}')
        print('--------------------------------------------------')

        model = ConvMixer(dim=self.dim, depth=self.depth, kernel_size=self.kernel_size, patch_size=self.patch_size)
        model = model.to(self.device)

        optimizer = optim.AdamW(list(model.parameters()), lr=self.lr)
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=self.lr, max_lr=0.1, step_size_up=50, step_size_down=100,
            cycle_momentum=False, mode="triangular")

        model, optimizer, scheduler, epoch = self.CkpHandler.load_ckp(model, optimizer, scheduler, imagenet=False)

        criterion = nn.CrossEntropyLoss()

        transform = self.TransformBuilder.set_train_transform(do_aug=True)
        tr_dataset = MakeDataset(self.dataset_path, self.data_folder[0], self.cls_folder, transform)
        te_dataset = MakeDataset(self.dataset_path, self.data_folder[1], self.cls_folder, transform)

        summary = SummaryWriter(self.output_log)

        for ep in range(epoch, self.full_epoch):
            tr_loader = DataLoader(dataset=tr_dataset, batch_size=self.batch_size, shuffle=True)
            te_loader = DataLoader(dataset=te_dataset, batch_size=2, shuffle=False)

            model.train()
            for idx, (item, label) in enumerate(tqdm.tqdm(tr_loader, desc=f'Train Epoch [{ep}/{self.full_epoch}]')):
                item = item.to(self.device)
                label = label.to(self.device)

                output = model(item)
                loss = criterion(output, label)

                acc = (output.argmax(1) == label).type(torch.float).sum().item()

                self.tr_disp.record([loss, acc])

                optimizer.zero_grad()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                for idx, (item, label) in enumerate(tqdm.tqdm(te_loader, desc=f'Test Epoch [{ep}/{self.full_epoch}]')):
                    item = item.to(self.device)
                    label = label.to(self.device)

                    output = model(item)
                    loss = criterion(output, label)

                    acc = (output.argmax(1) == label).type(torch.float).sum().item()

                    self.te_disp.record([loss, acc])

            scheduler.step()
            tr_acc_loss = self.tr_disp.get_avg_losses(len(tr_dataset))
            te_acc_loss = self.te_disp.get_avg_losses(len(te_dataset))

            print(f'==> [{ep}/{self.full_epoch}] || Train Accuracy : {tr_acc_loss[1]} | Train Loss : {tr_acc_loss[0]} | '
                  f'Test Accuracy : {te_acc_loss[1]} | Test Loss : {te_acc_loss[0]} ||')

            summary.add_scalar("train/Loss", tr_acc_loss[0], ep)
            summary.add_scalar("train/Acc", tr_acc_loss[1], ep)

            summary.add_scalar("test/Loss", te_acc_loss[0], ep)
            summary.add_scalar("test/Acc", te_acc_loss[1], ep)

            self.tr_disp.reset()
            self.te_disp.reset()

            self.CkpHandler.save_ckp(model, optimizer, scheduler, ep)
