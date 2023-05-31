import torch
import tqdm

from Utils.Options import Param


class OneEpoch(Param):
    def __init__(self, do_train=True):
        super(OneEpoch, self).__init__()
        self.do_train = do_train

    def train_run(self, ep, dataloader, model, criterion, optimizer, disp):
        if self.do_train:
            run_type = 'Train'
        else:
            run_type = 'Test'

        for idx, (item, label) in enumerate(tqdm.tqdm(dataloader, desc=f'{run_type} Epoch [{ep}/{self.full_epoch}]')):
            item = item.to(self.device)
            label = label.to(self.device)

            output = model(item)
            loss = criterion(output, label)

            acc = (output.argmax(1) == label).type(torch.float).sum().item()

            disp.record([loss, acc])

            if self.do_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            disp.display_value(do_display=self.do_display)

        return disp.get_avg_losses()

    def __call__(self, ep, dataloader, model, criterion, optimizer, disp):
        return self.train_run(ep, dataloader, model, criterion, optimizer, disp)