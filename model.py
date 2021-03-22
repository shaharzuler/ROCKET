import math
import random
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F


class RocketNet(pl.LightningModule):
    def __init__(
            self,
            x_dim: int,
            n_classes: int,
            kernel_seed: int,
            kernel_count: int,
            max_sequence_len: int, ):
        super(RocketNet, self).__init__()
        self.save_hyperparameters()
        self.kernel_seed = kernel_seed
        self.n_classes = n_classes
        self.kernel_count = kernel_count
        self.feature_dim = 2 * kernel_count
        self.kernel_len_list = [7, 9, 11]
        self.max_sequence_len = max_sequence_len
        # linear classifier
        self.fc = nn.Linear(self.feature_dim, n_classes)
        self.conv_list = nn.ModuleList()
        self.thr = 0.8

        # set random seed for kernel init
        random.seed(kernel_seed)
        torch.manual_seed(kernel_seed)

        bias_arr = 2 * (torch.rand(self.kernel_count) - 0.5)
        kernel_lengths_arr = random.choices(self.kernel_len_list, k=kernel_count)

        for i in range(kernel_count):
            stride = 1
            kernel_len = kernel_lengths_arr[i]
            A = math.log2((self.max_sequence_len - 1) / (kernel_len - 1))

            dial_core = torch.rand(1) * A
            dial = max(int(torch.floor(dial_core)), 1)
            padding = random.randint(0, 1)
            cur_conv = nn.Conv1d(
                in_channels=x_dim,
                out_channels=1,
                kernel_size=(1, kernel_len),
                dilation=dial,
                stride=stride,
                padding=padding)
            cur_conv.bias = torch.nn.Parameter(torch.tensor([bias_arr[i]]), requires_grad=False)
            cur_conv.weight = torch.nn.Parameter(torch.empty_like(cur_conv.weight).normal_(), requires_grad=False)
            self.conv_list.append(cur_conv)

    def forward(self, x):
        x = x.unsqueeze(dim=2)  # => [batch_size, x_dim,1, max_sequence_len]
        features_2d = []
        for conv_filter in self.conv_list:
            batch_feature_maps = conv_filter(x)
            if batch_feature_maps.shape[2] != 1:  # pytorch also does padding on the [2] axis (this is unwanted)
                dim = int(batch_feature_maps.shape[2] / 2)
                batch_feature_maps = batch_feature_maps[:, :, dim, :].unsqueeze(dim=2)
            global_max = self.get_global_max(batch_feature_maps)
            ppv = self.get_ppv(batch_feature_maps)
            features_2d.append(torch.stack([global_max, ppv], dim=1))
        x = torch.cat(features_2d, dim=1)  # [batch_size, feature_dim]
        x = self.fc(x)
        pred = torch.sigmoid(x)
        return pred

    def get_global_max(self, batch_feature_maps):
        max_pooling = nn.MaxPool2d(batch_feature_maps.shape[2:])
        mav_vals = max_pooling(batch_feature_maps).squeeze()
        if len(mav_vals.shape) < 1:  # handle batch size==1
            mav_vals = mav_vals.unsqueeze(dim=0)
        return mav_vals

    def get_ppv(self, batch_feature_maps):
        batch_feature_maps[batch_feature_maps <= 0] = 0
        ppv = torch.count_nonzero(batch_feature_maps, dim=3) / torch.numel(batch_feature_maps[0, :, :, :])
        ppv = ppv.squeeze()
        if len(ppv.shape) < 1:  # handle batch size==1
            ppv = ppv.unsqueeze(dim=0)
        return ppv

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = F.binary_cross_entropy(pred, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = F.binary_cross_entropy(pred, y)
        accuracy = ((pred > self.thr) == y).float().mean()
        self.log('val_acc', accuracy.item(), prog_bar=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('learning_rate', self.optim.param_groups[0]["lr"], on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = \
            {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=2,
                                                                        threshold=0.0001, cooldown=0, min_lr=1e-7,
                                                                        eps=1e-08),
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        self.optim = optimizer
        return [optimizer], [scheduler]