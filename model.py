import math
import torch
import random
import torch.nn as nn
import pytorch_lightning as pl


class RocketNet(pl.LightningModule):
    def __init__(
            self,
            x_dim: int,
            n_classes: int,
            kernel_count: int,
            max_sequence_len: int,
            kernel_lengths: list):
        super(RocketNet, self).__init__()
        self.n_classes = n_classes
        self.kernel_count = kernel_count
        self.feature_dim = 2 * kernel_count
        self.kernel_len_list = kernel_lengths
        self.max_sequence_len = max_sequence_len
        # linear classifier
        self._fc = nn.Linear(self.feature_dim, n_classes)
        self._softmax = nn.Softmax(dim=1)

        self.conv_list = nn.ModuleList()

        # TODO get random weights before conv init

        bias_arr = 2 * (torch.rand(self.kernel_count) - 0.5)
        kernel_lengths_arr = random.choices(self.kernel_len_list, k=kernel_count)

        for i in range(kernel_count):
            stride = 1
            kernel_len = kernel_lengths_arr[i]
            A = math.log2((self.max_sequence_len - 1) / (kernel_len - 1))

            dial_core = torch.rand(1) * A
            dial = max(int(torch.floor(dial_core)), 1)
            cur_conv = nn.Conv1d(
                in_channels=x_dim,
                out_channels=1,
                kernel_size=(1, kernel_len),
                dilation=dial,
                stride=stride)
            cur_conv.bias = torch.nn.Parameter(torch.tensor([bias_arr[i]]), requires_grad=False)
            cur_conv.weight = torch.nn.Parameter(torch.empty_like(cur_conv.weight).normal_(), requires_grad=False)
            self.conv_list.append(cur_conv)

    def forward(self, x):
        x = x.unsqueeze(dim=2)  # => [batch_size, x_dim,1, max_sequence_len]
        features_2d = []
        for conv_filter in self.conv_list:
            batch_feature_maps = conv_filter(x)
            global_max = self.get_global_max(batch_feature_maps)
            ppv = self.get_ppv(batch_feature_maps)
            features_2d.append(torch.stack([global_max, ppv], dim=1))
        x = torch.cat(features_2d, dim=1)  # [batch_size, feature_dim]
        x = self._fc(x)
        pred = self._softmax(x)
        return pred

    def get_global_max(self, batch_feature_maps):
        max_pooling = nn.MaxPool2d(batch_feature_maps.shape[2:])
        return max_pooling(batch_feature_maps).squeeze()

    def get_ppv(self, batch_feature_maps):
        batch_feature_maps[batch_feature_maps <= 0] = 0
        ppv = torch.count_nonzero(batch_feature_maps, dim=3) / torch.numel(batch_feature_maps[0, :, :, :])
        return ppv.squeeze()

    def training_step(self, train_batch):
        x, y = train_batch
        pred = self(x)
        loss = self.loss(pred, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch):
        x, y = val_batch
        pred = self(x)
        loss = self.loss(pred, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('learning_rate', self.optim.param_groups[0]["lr"], on_step=False, on_epoch=True)

# for i in range(100):
#     rocket = RocketNet(x_dim=20, n_classes=4, kernel_count=5, max_sequence_len=100, kernel_lengths=[3, 4, 5])
#     input = torch.randn([30, 20,  100])
#     probs = rocket(input)
#     print(probs)
