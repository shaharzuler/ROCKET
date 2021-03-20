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
        self._max_pooling = nn.MaxPool1d(self.feature_dim)  # check correct dim

        self.conv_list = nn.ModuleList()

        # TODO get random weights before conv init

        bias_arr = 2 * (torch.rand(self.kernel_count) - 0.5)
        kernel_lengths_arr = random.choices(self.kernel_len_list, k=kernel_count)

        for i in range(kernel_count):
            stride = 1
            kernel_len = kernel_lengths_arr[i]
            A = math.log2((self.max_sequence_len - 1)/(kernel_len - 1))

            dial_core = torch.rand(1) * A
            dial = int(torch.floor(dial_core))
            cur_conv = nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=(x_dim, kernel_len),
                dilation=dial,
                stride=stride)
            cur_conv.bias = torch.nn.Parameter(bias_arr[i], requires_grad=False)
            cur_conv.weight = torch.nn.Parameter(torch.empty_like(cur_conv.weight).normal_(), requires_grad=False)
            self.conv_list.append(cur_conv)

    def forward(self, x):
        batch_size = x.shape[0]  # [batch_size, x_dim, max_sequence_len]
        conv_result = []
        for conv_filter in self.conv_list:
            cur_result = conv_filter(x)
            conv_result.append(cur_result)
        conv_result = torch.cat(conv_result, dim=1)

    def training_step(self, *args, **kwargs):
        pass

    def validation_step(self, *args, **kwargs):
        pass
