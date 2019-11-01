from __future__ import print_function
from typing import Callable
import numpy as np
from sklearn.utils import shuffle
from sklearn import metrics
from random import seed
import time
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import torch.utils.data as utils
from tools import Logger, accuracy, TotalMeter

logger = Logger()

partition = np.loadtxt('example_adjacency.txt', dtype=int, delimiter=None)
expression = np.loadtxt('example_expression.csv', dtype=float, delimiter=",")
labels = np.array(expression[:, -1], dtype=int)
expression = np.array(expression[:, :-1])


# train/test data split
cut = int(0.8*np.shape(expression)[0])
expression, labels = shuffle(expression, labels)
x_train = expression[:cut, :]
x_test = expression[cut:, :]
y_train = labels[:cut]
y_test = labels[cut:]
print(x_test.shape)
partition = torch.from_numpy(partition).float()


class PartitionLayer(nn.Module):
    def __init__(self, partition: torch.Tensor, bias: bool = True):
        super().__init__()
        self.partition = partition
        self.in_features = partition.shape[0]
        self.out_features = partition.shape[0]
        self.weight = nn.Parameter(torch.zeros(
            self.out_features, self.in_features), requires_grad=True)

        if bias:
            self.bias = nn.Parameter(torch.zeros(
                self.out_features), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        assert x.shape[1] == self.in_features
        t = self.weight * self.partition
        return F.linear(x, t, self.bias)


class MultilayerPerceptron(nn.Module):
    def __init__(self, partition, out_features):
        super().__init__()
        dropout_rate = 0.3
        self.part_layer = PartitionLayer(partition)
        self.blocks = nn.Sequential(
            nn.Dropout(dropout_rate),
            # nn.BatchNorm1d(partition.shape[0]),
            nn.ReLU(),
            nn.Linear(partition.shape[0], 64),
            nn.Dropout(dropout_rate),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.Dropout(dropout_rate),
            # nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, out_features),
        )

    def forward(self, x):
        x = self.part_layer(x)
        return self.blocks(x)


L2 = False
max_pooling = False
droph1 = False
learning_rate = 0.0001
training_epochs = 100
batch_size = 8

train_dataset = utils.TensorDataset(
    torch.from_numpy(x_train).float(),
    torch.from_numpy(y_train)
)
train_dataloader = utils.DataLoader(train_dataset, batch_size=batch_size)

test_dataset = utils.TensorDataset(
    torch.from_numpy(x_test).float(),
    torch.from_numpy(y_test)
)
test_dataloader = utils.DataLoader(test_dataset, batch_size=8)

model = MultilayerPerceptron(partition, 2)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

train_loss, test_loss, train_accuracy, test_accuracy = [
    TotalMeter() for _ in range(4)]

for epoch in range(training_epochs):
    model.train()

    for meter in [train_accuracy, test_accuracy, train_loss, test_loss]:
        meter.reset()

    for index, (data_in, label) in enumerate(train_dataloader):
        output = model(data_in)
        loss = loss_fn(output, label)
        train_loss.update_with_weight(loss.item(), label.shape[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        top1 = accuracy(output, label)[0]
        train_accuracy.update_with_weight(top1, label.shape[0])

    model.eval()

    for data_in, label in test_dataloader:
        output = model(data_in)
        loss = loss_fn(output, label)
        test_loss.update_with_weight(loss.item(), label.shape[0])
        top1 = accuracy(output, label)[0]
        test_accuracy.update_with_weight(top1, label.shape[0])

    logger.info(" | ".join([
        f'Epoch[{epoch}/{training_epochs}]',
        f'Train Loss:{train_loss.avg: .3f}',
        f'Train Accuracy:{train_accuracy.avg: .3f}%',
        f'Test Loss:{test_loss.avg: .3f}',
        f'Test Accuracy:{test_accuracy.avg: .3f}%'
    ]))
