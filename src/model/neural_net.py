import torch
import torch.nn as net
import torch.nn.functional as Func


class CNN(net.Module):
    def __init__(self, dropout):
        super(CNN, self).__init__()
        dropout_prob = dropout

        self.conv1_1 = net.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.batch_norm1_1 = net.BatchNorm2d(32)  # глибина вхідного шару
        self.conv1_2 = net.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.batch_norm1_2 = net.BatchNorm2d(64)

        self.max_pool_1 = net.MaxPool2d(kernel_size=2, stride=2)
        self.dropout_1 = net.Dropout2d(dropout_prob)

        self.conv2_1 = net.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.batch_norm2_1 = net.BatchNorm2d(64)

        self.max_pool_2 = net.MaxPool2d(kernel_size=2, stride=2)
        self.dropout_2 = net.Dropout(dropout_prob)

        self.mlp1 = net.Linear(7 * 7 * 64, 256)
        self.batch_norm_mlp_1 = net.BatchNorm1d(256)
        self.dropout_mlp_1 = net.Dropout1d(dropout_prob)

        self.mlp2 = net.Linear(256, 128)
        self.batch_norm_mlp_2 = net.BatchNorm1d(128)

        self.mlp3 = net.Linear(128, 10)

    def forward(self, tensor):
        tensor = Func.relu(self.batch_norm1_1(self.conv1_1(tensor)))
        tensor = Func.relu(self.batch_norm1_2(self.conv1_2(tensor)))
        tensor = self.dropout_1(self.max_pool_1(tensor))
        tensor = Func.relu(self.batch_norm2_1(self.conv2_1(tensor)))
        tensor = self.dropout_2(self.max_pool_2(tensor))
        vector = torch.reshape(tensor, (-1, 7 * 7 * 64))
        vector = Func.relu(self.dropout_mlp_1(self.batch_norm_mlp_1(self.mlp1(vector))))
        vector = Func.relu(self.batch_norm_mlp_2(self.mlp2(vector)))
        vector = self.mlp3(vector)
        return vector





