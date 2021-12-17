"""Train sensor classification model."""

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import tqdm
import numpy as np
import pandas as pd
from sklearn import metrics


EPOCHS = 200


class LinearModel(nn.Module):
    """Model creation.
    """
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, output_dim)

    def forward(self, x):
        """Forward pass."""
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x


class ConvModel(nn.Module):
    """Convolution 2D model."""
    def __init__(self, input_dim, output_dim):
        super(ConvModel, self).__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        # 1 input image channel, 6 output channels, 2x2 square convolution
        # kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=2, padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=2, padding=(1, 1))
        # an affine operation: y = Wx + b
        # input size increased by 1 * 2 = 2 for 2 padded convolutions with 2x2 kernels
        self.fc1 = nn.Linear(in_features=16 * (input_dim[0]+2) * (input_dim[1]+2), out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x.unsqueeze(1)))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


def train_gesture_classification(model, train_X, val_X, train_y, val_y):
    """Train gesture classification from sensor data.

    :return:
    """
    train_X = Variable(torch.from_numpy(train_X)).float()
    val_X = Variable(torch.from_numpy(val_X)).float()
    train_y = Variable(torch.from_numpy(train_y)).long()
    val_y = Variable(torch.from_numpy(val_y)).long()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    loss_list = np.zeros((EPOCHS,))
    accuracy_list = np.zeros((EPOCHS,))
    for epoch in tqdm.trange(EPOCHS):
        pred_y = model(train_X)
        loss = loss_fn(pred_y, train_y)
        loss_list[epoch] = loss.item()

        # Zero gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred_val_y = model(val_X)
            correct = (torch.argmax(pred_val_y, dim=1) == val_y).type(torch.FloatTensor)
            accuracy_list[epoch] = correct.mean()
    val_df = pd.DataFrame({"val_loss": loss_list, "val_acc": accuracy_list})
    return val_df


def test_accuracy(model, label_coder, test_X, test_y):
    """

    :return:
    """
    test_X = Variable(torch.from_numpy(test_X)).float()
    test_y = Variable(torch.from_numpy(test_y)).long()
    pred_y = model(test_X)
    pred_y_index = torch.argmax(pred_y, dim=1)
    accuracy = metrics.accuracy_score(pred_y_index, test_y)
    pred_y_labels = label_coder.decode(pred_y_index)
    test_y_labels = label_coder.decode(test_y)
    test_matrix = metrics.confusion_matrix(y_true=test_y_labels, y_pred=pred_y_labels)
    return accuracy, test_matrix
