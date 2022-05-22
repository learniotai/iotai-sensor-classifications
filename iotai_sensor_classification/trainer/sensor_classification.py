"""Train sensor classification model."""

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import tqdm
import numpy as np
import pandas as pd

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


class LstmModel(nn.Module):
    """LSTM model."""
    def __init__(self, input_dim, output_dim, hidden_size=32, num_layers=1):
        super(LstmModel, self).__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._flat_dim = np.prod(input_dim)
        self.lstm = nn.LSTM(input_size=input_dim[-1], hidden_size=hidden_size, num_layers=num_layers, bidirectional=True)
        self.fc_hidden = nn.Linear(hidden_size * 2, output_dim)
        self.fc_out = nn.Linear(self._flat_dim, output_dim)

    def forward(self, x):
        packed_output, (hidden_state, cell_state) = self.lstm(x)
        hidden_output = torch.cat((hidden_state[-2, :, :], hidden_state[-1, :, :]), dim=1)
        hidden_fc = self.fc_hidden(hidden_output)
        flat = torch.flatten(hidden_fc)
        output = self.fc_out(flat)
        soft = F.softmax(output)
        return soft


def train_gesture_classification(model, X_train, y_train, X_val=None, y_val=None):
    """Train gesture classification from sensor data.

    :return:
    """
    X_train = Variable(torch.from_numpy(X_train)).float()
    y_train = Variable(torch.from_numpy(y_train)).long()
    if X_val is not None and y_val is not None:
        X_val = Variable(torch.from_numpy(X_val)).float()
        y_val = Variable(torch.from_numpy(y_val)).long()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    loss_list = np.zeros((EPOCHS,))
    val_accuracy_list = np.zeros((EPOCHS,))
    val_loss_list = np.zeros((EPOCHS,))
    for epoch in tqdm.trange(EPOCHS):
        pred_y = model(X_train)
        loss = loss_fn(pred_y, y_train)
        loss_list[epoch] = loss.item()

        # Zero gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if X_val is not None and y_val is not None:
            with torch.no_grad():
                pred_val_y = model(X_val)
                correct = (torch.argmax(pred_val_y, dim=1) == y_val).type(torch.FloatTensor)
                val_accuracy_list[epoch] = correct.mean()
                val_loss = loss_fn(pred_val_y, y_val)
                val_loss_list[epoch] = val_loss
    if X_val is not None and y_val is not None:
        return pd.DataFrame({"train_loss": loss_list, "val_loss": val_loss_list, "val_acc": val_accuracy_list})

    return pd.DataFrame({"train_loss": loss_list})
