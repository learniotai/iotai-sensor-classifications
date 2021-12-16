"""Train sensor classification model."""

import os
import numpy as np
import pandas as pd
import torch
from iotai_sensor_classification import dataset
from iotai_sensor_classification.preprocess import check_windows, parse_recording, SAMPLES_PER_RECORDING
from iotai_sensor_classification.recording import read_recordings
from iotai_sensor_classification.plot_util import group_label_bars
from data.gestures import linear_accelerometer
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
    return model, val_df
