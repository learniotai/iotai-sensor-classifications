"""Test training of sensor classification."""

import os
from data.gestures import linear_accelerometer
from iotai_sensor_classification import dataset
from iotai_sensor_classification.preprocess import check_windows, parse_recording, SAMPLES_PER_RECORDING
from iotai_sensor_classification.plot_util import group_label_bars
from iotai_sensor_classification.recording import read_recordings
from iotai_sensor_classification.trainer.sensor_classification import EPOCHS
from iotai_sensor_classification.plot_util import plot_columns
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import tqdm
import numpy as np
import pandas as pd


def get_accelerometer_dataset():
    """Read gesture recordings for all tests in file."""
    recordings_dir = os.path.dirname(linear_accelerometer.__file__)
    recordings = read_recordings(recordings_dir=recordings_dir)
    window_checked = check_windows(recordings)
    normed_gesture_measures, encoded_labels, label_coder = \
        parse_recording(window_checked, samples_per_recording=SAMPLES_PER_RECORDING)
    return normed_gesture_measures, encoded_labels, label_coder


class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, output_dim)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x


def test_train_sensor_classification():
    """
    Test traininer sensor classification model.
    :return:
    """
    normed_gesture_measures, encoded_labels, label_coder = get_accelerometer_dataset()
    train_X, val_X, test_X, train_y, val_y, test_y = \
        dataset.split_dataset(normed_gesture_measures, encoded_labels, val_size=dataset.VALIDATION_SPLIT,
                              test_size=dataset.TEST_SPLIT)
    assert len(train_X) == len(train_y)
    assert len(val_X) == len(val_y)
    assert len(test_X) == len(test_y)
    train_y_labels = label_coder.decode_one_hots(train_y)
    val_y_labels = label_coder.decode_one_hots(val_y)
    test_y_labels = label_coder.decode_one_hots(test_y)
    assert len(train_y_labels) == len(train_y)
    assert len(val_y_labels) == len(val_y)
    assert len(test_y_labels) == len(test_y)

    raw_labels = {}
    raw_labels['train'] = train_y_labels
    raw_labels['validation'] = val_y_labels
    raw_labels['test'] = test_y_labels

    test_output = os.path.join("test_output", "gestures", "trainer")
    os.makedirs(test_output, exist_ok=True)
    group_label_bars(raw_labels, title="Gesture classification datasets label count",
                     filepath=os.path.join(test_output, "gesture_classification_dataset.png"))

    train_X = Variable(torch.from_numpy(train_X)).float()
    val_X = Variable(torch.from_numpy(val_X)).float()
    test_X = Variable(torch.from_numpy(test_X)).float()
    train_y = Variable(torch.from_numpy(train_y)).float()
    val_y = Variable(torch.from_numpy(val_y)).float()
    test_y = Variable(torch.from_numpy(test_y)).float()
    model = Model(input_dim=train_X.shape[1]*train_X.shape[2], output_dim=train_y.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    loss_list = np.zeros((EPOCHS,))
    accuracy_list = np.zeros((EPOCHS,))
    for epoch in tqdm.trange(EPOCHS):
        pred_y = model(train_X)
        loss = loss_fn(pred_y, torch.argmax(train_y, dim=1))
        loss_list[epoch] = loss.item()

        # Zero gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred_val_y = model(val_X)
            correct = (torch.argmax(pred_val_y, dim=1) == torch.argmax(val_y, dim=1)).type(torch.FloatTensor)
            accuracy_list[epoch] = correct.mean()
    val_df = pd.DataFrame({"val_loss": loss_list, "val_acc": accuracy_list})
    plot_columns(val_df, name="Gesture classification training validation",
                 filepath=os.path.join(test_output, "gesture_classification_training_validation.png"),
                 title_mean=False)
