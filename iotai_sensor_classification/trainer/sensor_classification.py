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

EPOCHS = 200
