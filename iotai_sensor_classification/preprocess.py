"""Preprocess sensor data recordings."""

from typing import Dict
import pandas as pd
import numpy as np
from .normalization import normalize_mean_std
from .encode import LabelCoder
from iotai_sensor_classification.plot_util import plot_lines

SAMPLES_PER_RECORDING = 160


def check_windows(data, window_size=SAMPLES_PER_RECORDING):
    """Look for windows out of range at beginning and end and remove."""
    keep_data = {}
    for label_name in data.keys():
        label_data = data[label_name]
        if 'time' in label_data.columns:
            label_data = label_data.drop(columns=['time'])
        if 'label' in label_data.columns:
            label_data = label_data.drop(columns=['label'])
        label_means = label_data.mean()
        label_stds = label_data.std()
        label_lower_bound = label_means - (2.0*label_stds)
        label_upper_bound = label_means + (2.0*label_stds)
        n_windows = int(label_data.shape[0] / window_size)
        windows = np.array_split(label_data, n_windows)
        window_means = [s.mean() for s in windows]
        window_stds = [s.std() for s in windows]
        mean_keep_window = []
        std_keep_window = []
        keep_window = []
        keep_window_data = []
        for index in range(len(window_means)):
            window_mean = window_means[index]
            window_std = window_stds[index]
            keep = True
            if any(window_mean < label_lower_bound):
                keep = False
                mean_keep_window.append(False)
            elif any(window_mean > label_upper_bound):
                keep = False
                mean_keep_window.append(False)
            else:
                mean_keep_window.append(True)

            if any(window_std < label_stds/3.0):
                keep = False
                std_keep_window.append(False)
            elif any(window_std > label_stds*3.0):
                keep = False
                std_keep_window.append(False)
            else:
                std_keep_window.append(True)
            keep_window.append(keep)
            if keep:
                window_data = windows[index]
                # restore text label dropped to make numeric calculations
                window_data['label'] = label_name
                keep_window_data.append(window_data)
        # interactive look at data
        # plot_lines(label_data, label_name)
        keep_label_windows = pd.concat(keep_window_data)
        keep_data[label_name] = keep_label_windows
    return keep_data


def parse_recording(labeled_recordings: Dict[str, pd.DataFrame], samples_per_recording):
    """Parse recordings into a dataset and create label decoder.
    :return: parsed recordings, label coder for decoding labels."""
    all_recordings = pd.concat(list(labeled_recordings.values()), axis=0)
    all_samples = []
    all_sample_labels = []
    normalized_recordings = normalize_mean_std(all_recordings)
    for label in normalized_recordings['label'].unique():
        recording = normalized_recordings.loc[normalized_recordings['label'].isin([label])]
        # time is the same for every gestures and does not distinguish gestures
        if 'time' in recording.columns:
            recording = recording.drop(columns=['time'])
        if 'label' in recording.columns:
            recording = recording.drop(columns=['label'])
        n_samples = int(recording.shape[0]/samples_per_recording)
        # only keep rows that will make full samples with all samples per recording
        full_records = recording[0:n_samples*samples_per_recording]
        samples = np.array_split(full_records, n_samples)
        sample_tensor = np.array(samples)
        # create one label for each samples_per_recording measurements with a column for each measurement
        sample_labels = [label] * sample_tensor.shape[0]
        all_samples.append(sample_tensor)
        all_sample_labels += sample_labels
    all_sample_tensors = np.concatenate(all_samples)
    assert all_sample_tensors.shape[0] == len(all_sample_labels)
    label_coder = LabelCoder()
    encoded_labels = label_coder.encode_labels(all_sample_labels)
    assert len(labeled_recordings.keys()) == encoded_labels.shape[1]
    return all_sample_tensors, encoded_labels, label_coder

