"""Preprocess sensor data recordings."""

from typing import Dict
import pandas as pd
import numpy as np
from .normalization import normalize_mean_std
from .encode import LabelCoder

SAMPLES_PER_RECORDING = 160


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
        recording = recording.drop(columns=['time', 'label'])
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

