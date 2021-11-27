import os
import pandas as pd
import fnmatch
import numpy as np
from typing import Dict
from .encode import LabelCoder


def strip_extensions(filename: str):
    """Strip off all extensions even if more than one."""
    # strip off file extensions repeatedly
    name_ = filename
    while "." in name_:
        name_ = os.path.splitext(name_)[0]
    return name_


def filter_columns(data, keep_dtypes=[np.float]):
    """Filter columns by type."""
    is_float = [dtype in keep_dtypes for dtype in data.dtypes]
    float_cols = list(data.columns[is_float])
    return data[float_cols]


def get_recording_names(recordings_dir, ext=".csv.gz"):
    """Get recording names from filenames in directory of recordings."""
    csv_recordings = fnmatch.filter(os.listdir(recordings_dir), f"*{ext}")
    recording_names = [strip_extensions(recording) for recording in csv_recordings]
    return recording_names


def read_recordings(recordings_dir):
    """Read gesture recordings from CSV files.

    Use the base file name as the gesture name.
    :param recordings_dir: directory with recordings in CSV files.
    :return: dictionary of gestures names: pandas data frame recording."""
    csv_recordings = fnmatch.filter(os.listdir(recordings_dir), "*.csv.gz")
    named_recordings = {}
    for recording in csv_recordings:
        # strip off file extensions repeatedly
        record_label = strip_extensions(recording)
        recording_path = os.path.join(recordings_dir, recording)
        recording = pd.read_csv(recording_path)
        recording["label"] = record_label
        named_recordings[record_label] = recording
    return named_recordings


def parse_recording(labeled_recordings: Dict[str, pd.DataFrame], label_name="label"):
    """Parse recordings into a dataset and create label decoder.
    :return: parsed recordings, label coder for decoding labels."""
    all_recordings = pd.concat(list(labeled_recordings.values()), axis=0)
    label_coder = LabelCoder()
    encoded_labels = label_coder.encode_labels(all_recordings[label_name])
    all_recordings["label_code"] = list(encoded_labels)
    return all_recordings, label_coder
