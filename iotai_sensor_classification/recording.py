import os
import pandas as pd
import fnmatch


def strip_extensions(filename: str):
    """Strip off all extensions even if more than one."""
    # strip off file extensions repeatedly
    name_ = filename
    while "." in name_:
        name_ = os.path.splitext(name_)[0]
    return name_


def read_recordings(recordings_dir):
    """Read gesture recordings from CSV files.

    Use the base file name as the gesture name.
    :param recordings_dir: directory with recordings in CSV files.
    :return: dictionary of gestures names: pandas data frame recording."""
    csv_recordings = fnmatch.filter(os.listdir(recordings_dir), "*.csv.gz")
    named_recordings = {}
    for recording in csv_recordings:
        # strip off file extensions repeatedly
        record_name = strip_extensions(recording)
        recording_path = os.path.join(recordings_dir, recording)
        recording = pd.read_csv(recording_path)
        named_recordings[record_name] = recording
    return named_recordings
