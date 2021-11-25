import os
import pandas as pd
import fnmatch


def read_recordings(recordings_dir):
    """Read gesture recordings from CSV files.

    Use the base file name as the gesture name.
    :param recordings_dir: directory with recordings in CSV files.
    :return: dictionary of gestures names: pandas data frame recording."""
    csv_recordings = fnmatch.filter(os.listdir(recordings_dir), "*.csv.gz")
    named_recordings = {}
    for recording in csv_recordings:
        # strip off file extensions repeatedly
        record_name = recording
        while "." in record_name:
            record_name = os.path.splitext(record_name)[0]
        recording_path = os.path.join(recordings_dir, recording)
        recording = pd.read_csv(recording_path)
        named_recordings[record_name] = recording
    return named_recordings
