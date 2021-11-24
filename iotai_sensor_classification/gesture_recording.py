import os
import pandas as pd
import fnmatch


def read_gesture_recordings(recordings_dir):
    """Read gesture recordings from CSV files.

    Use the base file name as the gesture name.
    :param recordings_dir: directory with recordings in CSV files.
    :return: dictionary of gestures names: pandas data frame recording."""
    csv_recordings = fnmatch.filter(os.listdir(recordings_dir), "*.csv.gz")
    gesture_recording = {}
    for recording in csv_recordings:
        # strip off file extensions repeatedly
        gesture = recording
        while "." in gesture:
            gesture = os.path.splitext(gesture)[0]
        recording_path = os.path.join(recordings_dir, recording)
        recording = pd.read_csv(recording_path)
        gesture_recording[gesture] = recording
    return gesture_recording
