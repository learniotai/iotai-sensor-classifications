"""Preprocess sensor data recordings."""

from typing import Dict
import pandas as pd
from .normalization import normalize_mean_std
from .encode import LabelCoder


def parse_recording(labeled_recordings: Dict[str, pd.DataFrame], label_name="label"):
    """Parse recordings into a dataset and create label decoder.
    :return: parsed recordings, label coder for decoding labels."""
    all_recordings = pd.concat(list(labeled_recordings.values()), axis=0)
    normalized_recordings = normalize_mean_std(all_recordings)
    label_coder = LabelCoder()
    encoded_labels = label_coder.encode_labels(normalized_recordings[label_name])
    normalized_recordings["label_code"] = list(encoded_labels)
    return normalized_recordings, label_coder
