"""Dataset functionality for providing training, validation and testing data."""

from sklearn.model_selection import train_test_split

TEST_SPLIT = 0.15
VALIDATION_SPLIT = 0.15


def split_dataset(X, y, val_size, test_size, shuffle=True):
    """Split data into 3 parts: train, validate, test.

    Test size is determined by the remainder of the other two splits.
    :param X: input data
    :param y: output data
    :param val_size: validation fraction
    :param test_size: test fraction
    :param shuffle: randomize rows of data, default True
    :return: train_X, val_X, test_X, train_y, val_y, test_y
    """
    train_val_X, test_X, train_val_y, test_y = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
    train_val_size = 1.0 - test_size
    # adjust validation proportion of train_val data up to match test_size for second split
    val_of_tv_size = val_size/train_val_size
    train_X, val_X, train_y, val_y = \
        train_test_split(train_val_X, train_val_y, test_size=val_of_tv_size, shuffle=shuffle)

    return train_X, val_X, test_X, train_y, val_y, test_y
