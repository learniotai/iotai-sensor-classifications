import torch
from sklearn import metrics
from torch.autograd import Variable


def evaluate_prediction(predict, decode, X_test, y_test):
    """

    :return:
    """
    if not isinstance(X_test, torch.Tensor):
        X_test = Variable(torch.from_numpy(X_test)).float()
    if not isinstance(y_test, torch.Tensor):
        y_test = Variable(torch.from_numpy(y_test)).long()
    pred_y_labels = predict(X_test)
    test_y_labels = decode(y_test)
    accuracy = metrics.accuracy_score(pred_y_labels, test_y_labels)

    test_matrix = metrics.confusion_matrix(y_true=test_y_labels, y_pred=pred_y_labels)

    return accuracy, test_matrix


def evaluate_model(model, label_coder, X_test, y_test):
    """

    :return:
    """
    X_test = Variable(torch.from_numpy(X_test)).float()
    y_test = Variable(torch.from_numpy(y_test)).long()
    pred_y = model(X_test)
    pred_y_index = torch.argmax(pred_y, dim=1)
    accuracy = metrics.accuracy_score(pred_y_index, y_test)
    pred_y_labels = label_coder.decode(pred_y_index)
    test_y_labels = label_coder.decode(y_test)
    test_matrix = metrics.confusion_matrix(y_true=test_y_labels, y_pred=pred_y_labels)

    return accuracy, test_matrix