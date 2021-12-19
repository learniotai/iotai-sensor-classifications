import torch
from sklearn import metrics
from torch.autograd import Variable


def evaluate_prediction(predict, decode, test_X, test_y):
    """

    :return:
    """
    if not isinstance(test_X, torch.Tensor):
        test_X = Variable(torch.from_numpy(test_X)).float()
    if not isinstance(test_y, torch.Tensor):
        test_y = Variable(torch.from_numpy(test_y)).long()
    pred_y_labels = predict(test_X)
    test_y_labels = decode(test_y)
    accuracy = metrics.accuracy_score(pred_y_labels, test_y_labels)

    test_matrix = metrics.confusion_matrix(y_true=test_y_labels, y_pred=pred_y_labels)

    return accuracy, test_matrix


def test_accuracy(model, label_coder, test_X, test_y):
    """

    :return:
    """
    test_X = Variable(torch.from_numpy(test_X)).float()
    test_y = Variable(torch.from_numpy(test_y)).long()
    pred_y = model(test_X)
    pred_y_index = torch.argmax(pred_y, dim=1)
    accuracy = metrics.accuracy_score(pred_y_index, test_y)
    pred_y_labels = label_coder.decode(pred_y_index)
    test_y_labels = label_coder.decode(test_y)
    test_matrix = metrics.confusion_matrix(y_true=test_y_labels, y_pred=pred_y_labels)

    return accuracy, test_matrix