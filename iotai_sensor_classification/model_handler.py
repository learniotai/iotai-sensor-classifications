import torch
from torch.autograd import Variable


class ModelCall:
    """Make a callable model object for using models to make predictions."""
    def __init__(self, model, decode):
        """Make a callable model function object.

        :param model:
        """
        self._model = model
        self._decode = decode

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = Variable(torch.from_numpy(x)).float()
        pred_y = self._model(x)
        pred_index = torch.argmax(pred_y, dim=1)
        decoded = self._decode(pred_index)
        return decoded
