import torch
import torchvision
from typing import Tuple

batch_size_train = 64
batch_size_test = 1000

def mnist_dataset_ttsplit (batch_size_train: int, batch_size_test: int) -> Tuple:
    '''
    -Description-
    Load MNIST dataset from source
    :param batch_size_train: determine
    :param batch_size_test:
    :return:
    '''

    # MNIST dataset mean and standard deviation
    _MEAN_MNIST = (0.1307,)
    _STD_MNIST = (0.3081,)

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/',
                                   train=True,
                                   download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           _MEAN_MNIST, _STD_MNIST)
                                   ])),
        batch_size=batch_size_train,
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/',
                                   train=False,
                                   download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           _MEAN_MNIST, _STD_MNIST)
                                   ])),
        batch_size=batch_size_test,
        shuffle=True)
