# Import generic packages
from typing import Tuple

# Import torch related packages
import torchvision
from torch.utils.data import DataLoader


def mnist_dataloader(batch_size_train:int=64, batch_size_test:int=1000) -> Tuple[DataLoader, DataLoader]:

    '''
    -Description-
    Load MNIST dataset from source (http://yann.lecun.com/exdb/mnist/)
    :param batch_size_train:
    :param batch_size_test:
    :return: tuple: (train_loader,  test_loader)
    '''

    # MNIST dataset mean and standard deviation
    _MEAN_MNIST = (0.1307,)
    _STD_MNIST = (0.3081,)

    train_loader = DataLoader(
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

    test_loader = DataLoader(
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

    return train_loader, test_loader
