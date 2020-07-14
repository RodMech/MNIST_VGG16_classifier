# Import generic packages
from typing import Tuple

# Import torch related packages
import torchvision
from torch.utils.data import DataLoader


def mnist_dataloader(batch_size_train:int=64, batch_size_test:int=64) -> Tuple[DataLoader, DataLoader]:

    '''
    -Description-
    Load MNIST dataset from source (http://yann.lecun.com/exdb/mnist/)
    :param batch_size_train: 64 original
    :param batch_size_test: 1000 original
    :return: tuple: (train_loader,  test_loader)
    '''

    # MNIST dataset mean and standard deviation
    _MEAN_MNIST = (0.1307,)
    _STD_MNIST = (0.3081,)

    # TODO: Introduce a new Conv layer to give MNIST input 3 channels. See:
    # https://discuss.pytorch.org/t/how-to-get-mnist-data-from-torchvision-with-three-channels-for-some-pretrained-model-like-vgg/21872

    train_loader = DataLoader(
        torchvision.datasets.MNIST('/files/',
                                   train=True,
                                   download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.Resize(size=256),
                                       torchvision.transforms.Grayscale(3),
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
                                       torchvision.transforms.Resize(size=256),
                                       torchvision.transforms.Grayscale(3),
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           _MEAN_MNIST, _STD_MNIST)
                                   ])),
        batch_size=batch_size_test,
        shuffle=True)

    return train_loader, test_loader
