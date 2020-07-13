from model import Vgg16Mnist
from train import train
from mnist_dataloader import mnist_dataloader

from torch import optim
from torch import nn

def transfer_learning(train_epochs:int=20, early_stop_epochs:int=3, save_file_name:str="MNIST_VGG16_transfer.pt"):
    '''
    INFO
    ----
    From a VGG16 net pretrained in Imagenet, we iterate n epochs to train the net in MNIST.
    It serves for obtaining valid weights. No return allowed.
    :param train_epochs: number of epochs to train
    :param early_stop_epochs: stop epochs if there is no loss improvement
    :param save_file_name: save weights as a *.pt file
    :return there is no return. Weights are generated in the wkdir under save_file_name denomination.
    '''

    #TODO: Defensive. Check that the save_file_name has a *.pt file extension.

    # Initialise the VGG16 model (pretrained)
    vgg16_mnist = Vgg16Mnist()

    # Prepare the model for transfer learning training
    vgg16_mnist.transfer_learning_prep()

    # Retrieve the model for training
    vgg16 = vgg16_mnist.model()

    # Loss and optimizer
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(vgg16.parameters())

    # Import train_loader and valid_loader
    train_loader, valid_loader = mnist_dataloader()

    # Train: weights are generated in the working directory
    model, history = train(model=vgg16,
                           criterion=loss_function,
                           optimizer=optimizer,
                           train_loader=train_loader,
                           valid_loader=valid_loader,
                           save_file_name=save_file_name,
                           max_epochs_stop=early_stop_epochs,
                           n_epochs=train_epochs,
                           print_every=2
                           )

    # For analising loss and accuracy curves
    # history.to_csv("MNIST_VGG16_transfer.csv", sep=",", index=False)

