from torchvision import models
import torch.nn as nn
import torch


class Vgg16Mnist:

    def __init__(self, pretrained: bool=True):
        self.pretrained = pretrained
        self.vgg16 = models.vgg16_bn(pretrained=pretrained)

    def freeze_layers(self):
        # Freeze VGG16 model weights (all layers)
        for param in self.vgg16.parameters():
            param.requires_grad = False

    def add_classifier(self):
        # Replace the sixth layer by a new classifier (transfer learning)
        # with ten classes for MNIST dataset output prediction

        MNIST_N_CLASSES = 10

        # TODO: translate hardcoded layer parameters to constants outside the signature
        self.vgg16.classifier[6] = nn.Sequential(
                                                nn.Linear(4096, 256),
                                                nn.ReLU(),
                                                nn.Dropout(0.4),
                                                nn.Linear(256, MNIST_N_CLASSES),
                                                nn.LogSoftmax(dim=1)
                                                )

    def transfer_learning_prep(self):
        # Prepare the net for transfer learning classifier-only training
        self.freeze_layers()
        self.add_classifier()

    def model(self) -> models.vgg16_bn:
        # Return the model only
        return self.vgg16

    def set_test_mode(self, weight_path: str):
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        self.transfer_learning_prep()
        self.vgg16.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))

        if torch.cuda.is_available():
            self.vgg16.to("cuda")

    def evaluate(self, input):
        return self.vgg16(input)
