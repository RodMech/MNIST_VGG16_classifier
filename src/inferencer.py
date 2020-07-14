from src.model import Vgg16Mnist
from src.mnist_dataloader import mnist_dataloader

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time


class Inferencer:
    def __init__(self,
                 weight_path: str = "./MNIST_VGG16_transfer.pt",
                 image_number: int = 0,
                 top_preds: int = 5
                 ):
        self.weight_path = weight_path
        self.image_number = image_number    # Select the ith image in the batch for saving graphic output
        self.top_preds = top_preds          # Evaluate the top k predictions
        self._test_on_gpu = torch.cuda.is_available()
        self._test_data = None

        if self._test_on_gpu == True:       # CPU: 16, GPU: 64
            self.batch_size = 64
        else:
            self.batch_size = 16

    def load_test_data(self) -> DataLoader:
        # Import the MNIST data for test
        _, test_data = mnist_dataloader()
        self._test_data = test_data

    def graphic_output(self, tensors_sample, topclass):

        # Permute the torch dimensions for matplotlib I/O
        image = tensors_sample[self.image_number].cpu().permute(1, 2, 0).numpy()

        # Clip the image pixel values
        image = np.clip(image, 0, 1)

        # Save the ith image with the top 5 classes predicted
        top_classes = topclass.cpu().numpy()[0]
        plt.imsave(f"./images/{top_classes}.png", image)

    def perform_inference(self):
        # Do not track gradients
        with torch.no_grad():

            # Load the transfer learning weights into the model
            vgg16 = Vgg16Mnist(pretrained=False)
            vgg16.set_test_mode(weight_path=self.weight_path)

            # Iterate over the test dataloader
            for data, target in self._test_data:
                # Tensors to gpu
                if self._test_on_gpu:
                    data, target = data.cuda(), target.cuda()

                tensors_sample = data[0:self.batch_size, :, :, :]
                ground_truth = target.cpu().numpy()[:self.batch_size]

                # Forward pass
                initial_time = time.time()
                output = vgg16.evaluate(tensors_sample)
                final_time = time.time()

                eval_proba = torch.exp(output)

                # Find the topk predictions
                topk, topclass = eval_proba.topk(self.top_preds, dim=1)

                # Save a sample of MNIST test to a fodler
                self.graphic_output(
                                    tensors_sample=tensors_sample,
                                    topclass=topclass
                                    )

                # Print accuracy and bechmarks
                accuracy = (sum(topclass[:, 0].cpu().numpy() == ground_truth) / self.batch_size) * 100
                print(f"[INFERENCE] {self.batch_size} images analised in {final_time-initial_time} seconds. Accuracy: {accuracy}%")













