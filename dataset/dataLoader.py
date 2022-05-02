import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from dataset.utils import get_default_transform


class DL:
    def __init__(self, args):
        data = None
        self.n_classes = -1
        self.out_features = -1
        if args.dataset == 'mnist':
            T = get_default_transform(1)
            data = MNIST(args.data_path, train=True, download=True, transform=T)
            self.n_classes = 10
            self.out_features = 784
        self.dl = DataLoader(data, batch_size=400, shuffle=True)
