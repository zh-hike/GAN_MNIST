import torch.nn as nn


class G(nn.Module):
    def __init__(self, in_features, out_features):
        super(G, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, out_features),
            nn.Tanh(),
        )

    def forward(self, img):
        return self.net(img)


class D(nn.Module):
    def __init__(self, in_features):
        super(D, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        return self.net(img)


class ConvG(nn.Module):
    def __init__(self, in_features, out_features):
        super(ConvG, self).__init__()
        self.layer = nn.Sequential(
            ConvTransposeLayer(in_features, 256, 2, 1, 0, get_outputpadding(1, 2, 2, 1, 0)),  # n,256,2,2
            ConvTransposeLayer(256, 512, 4, 2, 0, get_outputpadding(2, 7, 4, 2, 0)),  # n,512,7,7
            ConvTransposeLayer(512, 128, 4, 2, 1, get_outputpadding(7, 14, 4, 2, 1)),  # n,128,14,14
            nn.ConvTranspose2d(128, out_features, 4, 2, 1, get_outputpadding(14, 28, 4, 2, 1)),  # n,1,28,28
            nn.Tanh(),
        )

    def forward(self, data):
        output = self.layer(data)
        return output


class ConvD(nn.Module):
    def __init__(self):
        super(ConvD, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1, 128, 3, 2, 1),  # n,128,14,14
            nn.LeakyReLU(),
            ConvolutionalLayer(128, 256, 3, 2, 1),  # n,256,7,7
            ConvolutionalLayer(256, 512, 3, 2),  # n,512,3,3
            nn.Conv2d(512, 1, 3),  # n,1,1,1
            nn.Sigmoid()
        )

    def forward(self, data):
        output = self.layer(data)
        return output


class ConvolutionalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, data):
        return self.layer(data)


class ConvTransposeLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=True):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, data):
        return self.layer(data)


def get_outputpadding(input_size, output_size, kernel_size, stride, padding):
    outputpadding = output_size - (input_size - 1) * stride + 2 * padding - kernel_size
    return outputpadding
