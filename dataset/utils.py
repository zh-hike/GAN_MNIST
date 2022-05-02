from torchvision.transforms import transforms as T
import torch.nn.functional as F


class SetScale(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img = (img - 128) / 128
        return img


def get_default_transform(n_channel=3):
    t = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

    return t
