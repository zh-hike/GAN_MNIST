from matplotlib import pyplot as plt
from utils import convert
import numpy as np


def plot_img(imgs, dataset, epoch):
    imgs = convert(imgs, dataset)
    rows, cols, _, _, _ = imgs.shape
    fig = plt.figure(figsize=(cols * 3, rows * 3))
    for row in range(rows):
        for col in range(cols):
            plt.subplot(rows, cols, rows * col + 1 + row)
            img = imgs[row, col]
            img = img.transpose([1, 2, 0])
            img = np.clip(img/2+0.5, 0, 1)
            img = (img * 255).squeeze()
            plt.axis('off')
            if img.ndim == 2:
                plt.imshow(img.astype('int'), cmap='gray')
            else:
                plt.imshow(img.astype('int'))

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('results/val/%d.png' % epoch, dpi=400)
    plt.close()

