def convert(imgs, dataset):
    if dataset == 'mnist':
        imgs = imgs.view(5, 10, 1, 28, 28)

    return imgs.numpy()
