import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
import cv2
import torchvision


def to_open_cv(im):
    return np.transpose((255. * im).astype(np.uint8), (1, 2, 0))

def display_batch(batch, display=True):
    cls_ims = []
    for cls in batch:
        im = torchvision.utils.make_grid(cls)
        im = im.cpu().numpy()
        im = to_open_cv(im)
        cls_ims.append(im)
    im = np.vstack(cls_ims)
    if display:
        plt.imshow(im)
        plt.show()
    else:
        return im

def save_recons_few_shot(x, xr, fname):
    im = display_batch(x, False)
    im_r = display_batch(xr, False)
    cv2.imwrite("{}_orig.png".format(fname), im)
    cv2.imwrite("{}_recon.png".format(fname), im_r)

def imshow(img):
    npimg = img.numpy()
    plt.imshow(to_open_cv(npimg))
    plt.show()


def save_recons(x, xr, fname):
    """
    :param x: input tensor
    :param xr: reconstructed input tensor
    :param fname: filename to save to
    :return:
    """
    im = to_open_cv(torchvision.utils.make_grid(x).numpy())
    im_r = to_open_cv(torchvision.utils.make_grid(xr).numpy())
    cv2.imwrite("{}_orig.png".format(fname), im)
    cv2.imwrite("{}_recon.png".format(fname), im_r)
