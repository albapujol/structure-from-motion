"""
Tools to interface with images
"""
import numpy as np
import matplotlib.pyplot as plt


def view_image(image):
    """Visualize an image using Matplotlib

    :param image: Numpy array with the image matrix
    :return: None
    """
    plt.imshow(image)
    plt.show()


def save_image(image, path=None):
    """Save an image to the specified file.

    :param image: Numpy array with the image matrix
    :param path: Output path
    :return: None
    """
    if path is None:
        raise ValueError('Output path not specified')
    plt.imsave(path, image)


def read_image(path):
    """Read a point cloud from the specified file.

    :param path: Input path
    :return: Np.array containing the image
    """
    if path is None:
        raise ValueError('Input path not specified')
    return plt.imread(path)
