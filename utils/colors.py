import numpy as np


def get_colors_from_image(image, points):
    """Extract the colors from an image on the locations specified in points

    :param image: Numpy image (RGB or grayscale)
    :param points: XYZ point array
    :return: Colors array (either RGB or Y)
    """
    points = np.array(points)
    if len(points.shape) == 1:
        points = points.reshape(1, 2)
    coord_points = np.array([[k[1] + image.shape[0] / 2, k[0] + image.shape[1] / 2] for k in points], dtype=int)
    if len(image.shape) == 2 or image.shape[2] == 1:
        c = np.array([image[k[0], k[1]] / 255.0 for k in coord_points])
    else:
        c = [image[k[0], k[1], :] / 255.0 for k in coord_points]
    return np.array(c)


def gray_to_RGB(colors):
    if len(colors.shape) == 1:
        colors = colors.reshape(colors.shape[:2])
        return np.tile(colors, (3, 1)).T
    else:
        return colors


def merge_colors(*colors):
    colors_mergeable = [gray_to_RGB(c) for c in colors]
    return np.sum(colors_mergeable, axis=0)/len(colors_mergeable)