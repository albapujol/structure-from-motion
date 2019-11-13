import utils.epipolar_geometry
import numpy as np


def sparse_2cam_reconstuction(bundle, im0, im1):
    p0, p1 = bundle.get_corres(im0, im1)
    p0d, p1d = bundle.get_corres(im0, im1, undistorted=False)
    p0h = np.concatenate((p0, np.ones((len(p0), 1))), axis=1)
    p1h = np.concatenate((p1, np.ones((len(p1), 1))), axis=1)

    P0 = bundle.get_camera_matrix(im0)
    P1 = bundle.get_camera_matrix(im1)
    points = []
    for p, q in zip(p0h, p1h):
        points.append(utils.epipolar_geometry.triangulate2(p, q, P0, P1))
    points = np.array(points)

    image0 = bundle.get_image(im0)
    image1 = bundle.get_image(im1)

    idxes0 = np.array([[k[0]+image0.shape[0]/2, k[1]+image0.shape[1]/2] for k in p0d], dtype=int)
    idxes1 = np.array([[k[0]+image1.shape[0]/2, k[1]+image1.shape[1]/2] for k in p1d], dtype=int)

    c0 = np.array([image0[k[0], k[1], :] for k in idxes0])/255.0
    c1 = np.array([image1[k[0], k[1], :] for k in idxes1])/255.0
    colors = (c0+c1)/2
    return points, colors
