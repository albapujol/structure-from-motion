import utils.epipolar_geometry
import utils.colors
import numpy as np



def _pair_color(bundle, im0, im1):
    p0d, p1d = bundle.get_corres(im0, im1, undistorted=False)
    image0 = bundle.get_image(im0)
    image1 = bundle.get_image(im1)
    c0 = utils.colors.get_colors_from_image(image0, p0d)
    c1 = utils.colors.get_colors_from_image(image1, p1d)
    colors = utils.colors.merge_colors(c0, c1)
    return colors



def sparse_2cam_reconstuction(bundle, im0, im1, colorize=True):
    p0, p1 = bundle.get_corres(im0, im1)
    P0 = bundle.get_camera_matrix(im0)
    P1 = bundle.get_camera_matrix(im1)
    points = []
    for p, q in zip(p0, p1):
        points.append(utils.epipolar_geometry.triangulate2(p, q, P0, P1))
    points = np.array(points)
    if colorize:
        colors = _pair_color(bundle, im0, im1)
    else:
        colors = None
    return points, colors


def sparse_Ncam_reconstuction(bundle, image_list, colorize=True):
    points = []
    colors = np.array([]).reshape((0, 3))
    for im0 in image_list:
        for im1 in image_list:
            if im0 >=im1:
                continue
            p0, p1 = bundle.get_corres(im0, im1)
            if len(p0) == 0:
                continue
            p0h = np.concatenate((p0, np.ones((len(p0), 1))), axis=1)
            p1h = np.concatenate((p1, np.ones((len(p1), 1))), axis=1)
            P0 = bundle.get_camera_matrix(im0)
            P1 = bundle.get_camera_matrix(im1)
            for p, q in zip(p0h, p1h):
                points.append(utils.epipolar_geometry.triangulate2(p, q, P0, P1))
            if colorize:
                colors = np.concatenate((colors, _pair_color(bundle, im0, im1)))
    points = np.array(points)
    return points, colors


def sparse_bundler_reconstuction(bundle, image_list, colorize=True):
    points = []
    colors = np.array([]).reshape((0, 3))
    image_list = set(image_list)
    for corres_points in bundle.matches:
        image_set_in_corres = image_list.intersection(corres_points.keys())
        if len(image_set_in_corres) < 2:
            continue
        query_points_it = np.zeros((len(image_set_in_corres), 2))
        color_points_it = []
        Ps= []
        for i, key in enumerate(image_set_in_corres):
            coords_und = bundle.undistort(np.array([corres_points[key]]), bundle.k[key]).flatten()
            query_points_it[i] = coords_und
            Ps.append(bundle.get_camera_matrix(key))
            image = bundle.get_image(key)
            if colorize:
                color_points_it.append(utils.colors.get_colors_from_image(image, corres_points[key]))
        points.append(utils.epipolar_geometry.triangulateN(query_points_it, Ps))
        if colorize:
            colors = np.concatenate((colors, utils.colors.merge_colors(*color_points_it)))
    return np.array(points), np.array(colors)
