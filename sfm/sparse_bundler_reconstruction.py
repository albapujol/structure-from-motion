import utils.epipolar_geometry
import numpy as np


def sparse_bundler_reconstuction(bundle, max_points=100):
    # corres_array = [[bundle.get_corres(a, b) for a in image_list] for b in image_list]
    # corres_array_dist = [[bundle.get_corres(a, b, undistorted=False) for a in image_list] for b in image_list]

    points = []
    colors = []

    for corres_points in bundle.matches:
        query_points_it = np.zeros((min(len(corres_points), 2), 2))
        colors_it = np.zeros((min(len(corres_points), 2), 3))
        Ps= []
        for i, (idx, coords) in enumerate(corres_points.items()):
            if len(query_points_it) <= i:
                break
            coords_und = bundle.undistort(np.array([coords]), bundle.k[idx]).flatten()
            query_points_it[i] = coords_und
            Ps.append(bundle.get_camera_matrix(idx))
            if len(colors_it) <= i:
                continue
            image = bundle.get_image(idx)
            if len(image.shape) == 2 or image.shape[2] == 1:
                colors_it[i, 0] = image[int(coords[1] + image.shape[0] / 2), int(coords[0] + image.shape[1] / 2)]
                colors_it[i, 1] = image[int(coords[1] + image.shape[0] / 2), int(coords[0] + image.shape[1] / 2)]
                colors_it[i, 2] = image[int(coords[1] + image.shape[0] / 2), int(coords[0] + image.shape[1] / 2)]
            else:
                colors_it[i] = image[int(coords[1] + image.shape[0] / 2), int(coords[0] + image.shape[1] / 2), :]
        points.append(utils.epipolar_geometry.triangulateN(query_points_it, Ps))
        colors.append(np.average(colors_it, axis=0)/255.0)
        if len(points) >= max_points:
            break
    return np.array(points), np.array(colors)


    # for im0 in image_list:
    #     for im1 in image_list:
    #         if im0 >=im1:
    #             continue
    #         p0, p1 = bundle.get_corres(im0, im1)
    #         if len(p0) == 0:
    #             continue
    #         # print(im0, im1)
    #         p0d, p1d = bundle.get_corres(im0, im1, undistorted=False)
    #         p0h = np.concatenate((p0, np.ones((len(p0), 1))), axis=1)
    #         p1h = np.concatenate((p1, np.ones((len(p1), 1))), axis=1)
    #         P0 = bundle.get_camera_matrix(im0)
    #         P1 = bundle.get_camera_matrix(im1)
    #         for p, q in zip(p0h, p1h):
    #             points.append(utils.epipolar_geometry.triangulate2(p, q, P0, P1))
    #         image0 = bundle.get_image(im0)
    #         image1 = bundle.get_image(im1)
    #         idxes0 = np.array([[k[1]+image0.shape[0]/2, k[0]+image0.shape[1]/2] for k in p0d], dtype=int)
    #         idxes1 = np.array([[k[1]+image1.shape[0]/2, k[0]+image1.shape[1]/2] for k in p1d], dtype=int)
    #         c0 = [image0[k[0], k[1], :]/255.0 for k in idxes0]
    #         c1 = [image1[k[0], k[1], :]/255.0 for k in idxes1]
    #         for x0, x1 in zip(c0, c1):
    #             colors.append((x0+x1)/2)
    # points = np.array(points)
    # colors = np.array(colors)
    # return points, colors
