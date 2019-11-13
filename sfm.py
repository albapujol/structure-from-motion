import argparse
import dataset_parser
import utils.stereo_computation
import utils.epipolar_geometry
import sfm.sparse_2cam_reconstruction
import sfm.sparse_Ncam_reconstruction
import numpy as np
import matplotlib.pyplot as plt
import visualization.point_cloud

def print_GT_transform(b, im0, im1):
    R1 = b.R[im0]
    R2 = b.R[im1]
    t1 = b.t[im0]
    t2 = b.t[im1]
    T1 = np.eye(4)
    T1[0:3, 0:3] = R1
    T1[0:3, 3] = t1
    T2 = np.eye(4)
    T2[0:3, 0:3] = R2
    T2[0:3, 3] = t2
    return np.linalg.inv(T1) @ T2


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--images', type=int, nargs='+',
                        help='Input images to the SfM algorithm')
    parser.add_argument('--path', type=str,
                        help='Path containing the dataset')

    args = parser.parse_args()

    bundle = dataset_parser.Bundler(args.path)

    print(bundle.image_paths[0])
    # im0, im1 = args.images[0], args.images[1]
    # points, colors = sfm.sparse_2cam_reconstruction.sparse_2cam_reconstuction(bundle, im0, im1)
    # visualization.point_cloud.view_cloud(points, colors=colors)

    points, colors = sfm.sparse_Ncam_reconstruction.sparse_Ncam_reconstuction(bundle, args.images)
    visualization.point_cloud.view_cloud(points, colors=colors)

    #
    # p1, p2 = bundle.get_corres(args.images[0], args.images[1])
    #
    # print(print_GT_transform(bundle, args.images[0], args.images[1]))
    #
    # ph = np.concatenate((p1, np.ones((len(p1), 1))), axis=1)
    # qh = np.concatenate((p2, np.ones((len(p2), 1))), axis=1)
    #
    # F = utils.epipolar_geometry.estimate_fundamental_matrix(ph, qh)
    # print('Fundamental')
    # print(F)
    #
    # H1, H2 = utils.stereo_computation.stereorectify(F, ph, qh)
    # print('Hs')
    # print(H1)
    # print(H2)
    # print("ends")
    #
    #
    # im1 = b.get_image(args.images[0])
    # im2 = b.get_image(args.images[1])
    # plt.imshow(utils.stereo_computation.apply_H(im1, H2))
    # plt.show()
    # plt.imshow(utils.stereo_computation.apply_H(im2, H1))
    # plt.show()



    # Pp, Pq = estimate_camera_matrices(p1, p2)


    # Read input data

    # Undistort input images

    # Estimate essential matrix

    # Estimate relative pose



if __name__ == '__main__':
    main()