import argparse
from dataset_parser.bundler import Bundler
import opencv_utils.stereo_computation
import opencv_utils.epipolar_geometry
import utils.epipolar_geometry
import cv2
import numpy as np
import matplotlib.pyplot as plt


def GT_extrinsics(b, im):
    R = b.R[im]
    t = b.t[im]
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    return T



def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--images', type=int, nargs='+',
                        help='Input images to the SfM algorithm')
    parser.add_argument('--path', type=str,
                        help='Path containing the dataset')

    args = parser.parse_args()

    b = Bundler(args.path)

    p1, p2 = b.get_corres(args.images[0], args.images[1])

    T1 = GT_extrinsics(b, args.images[0], args.images[1])
    T2 = GT_extrinsics(b, args.images[0], args.images[1])

    im1 = b.get_image(args.images[0])
    im2 = b.get_image(args.images[1])

    R = trans[0:3, 0:3]
    K1 = np.array([[b.k[args.images[0]][0], 0, im1.shape[0]/2],
                   [0, b.k[args.images[0]][0], im1.shape[1]/2],
                   [0, 0, 1]])
    K2 = np.array([[b.k[args.images[1]][0], 0, im2.shape[0]/2],
                   [0, b.k[args.images[1]][0], im2.shape[1]/2],
                   [0, 0, 1]])

    H1 = K1 @ R
    H2 = K2 @ R
    print('Hs')
    print(H1)
    print(H2)
    print("ends")


    point3ds = utils.epipolar_geometry.triangulate(p1, p22, P1, P2)


    ph = np.concatenate((p1, np.ones((len(p1), 1))), axis=1)
    qh = np.concatenate((p2, np.ones((len(p2), 1))), axis=1)

    F = cv2.findFundamentalMat(p1, p2)[0]
    #
    # H1, H2 = opencv_utils.stereo_computation.stereorectify(F, ph, qh)
    print('Hs')
    print(H1)
    print(H2)
    print("ends")


    im1 = b.get_image(args.images[0])
    im2 = b.get_image(args.images[1])

    plt.imshow(opencv_utils.stereo_computation.apply_H(im1, H2))
    plt.show()
    plt.imshow(opencv_utils.stereo_computation.apply_H(im2, H1))
    plt.show()



    # Pp, Pq = estimate_camera_matrices(p1, p2)


    # Read input data

    # Undistort input images

    # Estimate essential matrix

    # Estimate relative pose



if __name__ == '__main__':
    main()