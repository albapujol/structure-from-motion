import argparse
from parser.bundler import Bundler
from utils.epipolar_geometry import estimate_camera_matrices


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--images', type=int, nargs='+',
                        help='Input images to the SfM algorithm')
    parser.add_argument('--path', type=str,
                        help='Path containing the dataset')

    args = parser.parse_args()

    b = Bundler(args.path)

    print(args.images)

    p1, p2 = b.get_corres(args.images[0], args.images[1])
    im1 = b.get_image(args.images[0])
    im2 = b.get_image(args.images[1])
    Pp, Pq = estimate_camera_matrices(p1, p2, b.k[args.images[0], 0], b.k[args.images[1], 0], im1.shape[0:2], im2.shape[0:2])


    # Read input data

    # Undistort input images

    # Estimate essential matrix

    # Estimate relative pose



if __name__ == '__main__':
    main()