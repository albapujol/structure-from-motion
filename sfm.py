import argparse
from parser.bundler import Bundler


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--images', type=int, nargs='+',
                        help='Input images to the SfM algorithm')
    parser.add_argument('--path', type=str,
                        help='Path containing the dataset')

    args = parser.parse_args()

    b = Bundler(args.path)
    b.parse_bundler()

    print(args.images)
    # Read input data

    # Undistort input images

    # Estimate essential matrix

    # Estimate relative pose



if __name__ == '__main__':
    main()