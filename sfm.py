import argparse

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--images', type=str, nargs='+',
                        help='Input images to the SfM algorithm')
    parser.add_argument('--correspondences', type=str,
                        help='File containing ')

    args = parser.parse_args()

    # Read input data

    # Undistort input images

    # Estimate essential matrix

    # Estimate relative pose



if __name__ == '__main__':
    main()