import argparse
import dataset_parser
import utils.stereo_computation
import utils.epipolar_geometry
import sfm.sparse_reconstruction

import numpy as np
import matplotlib.pyplot as plt
import image_cloud_io.point_cloud


def create_points(matches, idxes):
    out_matches = []
    for m_dict in matches:
        d = {i: m_dict[idx] for i, idx in enumerate(idxes) if idx in m_dict}
        if len(d) >= 2:
            out_matches.append(d)
    return out_matches


def main():
    parser = argparse.ArgumentParser(description='Structure From Motion coding test uding the Photo Tourism dataset.')
    parser.add_argument('--path', type=str,
                        help='Path to the *.out file containing the bundler data')
    parser.add_argument('--images', type=int, nargs='+',
                        help='Input images to the SfM algorithm (int indexes). '
                             'The method will fail if a number larger than the dataset size is specified')
    parser.add_argument('--reconstruction_type', type=str, default='sparse',
                        help="Type of reconstruction: 'dense' or 'sparse'. Defaults to 'sparse'")
    parser.add_argument('--reconstruction_method', type=str, default='bundle',
                        help="Method to perform the reconstruction: either 'pair' or 'bundle'. Defaults to 'bundle'")
    parser.add_argument('--view', action='store_true',
                        help="View the resulting point cloud in an X window")
    parser.add_argument('--colorize', action='store_true',
                        help="Colorize the point cloud")
    parser.add_argument('--save', type=str, default='',
                        help="Save the resulting point cloud on the specified file")
    parser.add_argument('--demo', action='store_true',
                        help="Execute demo registration with proposed interface")

    args = parser.parse_args()

    # Exit if not enough images are given
    if len(args.images) < 2:
        print("Expected at least 2 image indexes to perform the registration")
        return -1

    if not args.view and not args.save:
        print("WARNING: not view or save path specified. The algorithm will not output anything")

    bundle = dataset_parser.Bundler(args.path)

    # DENSE RECONSTRUCTION
    if args.reconstruction_type == 'dense':
        raise NotImplementedError("The dense reconstruction is not implemented")

    # SPARSE RECONSTRUCTION
    elif args.reconstruction_type == 'sparse':
        if args.demo:
            bundle = dataset_parser.Bundler(args.path)
            img_idxes = args.images
            images = [bundle.get_image(i) for i in img_idxes]
            camera_matrices = [bundle.get_camera_matrix(i) for i in img_idxes]
            ks = [bundle.k[i] for i in img_idxes]
            pair_dict = create_points(bundle.matches, img_idxes)
            points, colors = sfm.sparse_reconstruction.sparse_bundle_reconstruction_data(images, pair_dict,
                                                                                         camera_matrices, ks,
                                                                                         colorize=True)
        elif len(args.images) == 2:
            im0, im1 = args.images[0], args.images[1]
            print("Performing 2 image sparse reconstruction with images %s and %s" % (im0, im1))
            points, colors = sfm.sparse_reconstruction.sparse_2cam_reconstuction(bundle, im0, im1,
                                                                                 colorize=args.colorize)
        else:
            if args.reconstruction_method == 'pair':
                print("Performing pairwise sparse reconstruction")
                points, colors = sfm.sparse_reconstruction.sparse_Ncam_reconstuction(bundle, args.images,
                                                                                 colorize=args.colorize)
            elif args.reconstruction_method == 'bundle':
                print("Performing bundle sparse reconstruction")
                points, colors = sfm.sparse_reconstruction.sparse_bundle_reconstuction(bundle, args.images,
                                                                                       colorize=args.colorize)
            else:
                print("Method not understood")
                return -1
        print("Reconstructed %s points" % (len(points)))
        if args.save:
            image_cloud_io.point_cloud.save_cloud(points, colors=colors, path=args.save)
        if args.view:
            image_cloud_io.point_cloud.view_cloud(points, colors=colors)


if __name__ == '__main__':
    main()
