# Structure From Motion (SfM) reconstruction library

This library provides a toolset to perform SfM reconstructions on the Photo Tourism dataset (or datasets with similar formats). Performed as a "weekend" coding test.


## Requeriments
This library is tested using Python 3.6.8. The full list of packages can be found in `requirements.txt`. 

### Dataset requeriments
This library is developed to be used in the Photo Tourism dataset or datasets with a similar format. The command line tool is given the path to the `*.out` file generated by [Bundler](http://www.cs.cornell.edu/~snavely/bundler/bundler-v0.3-manual.html#S6) and must follow the same file structure:
```
<dataset_folder>
├── images
├── list.txt
└── <bundle_file>.out
```

## Basic usage

Some basic commands that can be performed using this library are:

- Reconstruction using two images, view on a window and save the result on the file `reconstruction.ply`.

```bash
$  python sfm.py --path ~/Sources/coding_test/NotreDame/notredame.out --images 1 2 --colorize --view --save reconstruction.ply
```

- Reconstruction using four images on a pairwise reconstruction, view on a window and save the result on the file `reconstruction.ply`.

```bash
$ python sfm.py --path ~/Sources/coding_test/NotreDame/notredame.out --images 1 2 3 4 --colorize --reconstruction_method pair --view --save reconstruction.ply
```

- Reconstruction using four images on a bundle reconstruction, view on a window and save the result on the file `reconstruction.ply`.

```bash
$ python sfm.py --path ~/Sources/coding_test/NotreDame/notredame.out --images 1 2 3 4 --colorize --view --save reconstruction.ply
```

- Reconstruction using four images on a bundle reconstruction, view on a window and save the result on the file `reconstruction.ply`, using an universal interface.

```bash
python sfm.py --path ~/Sources/coding_test/NotreDame/notredame.out --images 1 2 3 4 --demo --view --save reconstruction.ply
```

## Parameters
The full list of parameters can be found using `python sfm.py --help`.
```bash       
usage: sfm.py [-h] [--path PATH] [--images IMAGES [IMAGES ...]]
              [--reconstruction_type RECONSTRUCTION_TYPE]
              [--reconstruction_method RECONSTRUCTION_METHOD] [--view]
              [--colorize] [--save SAVE] [--demo]

Structure From Motion coding test uding the Photo Tourism dataset.

optional arguments:
  -h, --help            show this help message and exit
  --path PATH           Path to the *.out file containing the bundler data
  --images IMAGES [IMAGES ...]
                        Input images to the SfM algorithm (int indexes). The
                        method will fail if a number larger than the dataset
                        size is specified
  --reconstruction_type RECONSTRUCTION_TYPE
                        Type of reconstruction: 'dense' or 'sparse'. Defaults
                        to 'sparse'
  --reconstruction_method RECONSTRUCTION_METHOD
                        Method to perform the reconstruction: either 'pair' or
                        'bundle'. Defaults to 'bundle'
  --view                View the resulting point cloud in an X window
  --colorize            Colorize the point cloud
  --save SAVE           Save the resulting point cloud on the specified file
  --demo                Execute demo registration with proposed interface
```

## Proposed interface
The global interface that can be used with any data source is the follwoing:
```python
points, colors = sfm.sparse_reconstruction.sparse_bundle_reconstruction_data(images, pair_dict, camera_matrices, ks, colorize=True)
```
This interface needs the following parameters:
* `images` List of numpy arrays with all the images.
* `pair_dict` List of dictionaries with all the matches. For an example of how to create such list of dictionaries, see `sfm.py`. 
* `camera_matrices` List containing all the 3x4 camera matrices (intrinsics+extrinsics).
* `ks` List containing all the intrinsic parameters for each image in the format `[f, k1, k2]`.
* `colorize` Optional parameter to indicate if the colors are also extracted. Defaults to `True`.


## Changelog 

V0.1 - Initial version with only sparse reconstruction working.