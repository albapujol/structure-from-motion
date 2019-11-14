"""
Tools to interface with point clouds
"""
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def build_o3d_cloud(points, colors=None):
    """Build an Open3D point cloud from numpy arrays.

    :param points: XYZ coordinates
    :param colors: RGB colors (optional)
    :return: Open3D point cloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None and len(colors) > 0:
        assert len(colors) == len(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def view_cloud(points, colors=None):
    """Visualize a point cloud using Open3D visualizer.

    :param points:  XYZ coordinates
    :param colors:  RGB colors (optional)
    :return: None
    """
    pcd = build_o3d_cloud(points, colors)
    o3d.visualization.draw_geometries([pcd])


def view_cloud_matplotlib(points):
    """Visualize a point cloud using Matplotlib visualizer.

    :param points:  XYZ coordinates
    :return: None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker='.')
    plt.show()


def save_cloud(points, colors=None, path=None):
    """Save a point cloud to the specified file.

    :param points: XYZ coordinates
    :param colors: RGB colors (optional)
    :param path: Output path
    :return: None
    """
    if path is None:
        raise ValueError('Output path not specified')
    pcd = build_o3d_cloud(points, colors)
    o3d.io.write_point_cloud(path, pcd)


def read_cloud(path):
    """Read a point cloud from the specified file.

    :param path: Input path
    :return: Np.array of points, np.array of colors
    """
    if path is None:
        raise ValueError('Input path not specified')
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points), np.asarray(pcd.colors)
