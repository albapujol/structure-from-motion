import os
import numpy as np
from matplotlib import pyplot as plt


class Bundler:
    """Class to parse files inputted by bundler
    """
    def __init__(self, path):
        self.path = path
        self.image_paths = []
        self.k = None
        self.R = None
        self.t = None
        self.matches = []
        self.parse_bundler()
        self.parse_cameras()

    @property
    def bundle_path(self):
        """Property to get bundle file path
        :return: Bundle file path
        """
        return os.path.join(self.path, 'notredame.out')

    @property
    def image_list_path(self):
        """Get image list path
        :return: Image list path
        """
        return os.path.join(self.path, 'list.txt')

    def parse_cameras(self):
        """Parse camera file into class
        :return: None
        """
        with open(self.image_list_path, 'r') as f:
            self.image_paths = f.readlines()

    def parse_bundler(self):
        """Parse bundler file into class
        :return: None
        """
        with open(self.bundle_path, 'r') as f:
            # Discard initial lines
            l = '#'
            while l.startswith('#'):
                l = f.readline()
            num_cameras, num_points = [int(w) for w in l.split()]
            self.k = np.zeros((num_cameras, 3))
            self.R = np.zeros((num_cameras, 3, 3))
            self.t = np.zeros((num_cameras, 3))
            for elem in range(num_cameras):
                self.k[elem] = [float(w) for w in f.readline().split()]
                self.R[elem][0] = [float(w) for w in f.readline().split()]
                self.R[elem][1] = [float(w) for w in f.readline().split()]
                self.R[elem][2] = [float(w) for w in f.readline().split()]
                self.t[elem] = [float(w) for w in f.readline().split()]
            for elem in range(num_points):
                _ = f.readline() # Discard 3D information
                _ = f.readline() # Discard color information
                viewlist = f.readline().split()
                current_m = {}
                for m in range(int(viewlist[0])):
                    current_m[int(viewlist[m*4+1])] = [float(viewlist[m*4+3]), float(viewlist[m*4+4])]
                self.matches.append(current_m)

    def get_image(self, idx, undistorted=False):
        """Get image from index
        :param idx: Index of the image (int)
        :param undistorted: If the image needs to be previously undistorted
        :return: Numpy array containing the image
        """
        if undistorted:
            raise NotImplementedError
        else:
            return plt.imread(os.path.join(self.path, self.image_paths[idx][:-1]))

    def get_image_size(self, idx):
        """Get image size
        :param idx: Index to get the image size for
        :return: Image size (3-elem)
        """
        return self.get_image(idx).shape

    def get_extrinsics(self, idx):
        """Get extrinsics matrix for index i
        :param idx: Index to get the extrinsics matrix
        :return: Extrinsics matrix
        """
        R = self.R[idx]
        t = self.t[idx]
        T = np.eye(4)
        T[0:3, 0:3] = -R.T
        T[0:3, 3] = t
        return T

    def get_intrinsics(self, idx):
        """Get intrinsics matrix for index i
        This function assumes that the pixels are already centered at image center.
        :param idx: Index to get the extrinsics matrix
        :return: Extrinsics matrix
        """
        f = self.k[idx][0]
        # cx, cy = self.get_image_size(idx)[0:2]/2
        # K = np.array([[f, 0, 0], [0, f, 0], [0, 0, 1]])
        K = np.eye(3)
        return K

    def get_camera_matrix(self, idx):
        """Get camera matrix for index i
        :param idx: Index to get the camera matrix
        :return: Camera matrix
        """
        return self.get_intrinsics(idx) @ self.get_extrinsics(idx)[:-1, :]

    def get_corres(self, idxa, idxb, undistorted=True):
        """ Get correspondences between two images
        Get the correpsondence list between two different images
        :param idxa: Index of the image A
        :param idxb: Index of the image B
        :param undistorted: If the points are undistorted
        :return: Arrays of matching points
        """
        pa = []
        pb = []
        for m in self.matches:
            if idxa in m and idxb in m:
                    pa.append(m[idxa])
                    pb.append(m[idxb])
        if undistorted:
            return (Bundler._undistort(np.array(pa), self.k[idxa]),
                    Bundler._undistort(np.array(pb), self.k[idxb]))
        else:
            return np.array(pa), np.array(pb)

    def get_corres_n(self, idxes, undistorted=True):
        """ Get correspondences between N images
        Get the correpsondence list between two different images
        :param idxes: Index of the images
        :param undistorted: If the points are undistorted
        :return: Arrays of matching points
        """
        pouts = [[] for _ in idxes]
        for m in self.matches:
            if all([i in m for i in idxes]):
                for q, i in enumerate(idxes):
                    pouts[q].append(m[i])
        if undistorted:
            return [Bundler._undistort(np.array(p), self.k[i]) for p, i in zip(pouts, idxes)]
        else:
            return [np.array(p) for p in pouts]


    @staticmethod
    def _undistort(points, k):
        """
        Undistort a list of points given the parameters K
        :param points: Array of points to be undistorted
        :param k: Intrinsic parameters in the format [f, k1, k2]
        :return: Undistorted points
        """
        if len(points) == 0:
            return points
        points /= k[0]
        rp = 1.0 + k[1] * (points[:, 0]**2+points[:, 1]**2) + k[2] * (points[:, 0]**2+points[:, 1]**2)**2
        return points * rp.reshape(len(points), 1)