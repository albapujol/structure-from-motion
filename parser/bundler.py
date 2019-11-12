import os
import numpy as np
from matplotlib import pyplot as plt

class Bundler():
    """
    Class to parse files inputted by bundler
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
        return os.path.join(self.path, 'notredame.out')

    @property
    def image_list_path(self):
        return os.path.join(self.path, 'list.txt')

    def parse_cameras(self):
        with open(self.image_list_path, 'r') as f:
            self.image_paths = f.readlines()

    def parse_bundler(self):
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

    def get_image(self, idx):
        return plt.imread(os.path.join(self.path, self.image_paths[idx][:-1]))



    def get_corres(self, idxa, idxb):
        pa = []
        pb = []
        for m in self.matches:
            if idxa in m and idxb in m:
                pa.append(m[idxa])
                pb.append(m[idxb])
        return np.array(pa), np.array(pb)

        pass
