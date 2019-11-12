

class Bundler():
    """
    Class to parse files inputted by bundler
    """
    def __init__(self, bundler_file, imagelist_file):
        self.bundler_file = bundler_file
        self.imagelist_file = imagelist_file
        self.image_paths = []
        self.k = []
        self.R = []
        self.T = []
        self.matches = []

    def parse_bundler(self):
        pass


    def get_image(self, idx):
        return read_image(self.image_paths[idx])


    def get_corres(self, idxa, idxb):
        pass
