import numpy as np
import matplotlib as mpl
import os


class Colormap:
    """ Represents a colormap to be used for plotting. """
    def __init__(self, filename):
        """ Construct from a spyview colormap. """
        dir = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(dir, filename)

        self.colors = np.loadtxt(path)
        self.gamma = 1
        self.min, self.max = 0, 1

        self.length = self.colors.shape[0]

    def get_limits(self):
        return self.min, self.max

    def get_colors(self):
        """
        After gamma-correcting the colormap curve, return an
        interpolated version of the colormap as 2D integer array.

        This array can be uploaded to the GPU in vispy/opengl as a
        1D texture to be used as a lookup table for coloring the data.
        """
        x = np.linspace(0, 1, self.length)
        y = x**self.gamma

        value = np.linspace(0, 1, len(self.colors))
        r = np.interp(y, value, self.colors[:,0])
        g = np.interp(y, value, self.colors[:,1])
        b = np.interp(y, value, self.colors[:,2])

        return np.dstack((r, g, b)).reshape(len(r), 3).astype(np.uint8)

    def get_mpl_colormap(self):
        """
        Create a matplotlib colormap object that can be used in the cmap
        argument of some matplotlib plotting functions.
        """
        return mpl.colors.ListedColormap(self.get_colors().astype(float) / 255.0)