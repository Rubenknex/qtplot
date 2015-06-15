import numpy as np
import matplotlib as mpl

def read_ppm_colormap(filename):
    with open(filename) as f:
        magic = f.readline()
        size_x, size_y = map(int, f.readline().split())
        max_val = int(f.readline())

        hexstring = f.readline()
        pixels = np.fromstring(hexstring, dtype=np.uint8)

        return pixels.reshape((len(pixels) / 3, 3))



class Colormap:
	def __init__(self, filename):
		self.colors = read_ppm_colormap(filename)
		self.gamma = 1
		self.min, self.max = 0, 1

		self.length = 2048

	def get_limits(self):
		return self.min, self.max

	def get_colors(self):
		x = np.linspace(0, 1, self.length)
		y = x**self.gamma

		colors_x = np.linspace(0, 1, len(self.colors))
		r = np.interp(y, colors_x, self.colors[:,0])
		g = np.interp(y, colors_x, self.colors[:,1])
		b = np.interp(y, colors_x, self.colors[:,2])
		
		return np.dstack((r, g, b)).reshape(len(r), 3).astype(np.uint8)

	def get_mpl_colormap(self):
		return mpl.colors.ListedColormap(self.get_colors().astype(float) / 255.0)