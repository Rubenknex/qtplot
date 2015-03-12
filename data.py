import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import qhull
import math
import matplotlib.pyplot as plt

class DatFile:
    """Class which contains the column based DataFrame of the data."""
    def __init__(self, filename):
        self.filename = filename

        self.columns = []

        with open(filename, 'r') as f:
            for line in f:
                line = line.rstrip('\n\t\r')

                if line.startswith('#\tname'):
                    name = line.split(': ', 1)[1]
                    self.columns.append(name)

        self.df = pd.read_table(filename, engine='c', sep='\t', comment='#', names=self.columns)

    def get_data(self, x, y, z, x_order, y_order):
        """Pivot the column based data into matrices."""
        # Sometimes there are multiple datapoints for the same coordinate, but there is only one coordinate axis
        # In this case, fill the second coordinate with 1,2,3,... so the datapoints can be plotted in 2D
        if (self.df[x_order] == 0).all():
            self.df[x_order] = self.df.groupby(y_order)[x_order].apply(lambda x: pd.Series(range(len(x.values)), x.index))

        if (self.df[y_order] == 0).all():
            self.df[y_order] = self.df.groupby(x_order)[y_order].apply(lambda x: pd.Series(range(len(x.values)), x.index))

        x_coords = self.df.pivot(y_order, x_order, x).values
        y_coords = self.df.pivot(y_order, x_order, y).values
        values   = self.df.pivot(y_order, x_order, z).values

        return Data(x_coords, y_coords, values, (x==x_order,y==y_order))

def create_kernel(x_dev, y_dev, cutoff, distr):
    distributions = {
        'gaussian': lambda r: np.exp(-(r**2) / 2.0),
        'exponential': lambda r: np.exp(-abs(r) * np.sqrt(2.0)),
        'lorentzian': lambda r: 1.0 / (r**2+1.0),
        'thermal': lambda r: np.exp(r) / (1 * (1+np.exp(r))**2)
    }
    func = distributions[distr]

    hx = math.floor((x_dev * cutoff) / 2.0)
    hy = math.floor((y_dev * cutoff) / 2.0)

    x = np.linspace(-hx, hx, hx * 2 + 1) / x_dev
    y = np.linspace(-hy, hy, hy * 2 + 1) / y_dev

    if x.size == 1:
        x = np.zeros(1)
    if y.size == 1:
        y = np.zeros(1)
    
    xv, yv = np.meshgrid(x, y)

    kernel = func(np.sqrt(xv**2+yv**2))
    kernel /= np.sum(kernel)

    return kernel

class Data:
    """Class which represents 2d data as two matrices with x and y coordinates and one with values."""
    def __init__(self, x_coords, y_coords, values, equidistant=(False, False)):
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.values = values

        self.equidistant = equidistant
        self.tri = None

    def interpolate(self, points):
        if self.tri == None:
            xc = self.x_coords.flatten()
            yc = self.y_coords.flatten()
            self.no_nan_values = self.values.flatten()

            if np.isnan(xc).any() and np.isnan(yc).any():
                xc = xc[~np.isnan(xc)]
                yc = yc[~np.isnan(yc)]
                self.no_nan_values = self.no_nan_values[~np.isnan(self.no_nan_values)]

            # Default: Qbb Qc Qz 
            self.tri = qhull.Delaunay(np.column_stack((xc, yc)), qhull_options='')

        simplices = self.tri.find_simplex(points)

        indices = np.take(self.tri.simplices, simplices, axis=0)
        transforms = np.take(self.tri.transform, simplices, axis=0)
        
        delta = points - transforms[:,2]
        bary = np.einsum('njk,nk->nj', transforms[:,:2,:], delta)

        temp = np.hstack((bary, 1-bary.sum(axis=1, keepdims=True)))

        values = np.einsum('nj,nj->n', np.take(self.no_nan_values, indices), temp)
        values[np.any(temp < 0, axis=1)] = 0

        return values

    def get_sorted(self):
        """Return the data sorted so that every coordinate increases."""
        x_indices = np.argsort(self.x_coords[0,:])
        y_indices = np.argsort(self.y_coords[:,0])

        return self.x_coords[:,x_indices], self.y_coords[y_indices,:], self.values[:,x_indices][y_indices,:]

    def get_quadrilaterals(self, xc, yc):
        """
        In order to generate quads for every datapoint we do the following for the x and y coordinates:
        -   Pad the coordinates with a column/row on each side
        -   Add the difference between all the coords divided by 2 to the coords, this generates midpoints
        -   Add a row/column at the end to satisfy the 1 larger requirements of pcolor
        """
        # If we are dealing with data that is 2-dimensional
        if len(xc[0,:]) > 1:
            # Pad both sides with a column of interpolated coordinates
            xc = np.hstack((xc[:,[0]] - (xc[:,[1]] - xc[:,[0]]), xc, xc[:,[-1]] + (xc[:,[-1]] - xc[:,[-2]])))
            # Create center points by adding the differences divided by 2 to the original coordinates
            x = xc[:,:-1] + np.diff(xc, axis=1) / 2.0
            # Add a row to the bottom so that the x coords have the same dimension as the y coords
            x = np.vstack((x, x[-1]))
        else:
            # If data is 1d, make one axis range from -.5 to .5
            x = np.hstack((xc - 0.5, xc[:,[0]] + 0.5))
            # Duplicate the only row/column so that pcolor has something to actually plot
            x = np.vstack((x, x[0]))
        
        if len(yc[:,0]) > 1:
            yc = np.vstack([yc[0] - (yc[1] - yc[0]), yc, yc[-1] + (yc[-1] - yc[-2])])
            y = yc[:-1,:] + np.diff(yc, axis=0) / 2.0
            y = np.hstack([y, y[:,[-1]]])
        else:
            y = np.vstack([yc - 0.5, yc[0] + 0.5])
            y = np.hstack([y, y[:,[0]]])

        return x, y

    def get_pcolor(self):
        """
        Return a version of the coordinates and values that can be plotted by pcolor, this means:
        -   Points are sorted by increasing coordinates
        -   Coordinates are converted to coordinates of quadrilaterals
        -   NaN values are masked to ignore them when plotting
        """
        xc, yc, c = self.get_sorted()

        x, y = self.get_quadrilaterals(xc, yc)

        return np.ma.masked_invalid(x), np.ma.masked_invalid(y), np.ma.masked_invalid(c)

    def get_column_at(self, x):
        x_index = np.where(self.x_coords[0,:]==self.get_closest_x(x))[0][0]

        if self.equidistant[0]:
            return self.y_coords[:,x_index], self.values[:,x_index]
        else:
            return self.y_coords[:,x_index], self.values[:,x_index]

    def get_row_at(self, y):
        y_index = np.where(self.y_coords[:,0]==self.get_closest_y(y))[0][0]

        if self.equidistant[1]:
            return self.x_coords[y_index], self.values[y_index]
        else:
            return self.x_coords[y_index], self.values[y_index]

    def get_closest_x(self, x_coord):
        return min(self.x_coords[0,:], key=lambda x:abs(x - x_coord))

    def get_closest_y(self, y_coord):
        return min(self.y_coords[:,0], key=lambda y:abs(y - y_coord))

    def get_dimensions(self):
        return np.min(self.x_coords), np.max(self.x_coords), np.min(self.y_coords), np.max(self.y_coords)

    def flip_axes(self, x_flip, y_flip):
        if x_flip:
            self.x_coords = np.fliplr(self.x_coords)
            self.y_coords = np.fliplr(self.y_coords)
            self.values = np.fliplr(self.values)

        if y_flip:
            self.x_coords = np.flipud(self.x_coords)
            self.y_coords = np.flipud(self.y_coords)
            self.values = np.flipud(self.values)

    def is_flipped(self):
        x_flip = self.x_coords[0,0] > self.x_coords[0,-1]
        y_flip = self.y_coords[0,0] > self.y_coords[-1,0]

        return x_flip, y_flip

    def copy(self):
        return Data(np.copy(self.x_coords), np.copy(self.y_coords), np.copy(self.values), self.equidistant)

    def abs(data, **kwargs):
        """Take the absolute value of every datapoint."""
        return Data(data.x_coords, data.y_coords, np.absolute(data.values), data.equidistant)

    def autoflip(data, **kwargs):
        """Flip the data so that the X and Y-axes increase to the top and right."""
        copy = data.copy()

        copy.flip_axes(*copy.is_flipped())
        
        return copy

    def crop(data, **kwargs):
        """Crop a region of the data by the columns and rows."""
        x1, x2 = int(kwargs.get('Left')), int(kwargs.get('Right'))
        y1, y2 = int(kwargs.get('Bottom')), int(kwargs.get('Top'))

        if x2 == -1: x2 = data.values.shape[1] 
        if y2 == -1: y2 = data.values.shape[0] 

        copy = data.copy()
        copy.x_coords = copy.x_coords[y1:y2,x1:x2]
        copy.y_coords = copy.y_coords[y1:y2,x1:x2]
        copy.values = copy.values[y1:y2,x1:x2]

        return copy

    def dderiv(data, **kwargs):
        """Calculate the component of the gradient in a specific direction."""
        xcomp = Data.xderiv(data)
        ycomp = Data.yderiv(data)

        xvalues = xcomp.values[:-1,:]
        yvalues = ycomp.values[:,:-1]

        theta = np.radians(float(kwargs.get('Theta')))
        xdir, ydir = np.cos(theta), np.sin(theta)

        return Data(xcomp.x_coords[:-1,:], ycomp.y_coords[:,:-1], xvalues * xdir + yvalues * ydir, data.equidistant)

    def equalize(data, **kwargs):
        """Perform histogramic equalization on the image."""
        copy = data.copy()

        binn = 65535

        # Create a density histogram with surface area 1
        hist, bins = np.histogram(copy.values.flatten(), binn)
        cdf = hist.cumsum()

        cdf = bins[0] + (bins[-1]-bins[0]) * (cdf / float(cdf[-1]))

        new = np.interp(copy.values.flatten(), bins[:-1], cdf)
        copy.values = np.reshape(new, data.values.shape)

        return copy

    def even_odd(data, **kwargs):
        """Extract even or odd rows, optionally flipping odd rows."""
        even = bool(kwargs.get('Even'))

        copy = data.copy()
        indices = np.arange(0, data.values.shape[0], 2)
        if not even: indices = np.arange(1, data.values.shape[0], 2)

        copy.values = copy.values[indices]
        copy.x_coords = copy.x_coords[indices]
        copy.y_coords = copy.y_coords[indices]

        return copy

    def flip(data, **kwargs):
        """Flip the X or Y axes."""
        copy = data.copy()

        if bool(kwargs.get('X Axis')):
            copy.flip_axes(True, False)

        if bool(kwargs.get('Y Axis')):
            copy.flip_axes(False, True)

        return copy

    def gradmag(data, **kwargs):
        """Calculate the length of every gradient vector."""
        xcomp = Data.xderiv(data)
        ycomp = Data.yderiv(data)

        xvalues = xcomp.values[:-1,:]
        yvalues = ycomp.values[:,:-1]

        return Data(xcomp.x_coords[:-1,:], ycomp.y_coords[:,:-1], np.sqrt(xvalues**2 + yvalues**2), data.equidistant)

    def highpass(data, **kwargs):
        """Perform a high-pass filter."""
        copy = data.copy()

        # X and Y sigma order?
        sx, sy = float(kwargs.get('X Width')), float(kwargs.get('Y Height'))
        kernel_type = str(kwargs.get('Type')).lower()

        kernel = create_kernel(sx, sy, 7, kernel_type)
        copy.values = copy.values - ndimage.filters.convolve(copy.values, kernel)

        copy.values = np.ma.masked_invalid(copy.values)

        return copy

    def hist2d(data, **kwargs):
        """Convert every column into a histogram, default bin amount is sqrt(n)."""
        hmin, hmax = float(kwargs.get('Min')), float(kwargs.get('Max'))
        hbins = int(kwargs.get('Bins'))
        
        if hmin == -1: hmin = np.min(data.values)
        if hmax == -1: hmax = np.max(data.values)

        hist = np.apply_along_axis(lambda x: np.histogram(x, bins=hbins, range=(hmin, hmax))[0], 0, data.values)

        binedges = np.linspace(hmin, hmax, hbins + 1)
        bincoords = (binedges[:-1] + binedges[1:]) / 2

        xcoords = np.tile(data.x_coords[0,:], (hist.shape[0], 1))
        ycoords = np.tile(bincoords[:,np.newaxis], (1, hist.shape[1]))
        
        return Data(xcoords, ycoords, hist, equidistant=(True, True))

    def interp_grid(data, **kwargs):
        """Interpolate the data onto a uniformly spaced grid using barycentric interpolation."""
        width, height = int(kwargs.get('Width')), int(kwargs.get('Height'))
        xmin, xmax, ymin, ymax = data.get_dimensions()

        x = np.linspace(xmin, xmax, width)
        y = np.linspace(ymin, ymax, height)
        xv, yv = np.meshgrid(x, y)

        values = data.interpolate(np.column_stack((xv.flatten(), yv.flatten())))

        return Data(xv, yv, np.reshape(values, xv.shape))

    def log(data, **kwargs):
        """The base-10 logarithm of every datapoint."""
        subtract = bool(kwargs.get('Subtract offset'))
        newmin = float(kwargs.get('New min'))

        copy = data.copy()
        min = np.min(copy.values)

        if subtract:
            copy.values = (copy.values - min) + newmin

        copy.values = np.log10(copy.values)

        return copy

    def lowpass(data, **kwargs):
        """Perform a low-pass filter."""
        copy = data.copy()

        sx, sy = float(kwargs.get('X Width')), float(kwargs.get('Y Height'))
        kernel_type = str(kwargs.get('Type')).lower()

        kernel = create_kernel(sx, sy, 7, kernel_type)
        copy.values = ndimage.filters.convolve(copy.values, kernel)

        copy.values = np.ma.masked_invalid(copy.values)

        return copy

    def neg(data, **kwargs):
        """Negate every datapoint."""
        return Data(data.x_coords, data.y_coords, np.negative(data.values), data.equidistant)

    def norm_columns(data, **kwargs):
        """Transform the values of every column so that they use the full colormap."""
        copy = data.copy()
        copy.values = np.apply_along_axis(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)), 0, copy.values)

        return copy

    def norm_rows(data, **kwargs):
        """Transform the values of every row so that they use the full colormap."""
        copy = data.copy()
        copy.values = np.apply_along_axis(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)), 1, copy.values)

        return copy

    def offset(data, **kwargs):
        """Add a value to every datapoint."""
        offset = float(kwargs.get('Offset'))

        return Data(data.x_coords, data.y_coords, data.values + offset, data.equidistant)

    def offset_axes(data, **kwargs):
        """Add an offset value to the axes."""
        x_off, y_off = float(kwargs.get('X Offset')), float(kwargs.get('Y Offset'))

        return Data(data.x_coords + x_off, data.y_coords + y_off, data.values, data.equidistant)

    def power(data, **kwargs):
        """Raise the datapoints to a power."""
        power = float(kwargs.get('Power'))

        return Data(data.x_coords, data.y_coords, np.power(data.values, power), data.equidistant)

    def scale_axes(data, **kwargs):
        """Multiply the axes values by a number."""
        x_sc, y_sc = float(kwargs.get('X Scale')), float(kwargs.get('Y Scale'))

        return Data(data.x_coords * x_sc, data.y_coords * y_sc, data.values, data.equidistant)

    def scale_data(data, **kwargs):
        """Multiply the datapoints by a number."""
        factor = float(kwargs.get('Factor'))

        return Data(data.x_coords, data.y_coords, data.values * factor, data.equidistant)

    def sub_linecut(data, **kwargs):
        """Subtract a horizontal/vertical linecut from every row/column."""
        linecut_type = kwargs.get('linecut_type')
        linecut_coord = kwargs.get('linecut_coord')

        if linecut_type == None:
            return data

        if linecut_type == 'horizontal':
            x, y = data.get_row_at(linecut_coord)
        elif linecut_type == 'vertical':
            x, y = data.get_column_at(linecut_coord)
            y = y[:,np.newaxis]

        return Data(data.x_coords, data.y_coords, data.values - y, data.equidistant)

    def sub_plane(data, **kwargs):
        """Subtract a plane with x and y slopes centered in the middle."""
        xs, ys = float(kwargs.get('X Slope')), float(kwargs.get('Y Slope'))
        xmin, xmax, ymin, ymax = data.get_dimensions()

        copy = data.copy()
        copy.values -= xs*(copy.x_coords - (xmax - xmin)/2) + ys*(copy.y_coords - (ymax - ymin)/2)
        
        return copy

    def xderiv(data, **kwargs):
        """Find the rate of change between every datapoint in the x-direction."""
        method = str(kwargs.get('Method'))
        copy = data.copy()

        if method == 'midpoint':
            dx = np.diff(copy.x_coords, axis=1)
            ddata = np.diff(copy.values, axis=1)

            copy.x_coords = copy.x_coords[:,:-1] + dx / 2.0
            copy.y_coords = copy.y_coords[:,:-1]
            copy.values = ddata / dx
        elif method == '2nd order central diff':
            copy.values = (copy.values[:,2:] - copy.values[:,:-2]) / (copy.x_coords[:,2:] - copy.x_coords[:,:-2])
            copy.x_coords = copy.x_coords[:,1:-1]
            copy.y_coords = copy.y_coords[:,1:-1]

        return copy

    def yderiv(data, **kwargs):
        """Find the rate of change between every datapoint in the y-direction."""
        method = str(kwargs.get('Method'))
        copy = data.copy()

        if method == 'midpoint':
            dy = np.diff(copy.y_coords, axis=0)
            ddata = np.diff(copy.values, axis=0)

            copy.x_coords = copy.x_coords[:-1,:]
            copy.y_coords = copy.y_coords[:-1,:] + dy / 2.0
            copy.values = ddata / dy
        elif method == '2nd order central diff':
            copy.values = (copy.values[2:] - copy.values[:-2]) / (copy.y_coords[2:] - copy.y_coords[:-2])
            copy.x_coords = copy.x_coords[1:-1]
            copy.y_coords = copy.y_coords[1:-1]

        return copy