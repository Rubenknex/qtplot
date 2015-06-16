import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import qhull
from scipy.interpolate import griddata

from util import create_kernel

class DatFile:
    """Class which contains the column based DataFrame of the data."""
    def __init__(self, filename):
        self.filename = filename

        self.columns = []
        self.sizes = []

        with open(filename, 'r') as f:
            for line in f:
                line = line.rstrip('\n\t\r')

                if line.startswith('#\tname'):
                    name = line.split(': ', 1)[1]
                    self.columns.append(name)
                elif line.startswith('#\tsize'):
                    size = int(line.split(': ', 1)[1])
                    self.sizes.append(size)

                if len(line) > 0 and line[0].isdigit():
                    break

        self.df = pd.read_table(filename, engine='c', sep='\t', comment='#', names=self.columns)

    def get_data(self, x, y, z, x_order, y_order):
        """Pivot the column based data into matrices."""
        # If an order column is filled with zeros, fill them with an increasing count based on the other order.
        if (self.df[x_order] == 0).all():
            self.df[x_order] = self.df.groupby(y_order)[x_order].apply(lambda x: pd.Series(range(len(x.values)), x.index))

        if (self.df[y_order] == 0).all():
            self.df[y_order] = self.df.groupby(x_order)[y_order].apply(lambda x: pd.Series(range(len(x.values)), x.index))

        rows, row_ind = np.unique(self.df[y_order].values, return_inverse=True)
        cols, col_ind = np.unique(self.df[x_order].values, return_inverse=True)

        # Initially, fill the array with NaN values before placing all the existing values
        pivot = np.zeros((len(rows), len(cols), 3)) * np.nan
        pivot[row_ind, col_ind] = self.df[[x, y, z]].values

        return Data(pivot[:,:,0], pivot[:,:,1], pivot[:,:,2], (x==x_order,y==y_order))



class Data:
    """
    Class which represents 2d data as two matrices with x and y coordinates 
    and one with values.
    """
    def __init__(self, x_coords, y_coords, values, equidistant=(False, False)):
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.values = values

        self.equidistant = equidistant
        self.tri = None

    def set_data(self, x_coords, y_coords, values):
        self.x_coords, self.y_coords, self.values = x_coords, y_coords, values

    def get_limits(self):
        self.xmin, self.xmax = np.nanmin(self.x_coords), np.nanmax(self.x_coords)
        self.ymin, self.ymax = np.nanmin(self.y_coords), np.nanmax(self.y_coords)
        self.zmin, self.zmax = np.nanmin(self.values), np.nanmax(self.values)

        if self.xmin == self.xmax:
            self.xmin, self.xmax = -0.5, 0.5
        if self.ymin == self.ymax:
            self.ymin, self.ymax = -0.5, 0.5

        return self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax

    def gen_delaunay(self):
        xc = self.x_coords.flatten()
        yc = self.y_coords.flatten()
        self.no_nan_values = self.values.flatten()

        if np.isnan(xc).any() and np.isnan(yc).any():
            xc = xc[~np.isnan(xc)]
            yc = yc[~np.isnan(yc)]
            self.no_nan_values = self.no_nan_values[~np.isnan(self.no_nan_values)]

        # Default: Qbb Qc Qz 
        self.tri = qhull.Delaunay(np.column_stack((xc, yc)), qhull_options='')

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
            self.tri = qhull.Delaunay(np.column_stack((xc, yc)), qhull_options='QbB')

        simplices = self.tri.find_simplex(points)

        indices = np.take(self.tri.simplices, simplices, axis=0)
        transforms = np.take(self.tri.transform, simplices, axis=0)

        delta = points - transforms[:,2]
        bary = np.einsum('njk,nk->nj', transforms[:,:2,:], delta)

        temp = np.hstack((bary, 1-bary.sum(axis=1, keepdims=True)))

        values = np.einsum('nj,nj->n', np.take(self.no_nan_values, indices), temp)

        #print values[np.any(temp<0, axis=1)]

        # This should put a NaN for points outside of any simplices
        # but is for some reason sometimes also true inside a simplex
        #values[np.any(temp < 0.0, axis=1)] = np.nan

        return values

    def get_sorted_by_coordinates(self):
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
        # -2 rows: both coords need non-nan values
        if xc.shape[1] > 1:
            # Pad both sides with a column of interpolated coordinates

            l0, l1, l2 = xc[:,[0]], xc[:,[1]], xc[:,[2]]
            nans = np.isnan(l0)
            l0[nans] = 2*l1[nans] - l2[nans]

            r2, r1, r0 = xc[:,[-3]], xc[:,[-2]], xc[:,[-1]]
            nans = np.isnan(r0)
            r0[nans] = 2*r1[nans] - r2[nans]

            xc = np.hstack((2*l0 - l1, xc, 2*r0 - r1))
            # Create center points by adding the differences divided by 2 to the original coordinates
            x = xc[:,:-1] + np.diff(xc, axis=1) / 2.0
            # Add a row to the bottom so that the x coords have the same dimension as the y coords
            x = np.vstack((x, x[-1]))
        else:
            # If data is 1d, make one axis range from -.5 to .5
            x = np.hstack((xc - 0.5, xc[:,[0]] + 0.5))
            # Duplicate the only row/column so that pcolor has something to actually plot
            x = np.vstack((x, x[0]))
        
        if yc.shape[0] > 1:
            t0, t1, t2 = yc[0], yc[1], yc[2]
            nans = np.isnan(t0)
            t0[nans] = 2*t1[nans] - t2[nans]

            b2, b1, b0 = yc[-3], yc[-2], yc[-1]
            nans = np.isnan(b0)
            b0[nans] = 2*b1[nans] - b2[nans]

            yc = np.vstack([2*t0 - t1, yc, 2*b0 - b1])
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
        return np.nanmin(self.x_coords), np.nanmax(self.x_coords), np.nanmin(self.y_coords), np.nanmax(self.y_coords)

    def flip_axes(self, x_flip, y_flip):
        if x_flip:
            self.set_data(np.fliplr(self.x_coords), np.fliplr(self.y_coords), np.fliplr(self.values))

        if y_flip:
            self.set_data(np.flipud(self.x_coords), np.fliplr(self.y_coords), np.fliplr(self.values))

    def is_flipped(self):
        x_flip = self.x_coords[0,0] > self.x_coords[0,-1]
        y_flip = self.y_coords[0,0] > self.y_coords[-1,0]

        return x_flip, y_flip

    def copy(self):
        return Data(np.copy(self.x_coords), np.copy(self.y_coords), np.copy(self.values), self.equidistant)

    def abs(data, **kwargs):
        """Take the absolute value of every datapoint."""
        data.values = np.absolute(data.values)

    def autoflip(data, **kwargs):
        """Flip the data so that the X and Y-axes increase to the top and right."""
        data.flip_axes(*data.is_flipped())

    def crop(data, **kwargs):
        """Crop a region of the data by the columns and rows."""
        x1, x2 = int(kwargs.get('Left')), int(kwargs.get('Right'))
        y1, y2 = int(kwargs.get('Bottom')), int(kwargs.get('Top'))

        if x2 < 0: x2 = data.values.shape[1] + x2 + 1
        if y2 < 0: y2 = data.values.shape[0] + y2 + 1

        self.set_data(data.x_coords[y1:y2,x1:x2], data.y_coords[y1:y2,x1:x2], data.values[y1:y2,x1:x2])

    def dderiv(data, **kwargs):
        """Calculate the component of the gradient in a specific direction."""
        theta = np.radians(float(kwargs.get('Theta')))
        xdir, ydir = np.cos(theta), np.sin(theta)
        method = str(kwargs.get('Method'))

        if method == 'midpoint':
            xcomp = Data.xderiv(data, Method=method)
            ycomp = Data.yderiv(data, Method=method)

            xvalues = xcomp.values[:-1,:]
            yvalues = ycomp.values[:,:-1]

            return Data(xcomp.x_coords[:-1,:], ycomp.y_coords[:,:-1], xvalues * xdir + yvalues * ydir, data.equidistant)
        elif method == '2nd order central diff':
            xcomp = Data.xderiv(data, Method=method)
            ycomp = Data.yderiv(data, Method=method)

            xvalues = xcomp.values[1:-1,:]
            yvalues = ycomp.values[:,1:-1]

            return Data(xcomp.x_coords[1:-1,:], ycomp.y_coords[:,1:-1], xvalues * xdir + yvalues * ydir, data.equidistant)

    def equalize(data, **kwargs):
        """Perform histogramic equalization on the image."""
        binn = 65535

        # Create a density histogram with surface area 1
        hist, bins = np.histogram(data.values.flatten(), binn)
        cdf = hist.cumsum()

        cdf = bins[0] + (bins[-1]-bins[0]) * (cdf / float(cdf[-1]))

        new = np.interp(data.values.flatten(), bins[:-1], cdf)
        data.values = np.reshape(new, data.values.shape)

    def even_odd(data, **kwargs):
        """Extract even or odd rows, optionally flipping odd rows."""
        even = bool(kwargs.get('Even'))

        indices = np.arange(0, data.values.shape[0], 2)
        if not even: indices = np.arange(1, data.values.shape[0], 2)

        data.values = data.values[indices]
        data.x_coords = data.x_coords[indices]
        data.y_coords = data.y_coords[indices]

    def flip(data, **kwargs):
        """Flip the X or Y axes."""
        if bool(kwargs.get('X Axis')):
            data.flip_axes(True, False)

        if bool(kwargs.get('Y Axis')):
            data.flip_axes(False, True)

    def gradmag(data, **kwargs):
        """Calculate the length of every gradient vector."""
        method = str(kwargs.get('Method'))

        if method == 'midpoint':
            xcomp = Data.xderiv(data, Method=method)
            ycomp = Data.yderiv(data, Method=method)

            xvalues = xcomp.values[:-1,:]
            yvalues = ycomp.values[:,:-1]

            return Data(xcomp.x_coords[:-1,:], ycomp.y_coords[:,:-1], np.sqrt(xvalues**2 + yvalues**2), data.equidistant)
        elif method == '2nd order central diff':
            xcomp = Data.xderiv(data, Method=method)
            ycomp = Data.yderiv(data, Method=method)

            xvalues = xcomp.values[1:-1,:]
            yvalues = ycomp.values[:,1:-1]

            return Data(xcomp.x_coords[1:-1,:], ycomp.y_coords[:,1:-1], np.sqrt(xvalues**2 + yvalues**2), data.equidistant)

    def highpass(data, **kwargs):
        """Perform a high-pass filter."""
        # X and Y sigma order?
        sx, sy = float(kwargs.get('X Width')), float(kwargs.get('Y Height'))
        kernel_type = str(kwargs.get('Type')).lower()

        kernel = create_kernel(sx, sy, 7, kernel_type)
        data.values = data.values - ndimage.filters.convolve(data.values, kernel)

        #copy.values = np.ma.masked_invalid(copy.values)

    def hist2d(data, **kwargs):
        """Convert every column into a histogram, default bin amount is sqrt(n)."""
        hmin, hmax = float(kwargs.get('Min')), float(kwargs.get('Max'))
        hbins = int(kwargs.get('Bins'))

        hist = np.apply_along_axis(lambda x: np.histogram(x, bins=hbins, range=(hmin, hmax))[0], 0, data.values)

        binedges = np.linspace(hmin, hmax, hbins + 1)
        bincoords = (binedges[:-1] + binedges[1:]) / 2

        data.x_coords = np.tile(data.x_coords[0,:], (hist.shape[0], 1))
        data.y_coords = np.tile(bincoords[:,np.newaxis], (1, hist.shape[1]))

    def interp_grid(data, **kwargs):
        """Interpolate the data onto a uniformly spaced grid using barycentric interpolation."""
        width, height = int(kwargs.get('Width')), int(kwargs.get('Height'))
        xmin, xmax, ymin, ymax = data.get_dimensions()

        x = np.linspace(xmin, xmax, width)
        y = np.linspace(ymin, ymax, height)
        xv, yv = np.meshgrid(x, y)

        data.x_coords = xv
        data.y_coords = yv
        data.values = np.reshape(data.interpolate(np.column_stack((xv.flatten(), yv.flatten()))), xv.shape)

    def interp_x(data, **kwargs):
        """Interpolate every row onto a uniformly spaced grid."""
        points = int(kwargs.get('Points'))
        xmin, xmax, ymin, ymax = data.get_dimensions()

        x = np.linspace(xmin, xmax, points)

        rows = data.values.shape[0]
        values = np.zeros((rows, points))
        for i in range(rows):
            values[i] = np.interp(x, data.x_coords[i], data.values[i], left=np.nan, right=np.nan)

        y_avg = np.average(data.y_coords, axis=1)[np.newaxis].T

        return Data(np.tile(x, (rows,1)), np.tile(y_avg, (1, points)), values)

    def interp_y(data, **kwargs):
        """Interpolate every column onto a uniformly spaced grid."""
        points = int(kwargs.get('Points'))
        xmin, xmax, ymin, ymax = data.get_dimensions()

        y = np.linspace(ymin, ymax, points)[np.newaxis].T

        cols = data.values.shape[1]
        values = np.zeros((points, cols))
        for i in range(cols):
            values[:,i] = np.interp(y.ravel(), data.y_coords[:,i].ravel(), data.values[:,i].ravel(), left=np.nan, right=np.nan)

        x_avg = np.average(data.x_coords, axis=0)

        return Data(np.tile(x_avg, (points,1)), np.tile(y, (1,cols)), values)

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
        copy.values = np.apply_along_axis(lambda x: (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x)), 0, copy.values)

        return copy

    def norm_rows(data, **kwargs):
        """Transform the values of every row so that they use the full colormap."""
        copy = data.copy()
        copy.values = np.apply_along_axis(lambda x: (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x)), 1, copy.values)

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
        linecut_type = kwargs.get('Horizontal')
        linecut_coord = float(kwargs.get('Row/Column'))

        if linecut_type == None:
            return data

        if linecut_type:
            x, y = data.get_row_at(linecut_coord)
        else:
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