import numpy as np
import pandas as pd
from scipy import ndimage, interpolate
from scipy.spatial import qhull

class DatFile:
    """Class which contains the column based DataFrame of the data."""
    def __init__(self, filename):
        self.filename = filename

        self.columns = []
        self.sizes = {}

        with open(filename, 'r') as f:
            first_line = f.readline()

            # Check whether the .dat file is produced by qtlab or by qcodes.
            if first_line.startswith('# Filename: '):
                for line in f:
                    line = line.rstrip('\n\t\r')

                    if line.startswith('#\tname'):
                        name = line.split(': ', 1)[1]
                        self.columns.append(name)
                    elif line.startswith('#\tsize'):
                        size = int(line.split(': ', 1)[1])
                        self.sizes[self.columns[-1]] = size

                    # When a line starts with a number we have reached the actual data
                    if len(line) > 0 and line[0].isdigit():
                        break
            else:
                column_names = f.readline().strip()[2:]
                self.columns = [s[1:-1] for s in column_names.split('\t')]

                column_sizes = f.readline().strip()[2:]
                parsed_sizes = [int(s) for s in column_sizes.split('\t')]

                self.sizes = dict(zip(self.columns, parsed_sizes))

        self.df = pd.read_table(filename, engine='c', sep='\t', comment='#', names=self.columns)

    def read_gnuplot_dat(self):
        pass

    def read_qtlab_dat(self):
        pass

    def has_columns(self, columns):
        existance = [col in self.df.columns for col in columns]

        if False in existance:
            return columns[existance.index(False)]

        return None

    def get_data(self, x, y, z, x_order, y_order):
        """ 
        Pivot the column based data into matrices according to the dependent
        variables x_order and y_order.
        """

        def generate_series(column):
            # Return an integer series from 0 to the column length.
            return pd.Series(range(len(column.values)), column.index)

        def create_func(minimum):
            def func(column):
                # Return an integer series from min_idx to max_idx, according
                # to the minimum and maximum values, and step size.
                step_size =  np.average(np.diff(column.values))
                min_idx = np.floor((np.nanmin(column.values) - minimum) / step_size)
                max_idx = np.floor((np.nanmax(column.values) - minimum) / step_size)
                
                return pd.Series(np.arange(min_idx, max_idx + 1), column.index)

            return func

        minx, miny = np.nanmin(self.df[x].values), np.nanmin(self.df[y].values)
        varying_x, varying_y = False, False

        # If all the values of one of the dependent variables are zero, 
        # it is either a single sweep or repeated sweeps where this dependent
        # variable is either not adjusted or non-existent.
        if (self.df[x_order] == 0).all():
            # In this case, replace the blocks of the dependent variable that
            # is zero with a series of increasing integers. This is so the data
            # can be plotted underneath each other or side-by-side in the
            # proper order for visualization.
            x_order = 'new_x_order'
            self.df[x_order] = self.df.groupby(y_order)[x].apply(generate_series)
        # If there are more unique dependent parameter values in the full
        # data than the length of a single sweep, then the ranges must be
        # varying between the sweeps.
        elif len(np.unique(self.df[x_order].values)) > self.sizes[x_order]:
            # In order to pivot the data correctly,
            # a new dependent variable column is created which contains integer
            # ranges corresponding to the actual swept range.
            # Example:
            #
            # 123
            #  2345
            #    45678
            #
            x_order = 'new_x_order'
            self.df[x_order] = self.df.groupby(y_order)[x].apply(create_func(minx))
            varying_x = True

        # Do the same as above but for the other independent variable.
        if (self.df[y_order] == 0).all():
            y_order = 'new_y_order'
            self.df[y_order] = self.df.groupby(x_order)[y].apply(generate_series)
        elif len(np.unique(self.df[y_order].values)) > self.sizes[y_order]:
            y_order = 'new_y_order'
            self.df[y_order] = self.df.groupby(x_order)[y].apply(create_func(miny))
            varying_y = True

        # Find the indices of the unique values in the dependent variables.
        # With this we can index and pivot the data.
        rows, row_ind = np.unique(self.df[y_order].values, return_inverse=True)
        cols, col_ind = np.unique(self.df[x_order].values, return_inverse=True)

        # Initially, fill the array with NaN values before placing all the existing values
        pivot = np.zeros((len(rows), len(cols), 3)) * np.nan
        pivot[row_ind, col_ind] = self.df[[x, y, z]].values

        return Data2D(pivot[:,:,0], pivot[:,:,1], pivot[:,:,2], (x==x_order,y==y_order), (varying_x,varying_y))

def create_kernel(x_dev, y_dev, cutoff, distr):
    distributions = {
        'gaussian': lambda r: np.exp(-(r**2) / 2.0),
        'exponential': lambda r: np.exp(-abs(r) * np.sqrt(2.0)),
        'lorentzian': lambda r: 1.0 / (r**2+1.0),
        'thermal': lambda r: np.exp(r) / (1 * (1+np.exp(r))**2)
    }
    func = distributions[distr]

    hx = np.floor((x_dev * cutoff) / 2.0)
    hy = np.floor((y_dev * cutoff) / 2.0)

    x = np.linspace(-hx, hx, hx * 2 + 1) / x_dev
    y = np.linspace(-hy, hy, hy * 2 + 1) / y_dev

    if x.size == 1: x = np.zeros(1)
    if y.size == 1: y = np.zeros(1)
    
    xv, yv = np.meshgrid(x, y)

    kernel = func(np.sqrt(xv**2+yv**2))
    kernel /= np.sum(kernel)

    return kernel


class Data2D:
    """
    Class which represents 2d data as two matrices with x and y coordinates 
    and one with values.
    """
    def __init__(self, x, y, z, equidistant=(False, False), varying=(False, False)):
        self.x, self.y, self.z = x, y, z

        self.equidistant = equidistant
        self.varying = varying
        self.tri = None

        if self.varying[0] == True or self.varying[1] == True:
            minx, maxx = np.nanmin(x), np.nanmax(x)
            diffx = np.nanmean(np.diff(x, axis=1))
            xrow = minx + np.arange(x.shape[1]) * diffx
            self.x = np.tile(xrow, (x.shape[0], 1))

            miny, maxy = np.nanmin(y), np.nanmax(y)
            diffy = np.nanmean(np.diff(y, axis=0))
            yrow = miny + np.arange(y.shape[0]) * diffy
            self.y = np.tile(yrow[:,np.newaxis], (1, y.shape[1]))

    def set_data(self, x, y, z):
        self.x, self.y, self.z = x, y, z

    def get_limits(self):
        xmin, xmax = np.nanmin(self.x), np.nanmax(self.x)
        ymin, ymax = np.nanmin(self.y), np.nanmax(self.y)
        zmin, zmax = np.nanmin(self.z), np.nanmax(self.z)

        # Thickness for 1d scans, should we do this here or in the drawing code?
        if xmin == xmax:
            xmin, xmax = -1, 1

        if ymin == ymax:
            ymin, ymax = -1, 1

        return xmin, xmax, ymin, ymax, zmin, zmax

    def gen_delaunay(self):
        xc = self.x.flatten()
        yc = self.y.flatten()
        self.no_nan_values = self.z.flatten()

        if np.isnan(xc).any() and np.isnan(yc).any():
            xc = xc[~np.isnan(xc)]
            yc = yc[~np.isnan(yc)]
            self.no_nan_values = self.no_nan_values[~np.isnan(self.no_nan_values)]

        # Default: Qbb Qc Qz 
        self.tri = qhull.Delaunay(np.column_stack((xc, yc)), qhull_options='')

    def interpolate(self, points):
        if self.tri == None:
            xc = self.x.flatten()
            yc = self.y.flatten()
            self.no_nan_values = self.z.flatten()

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
        x_indices = np.argsort(self.x[0,:])
        y_indices = np.argsort(self.y[:,0])

        return self.x[:,x_indices], self.y[y_indices,:], self.z[:,x_indices][y_indices,:]

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
            x = np.hstack((xc - 1, xc[:,[0]] + 1))
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
            y = np.vstack([yc - 1, yc[0] + 1])
            y = np.hstack([y, y[:,[0]]])

        return x, y

    def get_pcolor(self):
        """
        Return a version of the coordinates and values that can be plotted by pcolor, this means:
        -   Points are sorted by increasing coordinates
        -   Quadrilaterals are generated for every datapoint
        -   NaN values are masked to ignore them when plotting

        Can be plotted using matplotlib's pcolor/pcolormesh(*data.get_pcolor())
        """
        #xc, yc, z = self.get_sorted_by_coordinates()

        #x, y = self.get_quadrilaterals(xc, yc)
        x, y = self.get_quadrilaterals(self.x, self.y)

        return np.ma.masked_invalid(x), np.ma.masked_invalid(y), np.ma.masked_invalid(self.z)

    def get_column_at(self, x):
        x_index = np.where(self.x[0,:]==self.get_closest_x(x))[0][0]

        if self.equidistant[0]:
            return self.y[:,x_index], self.z[:,x_index], x_index
        else:
            return self.y[:,x_index], self.z[:,x_index], x_index

    def get_row_at(self, y):
        y_index = np.where(self.y[:,0]==self.get_closest_y(y))[0][0]

        if self.equidistant[1]:
            return self.x[y_index], self.z[y_index], y_index
        else:
            return self.x[y_index], self.z[y_index], y_index

    def get_closest_x(self, x_coord):
        return min(self.x[0,:], key=lambda x:abs(x - x_coord))

    def get_closest_y(self, y_coord):
        return min(self.y[:,0], key=lambda y:abs(y - y_coord))

    def flip_axes(self, x_flip, y_flip):
        if x_flip:
            self.set_data(np.fliplr(self.x), np.fliplr(self.y), np.fliplr(self.z))

        if y_flip:
            self.set_data(np.flipud(self.x), np.fliplr(self.y), np.fliplr(self.z))

    def is_flipped(self):
        x_flip = self.x[0,0] > self.x[0,-1]
        y_flip = self.y[0,0] > self.y[-1,0]

        return x_flip, y_flip

    def copy(self):
        return Data2D(np.copy(self.x), np.copy(self.y), np.copy(self.z), self.equidistant, self.varying)

    def abs(self):
        """Take the absolute value of every datapoint."""
        self.z = np.absolute(self.z)

    def autoflip(self):
        """Flip the data so that the X and Y-axes increase to the top and right."""
        self.flip_axes(*self.is_flipped())

    def crop(self, left=0, right=-1, bottom=0, top=-1):
        """Crop a region of the data by the columns and rows."""
        if right < 0: 
            right = self.z.shape[1] + right + 1

        if top < 0: 
            top = self.z.shape[0] + top + 1

        self.set_data(self.x[bottom:top,left:right], self.y[bottom:top,left:right], self.z[bottom:top,left:right])

    def dderiv(self, theta=0.0, method='midpoint'):
        """Calculate the component of the gradient in a specific direction."""
        xdir, ydir = np.cos(theta), np.sin(theta)

        xcomp = self.copy()
        xcomp.xderiv(method=method)
        ycomp = self.copy()
        ycomp.yderiv(method=method)

        if method == 'midpoint':
            xvalues = xcomp.z[:-1,:]
            yvalues = ycomp.z[:,:-1]

            self.set_data(xcomp.x[:-1,:], ycomp.y[:,:-1], xvalues * xdir + yvalues * ydir)
        elif method == '2nd order central diff':
            xvalues = xcomp.z[1:-1,:]
            yvalues = ycomp.z[:,1:-1]

            self.set_data(xcomp.x[1:-1,:], ycomp.y[:,1:-1], xvalues * xdir + yvalues * ydir)

    def equalize(self):
        """Perform histogramic equalization on the image."""
        binn = 65535

        # Create a density histogram with surface area 1
        no_nans = self.z[~np.isnan(self.z)]
        hist, bins = np.histogram(no_nans.flatten(), binn)
        cdf = hist.cumsum()

        cdf = bins[0] + (bins[-1]-bins[0]) * (cdf / float(cdf[-1]))

        new = np.interp(self.z.flatten(), bins[:-1], cdf)
        self.z = np.reshape(new, self.z.shape)

    def even_odd(self, even):
        """Extract even or odd rows, optionally flipping odd rows."""
        indices = np.arange(0, self.z.shape[0], 2)

        if not even: 
            indices = np.arange(1, self.z.shape[0], 2)

        self.set_data(self.x[indices], self.y[indices], self.z[indices])

    def flip(self, x_flip, y_flip):
        """Flip the X or Y axes."""
        self.flip_axes(x_flip, y_flip)

    def gradmag(self, method='midpoint'):
        """Calculate the length of every gradient vector."""
        xcomp = self.copy()
        xcomp.xderiv(method=method)
        ycomp = self.copy()
        ycomp.yderiv(method=method)

        if method == 'midpoint':
            xvalues = xcomp.z[:-1,:]
            yvalues = ycomp.z[:,:-1]

            self.set_data(xcomp.x[:-1,:], ycomp.y[:,:-1], np.sqrt(xvalues**2 + yvalues**2))
        elif method == '2nd order central diff':
            xvalues = xcomp.z[1:-1,:]
            yvalues = ycomp.z[:,1:-1]

            self.set_data(xcomp.x[1:-1,:], ycomp.y[:,1:-1], np.sqrt(xvalues**2 + yvalues**2))

    def highpass(self, x_width=3, y_height=3, method='gaussian'):
        """Perform a high-pass filter."""
        kernel = create_kernel(x_width, y_height, 7, method)
        self.z = self.z - ndimage.filters.convolve(self.z, kernel)

    def hist2d(self, min, max, bins):
        """Convert every column into a histogram, default bin amount is sqrt(n)."""
        hist = np.apply_along_axis(lambda x: np.histogram(x, bins=bins, range=(min, max))[0], 0, self.z)

        binedges = np.linspace(min, max, bins + 1)
        bincoords = (binedges[:-1] + binedges[1:]) / 2

        self.x = np.tile(self.x[0,:], (hist.shape[0], 1))
        self.y = np.tile(bincoords[:,np.newaxis], (1, hist.shape[1]))
        self.z = hist

    def interp_grid(self, width, height):
        """Interpolate the data onto a uniformly spaced grid using barycentric interpolation."""
        # NOT WOKRING FOR SOME REASON
        xmin, xmax, ymin, ymax, _, _ = self.get_limits()

        x = np.linspace(xmin, xmax, width)
        y = np.linspace(ymin, ymax, height)
        xv, yv = np.meshgrid(x, y)

        self.x, self.y = xv, yv
        self.z = np.reshape(self.interpolate(np.column_stack((xv.flatten(), yv.flatten()))), xv.shape)

    def interp_x(self, points):
        """Interpolate every row onto a uniformly spaced grid."""
        xmin, xmax, ymin, ymax, _, _ = self.get_limits()

        x = np.linspace(xmin, xmax, points)

        rows = self.z.shape[0]
        values = np.zeros((rows, points))

        for i in range(rows):
            f = interpolate.interp1d(self.x[i], self.z[i], bounds_error=False, fill_value=np.nan)
            values[i] = f(x)

        y_avg = np.average(self.y, axis=1)[np.newaxis].T

        self.set_data(np.tile(x, (rows,1)), np.tile(y_avg, (1, points)), values)

    def interp_y(self, points):
        """Interpolate every column onto a uniformly spaced grid."""
        xmin, xmax, ymin, ymax, _, _ = self.get_limits()

        y = np.linspace(ymin, ymax, points)[np.newaxis].T

        cols = self.z.shape[1]
        values = np.zeros((points, cols))

        for i in range(cols):
            f = interpolate.interp1d(self.y[:,i].ravel(), self.z[:,i].ravel(), bounds_error=False, fill_value=np.nan)
            values[i] = f(x)

        x_avg = np.average(self.x, axis=0)

        self.set_data(np.tile(x_avg, (points,1)), np.tile(y, (1,cols)), values)

    def log(self, subtract, min):
        """The base-10 logarithm of every datapoint."""
        minimum = np.nanmin(self.z)

        if subtract:
            #self.z[self.z < 0] = newmin
            self.z += (min - minimum)

        self.z = np.log10(self.z)

    def lowpass(self, x_width=3, y_height=3, method='gaussian'):
        """Perform a low-pass filter."""
        kernel = create_kernel(x_width, y_height, 7, method)
        self.z = ndimage.filters.convolve(self.z, kernel)

        self.z = np.ma.masked_invalid(self.z)

    def negate(self):
        """Negate every datapoint."""
        self.z *= -1

    def norm_columns(self):
        """Transform the values of every column so that they use the full colormap."""
        self.z = np.apply_along_axis(lambda x: (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x)), 0, self.z)

    def norm_rows(self):
        """Transform the values of every row so that they use the full colormap."""
        self.z = np.apply_along_axis(lambda x: (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x)), 1, self.z)

    def offset(self, offset=0):
        """Add a value to every datapoint."""
        self.z += offset

    def offset_axes(self, x_offset=0, y_offset=0):
        """Add an offset value to the axes."""
        self.x += x_offset
        self.y += y_offset

    def power(self, power=1):
        """Raise the datapoints to a power."""
        self.z = np.power(self.z, power)

    def scale_axes(self, x_scale=1, y_scale=1):
        """Multiply the axes values by a number."""
        self.x *= x_scale
        self.y *= y_scale

    def scale_data(self, factor):
        """Multiply the datapoints by a number."""
        self.z *= factor

    def sub_linecut(self, type, position):
        """Subtract a horizontal/vertical linecut from every row/column."""
        if type == 'horizontal':
            x, y, index = self.get_row_at(position)
            y = np.tile(self.z[index,:], (self.z.shape[0],1))
        elif type == 'vertical':
            x, y, index = self.get_column_at(position)
            y = np.tile(self.z[:,index][:,np.newaxis], (1, self.z.shape[1]))

        self.z -= y

    def sub_linecut_avg(self, type, position, size):
        """Subtract a horizontal/vertical averaged linecut from every row/column."""
        if size % 2 == 0:
            start, end = -size/2, size/2-1
        else:
            start, end = -(size-1)/2, (size-1)/2

        indices = np.arange(start, end + 1)

        if type == 'horizontal':
            x, y, index = self.get_row_at(position)
            y = np.mean(self.z[index+indices,:], axis=0)
            y = np.tile(y, (self.z.shape[0],1))
        elif type == 'vertical':
            x, y, index = self.get_column_at(position)
            y = np.mean(self.z[:,index+indices][:,np.newaxis], axis=1)
            y = np.tile(y, (1, self.z.shape[1]))

        self.z -= y

    def sub_plane(self, x_slope, y_slope):
        """Subtract a plane with x and y slopes centered in the middle."""
        xmin, xmax, ymin, ymax, _, _ = self.get_limits()

        self.z -= x_slope*(self.x - (xmax - xmin)/2) + y_slope*(self.y - (ymax - ymin)/2)

    def xderiv(self, method='midpoint'):
        """Find the rate of change between every datapoint in the x-direction."""
        if method == 'midpoint':
            dx = np.diff(self.x, axis=1)
            ddata = np.diff(self.z, axis=1)

            self.x = self.x[:,:-1] + dx / 2.0
            self.y = self.y[:,:-1]
            self.z = ddata / dx
        elif method == '2nd order central diff':
            self.z = (self.z[:,2:] - self.z[:,:-2]) / (self.x[:,2:] - self.x[:,:-2])
            self.x = self.x[:,1:-1]
            self.y = self.y[:,1:-1]

    def yderiv(self, method='midpoint'):
        """Find the rate of change between every datapoint in the y-direction."""
        if method == 'midpoint':
            dy = np.diff(self.y, axis=0)
            ddata = np.diff(self.z, axis=0)

            self.x = self.x[:-1,:]
            self.y = self.y[:-1,:] + dy / 2.0
            self.z = ddata / dy
        elif method == '2nd order central diff':
            self.z = (self.z[2:] - self.z[:-2]) / (self.y[2:] - self.y[:-2])
            self.x = self.x[1:-1]
            self.y = self.y[1:-1]