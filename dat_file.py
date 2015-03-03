import numpy as np
import pandas as pd
from scipy import ndimage

class DatFile:
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
        x_coords = self.df.pivot(y_order, x_order, x).values
        y_coords = self.df.pivot(y_order, x_order, y).values
        values   = self.df.pivot(y_order, x_order, z).values

        return Data(x_coords, y_coords, values)

class Data:
    def __init__(self, x_coords, y_coords, values):
        self.x_coords = np.ma.masked_invalid(x_coords)
        self.y_coords = np.ma.masked_invalid(y_coords)
        self.values = np.ma.masked_invalid(values)

    def get_sorted(self):
        x_indices = np.argsort(self.x_coords[0,:])
        y_indices = np.argsort(self.y_coords[:,0])

        return self.x_coords[:,x_indices], self.y_coords[y_indices,:], self.values[:,x_indices][y_indices,:]

    def get_quadrilaterals(self, xc, yc):
        """
        In order to generate quads for every datapoint we do the following for the x and y coordinates:
        - Pad the coordinates with a column/row on each side
        - Add the difference between all the coords divided by 2 to the coords, this generates midpoints
        - Add a row/column at the end to satisfy the 1 larger requirements of pcolor
        """
        if len(xc[0,:]) > 1:
            xc = np.hstack((xc[:,[0]] - (xc[:,[1]] - xc[:,[0]]), xc, xc[:,[-1]] + (xc[:,[-1]] - xc[:,[-2]])))
            x = xc[:,:-1] + np.diff(xc, axis=1) / 2.0
            x = np.vstack((x, x[-1]))
        else:
            x = np.hstack((xc - 0.5, xc[:,[0]] + 0.5))
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
        xc, yc, c = self.get_sorted()

        x, y = self.get_quadrilaterals(xc, yc)

        return np.ma.masked_where(np.isnan(x), x), np.ma.masked_where(np.isnan(y), y), np.ma.masked_where(np.isnan(c), c)

    def get_column_at(self, x):
        x_index = np.where(self.x_coords[0,:]==self.get_closest_x(x))[0][0]

        return self.y_coords[:,x_index], self.values[:,x_index]

    def get_row_at(self, y):
        y_index = np.where(self.y_coords[:,0]==self.get_closest_y(y))[0][0]

        return self.x_coords[y_index], self.values[y_index]

    def get_closest_x(self, x_coord):
        return min(self.x_coords[0,:], key=lambda x:abs(x - x_coord))

    def get_closest_y(self, y_coord):
        return min(self.y_coords[:,0], key=lambda y:abs(y - y_coord))

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
        return Data(np.copy(self.x_coords), np.copy(self.y_coords), np.copy(self.values))

    def abs(data, **kwargs):
        """Take the absolute value of every datapoint."""
        return Data(data.x_coords, data.y_coords, np.absolute(data.values))

    def autoflip(data, **kwargs):
        """Flip the data so that the X and Y-axes increase to the top and right."""
        copy = data.copy()

        copy.flip_axes(*copy.is_flipped())
        
        return copy

    def crop(data, **kwargs):
        """Crop a region of the data by the columns and rows."""
        x1, x2 = int(kwargs.get('Left')), int(kwargs.get('Right'))
        y1, y2 = int(kwargs.get('Bottom')), int(kwargs.get('Top'))

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

        return Data(xcomp.x_coords[:-1,:], ycomp.y_coords[:,:-1], xvalues * xdir + yvalues * ydir)

    def equalize(data, **kwargs):
        return data

    def even_odd(data, **kwargs):
        return data

    def flip(data, **kwargs):
        """Flip the X or Y-axes."""
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

        return Data(xcomp.x_coords[:-1,:], ycomp.y_coords[:,:-1], np.sqrt(xvalues**2 + yvalues**2))

    def highpass(data, **kwargs):
        """Perform a high-pass filter."""
        sx, sy = op.get_property('X Width', float), op.get_property('Y Height', float)
        values = ndimage.filters.gaussian_filter(data.values, [sy, sx])

        return pd.DataFrame(data.values - values, data.index, data.columns)

    def hist2d(data, **kwargs):
        return data

    def log(data, **kwargs):
        """The base-10 logarithm of every datapoint."""
        return Data(data.x_coords, data.y_coords, np.log10(data.values))

    def lowpass(data, **kwargs):
        """Perform a low-pass filter."""
        copy = data.copy()

        # X and Y sigma order?
        sx, sy = float(kwargs.get('X Width')), float(kwargs.get('Y Height'))

        copy.values = ndimage.filters.gaussian_filter(copy.values, [sy, sx])

        return copy

    def neg(data, **kwargs):
        """Negate every datapoint."""
        return Data(data.x_coords, data.y_coords, np.negative(data.values))

    def normalize(data, **kwargs):
        return data

    def offset(data, **kwargs):
        """Add a value to every datapoint."""
        offset = float(kwargs.get('Offset'))

        return Data(data.x_coords, data.y_coords, data.values + offset)

    def offset_axes(data, **kwargs):
        """Add a value to the axes."""
        x_off, y_off = float(kwargs.get('X Offset')), float(kwargs.get('Y Offset'))

        return Data(data.x_coords + x_off, data.y_coords + y_off, data.values)

    def power(data, **kwargs):
        """Raise the datapoints to a power."""
        power = float(kwargs.get('Power'))

        return Data(data.x_coords, data.y_coords, np.power(data.values, power))

    def scale_axes(data, **kwargs):
        """Multiply the axes values by a number."""
        x_sc, y_sc = float(kwargs.get('X Scale')), float(kwargs.get('Y Scale'))

        return Data(data.x_coords * x_sc, data.y_coords * y_sc, data.values)

    def scale_data(data, **kwargs):
        """Multiply the datapoints by a number."""
        factor = float(kwargs.get('Factor'))

        return Data(data.x_coords, data.y_coords, data.values * factor)

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

        return Data(data.x_coords, data.y_coords, data.values - y)

    def xderiv(data, **kwargs):
        """Find the rate of change between every datapoint in the x-direction."""
        copy = data.copy()

        dx = np.diff(copy.x_coords, axis=1)
        ddata = np.diff(copy.values, axis=1)

        copy.x_coords = copy.x_coords[:,:-1] + dx / 2.0
        copy.y_coords = copy.y_coords[:,:-1]
        copy.values = ddata / dx

        return copy

    def yderiv(data, **kwargs):
        """Find the rate of change between every datapoint in the y-direction."""
        copy = data.copy()

        dy = np.diff(copy.y_coords, axis=0)
        ddata = np.diff(copy.values, axis=0)

        copy.x_coords = copy.x_coords[:-1,:]
        copy.y_coords = copy.y_coords[:-1,:] + dy / 2.0
        copy.values = ddata / dy

        return copy