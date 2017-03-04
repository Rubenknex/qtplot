import json

from .colormap import Colormap
from .data import DatFile, Data2D


class DataException(Exception):
    """ Exception for errors that relate to the data itself """
    pass


class Signal:
    """ A signal that can be fired and is then handled by subscribers """
    def __init__(self):
        self._handlers = []

    def connect(self, handler):
        self._handlers.append(handler)

    def fire(self, *args):
        for handler in self._handlers:
            handler(*args)


class Operation:
    """ A data operation that can be performed on the Data2D class """
    def __init__(self, name, enabled=True, parameters={}):
        self.name = name
        self.enabled = enabled
        self.parameters = parameters

    def apply(self, data2d):
        if self.enabled:
            func = getattr(Data2D, self.name)
            func(data2d, **self.kwargs)


class Linetrace:
    """ This class represents a linetrace in 2D data """
    def __init__(self, x, y, type):
        self.x, self.y = x, y
        self.type = type

    def get_matplotlib(self):
        return self.x, self.y
        return {'x': self.x, 'y': self.y}


class Model:
    """
    This class should be able to do all data manipulation as in the
    final application, but contain no GUI code.

    DatFile:
        Separate into QTLabFile and QcodesFile? These would implement some
        general methods like get_data(x, y, z)
    """
    def __init__(self):
        self.filename = None
        self.data_file = None

        self.x, self.y, self.z = None, None, None
        self.data2d = None

        self.colormap = Colormap('transform/Seismic.npy')

        self.operations = []

        self.linetrace = None

        # Define signals that can be listened to
        self.data_file_changed = Signal()
        self.data2d_changed = Signal()
        self.cmap_changed = Signal()
        self.linetrace_changed = Signal()

    def load_data_file(self, filename):
        self.filename = filename
        self.data_file = DatFile(filename)

        self.data_file_changed.fire()

    def refresh(self):
        if self.filename is None:
            raise DataException('No data file has been loaded yet')

        self.load_data_file(self.filename)

        self.select_parameters(self.x, self.y, self.z)

    def swap_axes(self):
        self.select_parameters(self.y, self.x, self.z)

    def select_parameters(self, x, y, z):
        if self.data_file is None:
            raise DataException('No data file has been loaded yet')

        if None in [x, z]:
            raise ValueError('The x/z parameters cannot be None')

        # If something changed
        if self.x != x or self.y != y or self.z != z:
            self.x, self.y, self.z = x, y, z
            self.data2d = self.data_file.get_data(x, y, z)

            self.data2d_changed.fire()

    def set_colormap(self, name):
        settings = self.colormap.get_settings()
        self.colormap = Colormap(name)
        self.colormap.set_settings(*settings)

        self.cmap_changed.fire()

    def set_colormap_settings(self, min, max, gamma):
        self.colormap.set_settings(min, max, gamma)

        self.cmap_changed.fire()

    def load_operations(self, filename):
        with open(filename) as f:
            data = json.load(f)

        self.operations = []

        for i, info in sorted(data.items()):
            operation = Operation(**info)

            self.operations.append(operation)

    def save_operations(self, filename):
        data = {}

        for i, operation in enumerate(self.operations):
            data[i] = operation.__dict__

        with open(filename, 'w') as f:
            f.write(json.dumps(data, indent=4))

    def apply_operations(self):
        if self.data2d is None:
            raise DataException('No parameters have been selected yet')

        self.select_parameters(self.x, self.y, self.z)

        for operation in self.operations:
            operation.apply(self.data2d)

    def take_linetrace(self, x, y, type):
        if self.data2d is None:
            raise DataException('No parameters have been selected yet')

        row, column = self.data2d.get_closest_point(x, y)

        if type == 'horizontal':
            x = self.data2d.x[row]
            y = self.data2d.z[row]
        elif type == 'vertical':
            x = self.data2d.y[:,column]
            y = self.data2d.z[:,column]
        elif type == 'arbitrary':
            pass

        self.linetrace = Linetrace(x, y, type)

        self.linetrace_changed.fire()