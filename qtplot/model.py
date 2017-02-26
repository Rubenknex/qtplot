import json

from .colormap import Colormap
from .data import DatFile, Data2D


class DataException(Exception):
    pass


class Signal:
    def __init__(self):
        self._handlers = []

    def connect(self, handler):
        self._handlers.append(handler)

    def fire(self, *args):
        for handler in self._handlers:
            handler(*args)


class Operation:
    def __init__(self, name, enabled=True, parameters={}):
        self.name = name
        self.enabled = enabled
        self.parameters = parameters

    def apply(self, data2d):
        if self.enabled:
            func = getattr(Data2D, self.name)
            func(data2d, **self.kwargs)


class Model:
    """
    This class should be able to do all data manipulation as in the
    final application, but contain NO GUI code.

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

    def select_parameters(self, x, y, z):
        if self.data_file is None:
            raise DataException('No data file has been loaded yet')

        if None in [x, z]:
            raise ValueError('The x/z parameters cannot be None')

        self.x, self.y, self.z = x, y, z
        self.data2d = self.data_file.get_data(x, y, z)

        self.data2d_changed.fire()

    def set_colormap(self, name):
        settings = self.colormap.get_settings()
        self.colormap = Colormap(name)
        self.colormap.set_settings(**settings)

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
        if self.z is None:
            raise DataException('No parameters have been selected yet')

        self.select_parameters(self.x, self.y, self.z)

        for operation in self.operations:
            operation.apply(self.data2d)
