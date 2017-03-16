import json
import os

from collections import OrderedDict

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
    def __init__(self, name, enabled=True, **parameters):
        self.name = name
        self.enabled = enabled
        self.parameters = parameters

    def apply(self, data2d):
        if self.enabled:
            func = getattr(Data2D, self.name)
            func(data2d, **self.parameters)


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
        self.dir = os.path.dirname(os.path.realpath(__file__))

        self.profile = None

        self.filename = None
        self.data_file = None

        self.x, self.y, self.z = None, None, None
        self.data2d = None

        self.colormap = Colormap('transform/Seismic.npy')

        self.operations = []
        self.linetraces = []

        # Define signals that can be listened to
        self.profile_changed = Signal()
        self.data_file_changed = Signal()
        self.data2d_changed = Signal()
        self.cmap_changed = Signal()
        self.operations_changed = Signal()
        self.linetrace_changed = Signal()

    def init_settings(self):
        # /home/name on Linux, C:/Users/Name on Windows
        self.home_dir = os.path.expanduser('~')

        # Create the required directories paths
        self.settings_dir = os.path.join(self.home_dir, '.qtplot')
        self.profiles_dir = os.path.join(self.settings_dir, 'profiles')
        self.operations_dir = os.path.join(self.settings_dir, 'operations')

        # Create the directories if they don't exist yet
        for dir in [self.settings_dir, self.profiles_dir, self.operations_dir]:
            if not os.path.exists(dir):
                os.makedirs(dir)

        self.settings_file = os.path.join(self.settings_dir, 'qtplot.json')

        # Save the default settings if no file exists yet
        if not os.path.exists(self.settings_file):
            self.save_json({'default_profile': 'default.json'},
                           self.settings_file)

        # Load the settings
        self.settings = self.load_json(self.settings_file)

        profile_file = os.path.join(self.profiles_dir,
                                    self.settings['default_profile'])

        # Use the default profile if the specified profile is not found
        if not os.path.isfile(profile_file):
            profile_file = os.path.join(self.dir, 'default_profile.json')

        # Load the profile
        self.profile = self.load_json(profile_file)

        self.profile_changed.fire(self.profile)

    def load_json(self, filename):
        with open(filename, 'r') as f:
            profile = json.load(f, object_pairs_hook=OrderedDict)

        return profile

    def save_json(self, profile, filename):
        with open(filename, 'w') as f:
            json.dump(profile, f, indent=4)

    def load_data_file(self, filename):
        different_file = self.filename != filename

        self.filename = filename
        self.data_file = DatFile(filename)

        self.data_file_changed.fire(different_file)

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

        for info in data['operations']:
            self.add_operation(**info)

    def save_operations(self, filename):
        data = []

        for i, operation in enumerate(self.operations):
            data.append(operation.__dict__)

        with open(filename, 'w') as f:
            json.dump({'operations': data}, f, indent=4)

    def apply_operations(self):
        if self.data2d is None:
            raise DataException('No parameters have been selected yet')

        # Get a fresh copy of the data
        self.data2d = self.data_file.get_data(self.x, self.y, self.z)

        for operation in self.operations:
            operation.apply(self.data2d)

        self.data2d_changed.fire()

    def set_operation_parameters(self, index, parameters):
        self.operations[index].parameters = parameters

        self.operations_changed.fire('values')

    def add_operation(self, name, enabled=True, **parameters):
        operation = Operation(name, **parameters)
        self.operations.append(operation)

        self.operations_changed.fire('add', operation)

    def set_operation_enabled(self, index, enabled):
        self.operations[index].enabled = enabled

        self.operations_changed.fire()

    def swap_operations(self, index):
        tmp = self.operations[index]
        self.operations[index] = self.operations[index + 1]
        self.operations[index + 1] = tmp

        self.operations_changed.fire('swap', index)

    def remove_operation(self, index):
        del self.operations[index]

        self.operations_changed.fire('remove', index)

    def clear_operations(self):
        self.operations = []

        self.operations_changed.fire('clear')

    def add_linetrace(self, x, y, type):
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
            # calculate points
            pass

        line = Linetrace(x, y, type)
        self.linetraces.append(line)

        self.linetrace_changed.fire('add', line)
