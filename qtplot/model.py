import json
import os

import numpy as np

from collections import OrderedDict

from .colormap import Colormap
from .data import DatFile, Data2D


def load_json(filename):
    """ Load a JSON file into an OrderedDict """
    with open(filename, 'r') as f:
        profile = json.load(f, object_pairs_hook=OrderedDict)

    return profile


def save_json(profile, filename):
    """ Save a dict to a JSON file """
    with open(filename, 'w') as f:
        json.dump(profile, f, indent=4)


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
    def __init__(self, x_pos, y_pos, x, y, z, row_numbers, type):
        """
        Args:
            x_pos / y_pos: Position at which the linetrace was taken in
                data coordinates

            x / y / z: Data belonging to the selected linetrace

            row_numbers: The row in the original data file to which a
                datapoint belongs. Used to select datapoints and display
                information
        """
        self.x_pos, self.y_pos = x_pos, y_pos
        self.x, self.y, self.z = x, y, z
        self.row_numbers = row_numbers
        self.type = type

    def get_positions(self):
        """ Return datapoint positions on the 2D plot """
        return self.x, self.y

    def get_data(self):
        """ Return the data to plot based on the linetrace type """
        if self.type == 'horizontal':
            return self.x, self.z
        elif self.type == 'vertical':
            return self.y, self.z
        elif self.type == 'arbitrary':
            # Calculate the distance along the line
            return np.hypot(self.x - self.x[0], self.y - self.y[0]), self.z
        else:
            raise ValueError('No proper linetrace type')


class Model:
    """
    This class should be able to do all data manipulation as in the
    final application, but contain no GUI code.

    The structure is as follows: GUI logic will call the model to update
    it's state, after which the model will fire certain events that represent
    changes in the state. GUI code will listen for these events and update
    themselves appropriately.

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

        # Load a default colormap
        self.colormap = Colormap('transform/Seismic.npy')

        self.operations = []
        self.linetraces = []
        self.linetrace_start = (0, 0)

        # Define signals that can be listened to
        self.profile_changed = Signal()
        self.data_file_changed = Signal()
        self.data2d_changed = Signal()
        self.cmap_changed = Signal()
        self.operations_changed = Signal()
        self.linetrace_changed = Signal()

    def init_settings(self):
        """
        Initialize the qtplot settings and profiles

        Structure:

        .qtplot
            profiles
                test.json
                another.json
            qtplot.json
        """
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
            save_json({'default_profile': 'default.json'},
                      self.settings_file)

        # Load the settings
        self.settings = load_json(self.settings_file)

        profile_file = os.path.join(self.profiles_dir,
                                    self.settings['default_profile'])

        # Use the default profile if the specified profile is not found
        if not os.path.isfile(profile_file):
            profile_file = os.path.join(self.dir, 'default_profile.json')

        # Load the profile
        self.profile = load_json(profile_file)
        self.default_profile = self.profile.copy()

        self.profile_changed.fire(self.profile)

    def save_settings(self):
        save_json(self.settings, self.settings_file)

    def load_profile(self, filename):
        profile_path = os.path.join(self.profiles_dir, filename)
        operations_path = os.path.join(self.operations_dir, filename)

        self.load_operations(operations_path)

        self.profile = load_json(profile_path)

        self.profile_changed.fire(self.profile)

    def save_profile(self, profile, filename):
        operations_path = os.path.join(self.operations_dir, filename)

        self.save_operations(operations_path)

        new_profile = self.profile.copy()
        new_profile['operations'] = filename

        # Set non-existing values to the default value
        for key, value in self.default_profile.items():
            if key in profile:
                new_profile[key] = profile[key]
            elif key not in new_profile:
                new_profile[key] = self.default_profile[key]

        profile_path = os.path.join(self.profiles_dir, filename)
        save_json(new_profile, profile_path)

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

        self.clear_operations()

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
            return
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

    def take_linetrace(self, x_pos, y_pos, type, initial_press=True):
        """
        Take a linetrace

        Args:
            x_pos, y_pos: Mouse position in data coordinates

            type: Linetrace type (horizontal/vertical/arbitrary)

            initial_press: Whether this is the initial mouse press
                or a movement. Needed for arbitrary linetraces.
        """
        if self.data2d is None:
            raise DataException('No parameters have been selected yet')

        row, column = self.data2d.get_closest_point(x_pos, y_pos)

        # For horizontal and vertical linetraces the logic is simple
        if type in ['horizontal', 'vertical']:
            if type == 'horizontal':
                x = self.data2d.x[row]
                y = self.data2d.y[row]
                z = self.data2d.z[row]
                row_numbers = self.data2d.row_numbers[row]
            elif type == 'vertical':
                x = self.data2d.x[:, column]
                y = self.data2d.y[:, column]
                z = self.data2d.z[:, column]
                row_numbers = self.data2d.row_numbers[:, column]

            line = Linetrace(x_pos, y_pos, x, y, z, row_numbers, type)
            self.linetraces.append(line)

            self.linetrace_changed.fire('add', line)
        elif type == 'arbitrary':
            # For arbitrary linetraces we track the mouse start and end point
            if initial_press:
                # Store the starting location
                self.linetrace_start = (x_pos, y_pos)
                self.linetraces = []

                self.linetrace_changed.fire('clear')
            else:
                # Calculate the interpolated points along the linetrace
                x0, y0 = self.linetrace_start
                x_points = np.linspace(x0, x_pos, 500)
                y_points = np.linspace(y0, y_pos, 500)

                points = np.column_stack((x_points, y_points))

                # Here we get the actual interpolated data
                values = self.data2d.interpolate(points)

                line = Linetrace(x_pos, y_pos,
                                 x_points, y_points, values, type)

                # Add it to the list
                if len(self.linetraces) == 0:
                    self.linetraces.append(line)
                    self.linetrace_changed.fire('add', line)
                else:
                    self.linetraces[0] = line
                    self.linetrace_changed.fire('update', line)

    def update_linetrace(self):
        """ Update the last linetrace that was taken """
        if len(self.linetraces) > 0:
            line = self.linetraces[-1]

            # Only do this for horizontal/vertical linetraces now
            if line.type in ['horizontal', 'vertical']:
                self.take_linetrace(line.x_pos, line.y_pos, line.type)
