from __future__ import print_function

from six.moves import configparser
import numpy as np
import os
import logging
import sys
from collections import OrderedDict

from PyQt4 import QtGui, QtCore

from .colormap import Colormap
from .data import DatFile, Data2D
from .export import ExportWidget
from .linecut import Linecut
from .operations import Operations
from .settings import Settings
from .canvas import Canvas

logger = logging.getLogger(__name__)

profile_defaults = OrderedDict((
    ('operations', ''),
    ('sub_series_V', ''),
    ('sub_series_I', ''),
    ('sub_series_R', ''),
    ('open_directory', ''),
    ('save_directory', ''),
    ('x', '-'),
    ('y', '-'),
    ('z', '-'),
    ('colormap', 'transform\\Seismic.npy'),
    ('title', '<filename>'),
    ('DPI', '80'),
    ('rasterize', False),
    ('x_label', '<x>'),
    ('y_label', '<y>'),
    ('z_label', '<z>'),
    ('x_format', '%%.0f'),
    ('y_format', '%%.0f'),
    ('z_format', '%%.0f'),
    ('x_div', '1e0'),
    ('y_div', '1e0'),
    ('z_div', '1e0'),
    ('font', 'Vera Sans'),
    ('font_size', '12'),
    ('width', '3'),
    ('height', '3'),
    ('cb_orient', 'vertical'),
    ('cb_pos', '0 0 1 1'),
    ('triangulation', False),
    ('tripcolor', False),
    ('linecut', False),
    ('line_style', 'solid'),
    ('line_width', '0.5'),
    ('marker_style', 'None'),
    ('marker_size', '6'),
))


class QTPlot(QtGui.QMainWindow):
    """ The main window of the qtplot application. """

    def __init__(self, filename=None):
        super(QTPlot, self).__init__(None)

        self.first_data_file = True
        self.name = None
        self.closed = False

        # In case of a .dat file
        self.filename = None
        self.abs_filename = None
        self.dat_file = None

        # In case of a qcodes DataSet(Lite)
        self.data_set = None

        # Data2D object derived from either DatFile or DataSet(Lite)
        self.data = None

        # Create the subwindows
        self.linecut = Linecut(self)
        self.operations = Operations(self)
        self.settings = Settings(self)

        self.init_ui()
        self.init_settings()
        self.init_logging()

        self.settings.populate_ui()

        if filename is not None:
            self.load_dat_file(filename)

    def init_settings(self):
        # Get the home directory of the computer user
        self.home_dir = os.path.expanduser('~')

        self.settings_dir = os.path.join(self.home_dir, '.qtplot')
        self.profiles_dir = os.path.join(self.home_dir, '.qtplot',
                                                        'profiles')
        self.operations_dir = os.path.join(self.home_dir, '.qtplot',
                                                          'operations')

        # Create the program directories if they don't exist yet
        for dir in [self.settings_dir, self.profiles_dir, self.operations_dir]:
            if not os.path.exists(dir):
                os.makedirs(dir)

        self.qtplot_ini_file = os.path.join(self.settings_dir, 'qtplot.ini')

        defaults = {'default_profile': 'default.ini'}
        self.qtplot_ini = configparser.SafeConfigParser(defaults)
        self.profile_ini = configparser.SafeConfigParser(profile_defaults)

        # If a qtplot.ini exists: read it to extract the default profile
        # Else: save the default qtplot.ini
        if os.path.exists(self.qtplot_ini_file):
            self.qtplot_ini.read(self.qtplot_ini_file)
        else:
            with open(self.qtplot_ini_file, 'w') as config_file:
                self.qtplot_ini.write(config_file)

        default_profile = self.qtplot_ini.get('DEFAULT', 'default_profile')
        self.profile_ini_file = os.path.join(self.profiles_dir,
                                             default_profile)

        # if the default profile ini doesn't exist, write defaults to a file
        if not os.path.isfile(self.profile_ini_file):
            with open(self.profile_ini_file, 'w') as config_file:
                self.profile_ini.write(config_file)

        self.profile_settings = defaults

    def init_logging(self):
        formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        log_file = os.path.join(self.settings_dir, 'log.txt')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # Write exceptions to the log
        def my_handler(exc_type, exc_value, exc_traceback):
            exc_info = (exc_type, exc_value, exc_traceback)
            logger.error('Uncaught exception', exc_info=exc_info)

        sys.excepthook = my_handler

    def init_ui(self):
        self.setWindowTitle('qtplot')

        self.main_widget = QtGui.QTabWidget(self)

        self.view_widget = QtGui.QWidget()
        self.main_widget.addTab(self.view_widget, 'View')
        self.export_widget = ExportWidget(self)
        self.main_widget.addTab(self.export_widget, 'Export')

        self.canvas = Canvas(self)

        # Top row buttons
        hbox = QtGui.QHBoxLayout()

        self.b_load = QtGui.QPushButton('Load...')
        self.b_load.clicked.connect(self.on_load_dat)
        hbox.addWidget(self.b_load)

        self.b_refresh = QtGui.QPushButton('Refresh')
        self.b_refresh.clicked.connect(self.on_refresh)
        hbox.addWidget(self.b_refresh)

        self.b_swap_axes = QtGui.QPushButton('Swap axes', self)
        self.b_swap_axes.clicked.connect(self.on_swap_axes)
        hbox.addWidget(self.b_swap_axes)

        self.b_linecut = QtGui.QPushButton('Linecut')
        self.b_linecut.clicked.connect(self.linecut.show_window)
        hbox.addWidget(self.b_linecut)

        self.b_operations = QtGui.QPushButton('Operations')
        self.b_operations.clicked.connect(self.operations.show_window)
        hbox.addWidget(self.b_operations)

        # Subtracting series R
        r_hbox = QtGui.QHBoxLayout()

        lbl_sub = QtGui.QLabel('Sub series R:')
        lbl_sub.setMaximumWidth(70)
        r_hbox.addWidget(lbl_sub)

        lbl_v = QtGui.QLabel('V:')
        lbl_v.setMaximumWidth(10)
        r_hbox.addWidget(lbl_v)

        self.cb_v = QtGui.QComboBox(self)
        self.cb_v.setMaxVisibleItems(25)
        r_hbox.addWidget(self.cb_v)

        lbl_i = QtGui.QLabel('I:')
        lbl_i.setMaximumWidth(10)
        r_hbox.addWidget(lbl_i)

        self.cb_i = QtGui.QComboBox(self)
        self.cb_i.setMaxVisibleItems(25)
        r_hbox.addWidget(self.cb_i)

        lbl_r = QtGui.QLabel('R:')
        lbl_r.setMaximumWidth(10)
        r_hbox.addWidget(lbl_r)

        self.le_r = QtGui.QLineEdit(self)
        self.le_r.setMaximumWidth(50)
        self.le_r.returnPressed.connect(self.on_sub_series_r)
        r_hbox.addWidget(self.le_r)

        self.b_ok = QtGui.QPushButton('Ok', self)
        self.b_ok.clicked.connect(self.on_sub_series_r)
        self.b_ok.setMaximumWidth(50)
        r_hbox.addWidget(self.b_ok)

        # Selecting columns and orders
        grid = QtGui.QGridLayout()

        lbl_x = QtGui.QLabel("X:", self)
        lbl_x.setMaximumWidth(10)
        grid.addWidget(lbl_x, 1, 1)

        self.cb_x = QtGui.QComboBox(self)
        self.cb_x.activated.connect(self.on_data_change)
        self.cb_x.setMaxVisibleItems(25)
        grid.addWidget(self.cb_x, 1, 2)

        lbl_y = QtGui.QLabel("Y:", self)
        grid.addWidget(lbl_y, 2, 1)

        self.cb_y = QtGui.QComboBox(self)
        self.cb_y.activated.connect(self.on_data_change)
        self.cb_y.setMaxVisibleItems(25)
        grid.addWidget(self.cb_y, 2, 2)

        lbl_d = QtGui.QLabel("Data:", self)
        grid.addWidget(lbl_d, 3, 1)

        self.cb_z = QtGui.QComboBox(self)
        self.cb_z.activated.connect(self.on_data_change)
        self.cb_z.setMaxVisibleItems(25)
        grid.addWidget(self.cb_z, 3, 2)

        self.combo_boxes = [self.cb_v, self.cb_i,
                            self.cb_x, self.cb_y, self.cb_z]

        groupbox = QtGui.QGroupBox('Data selection')
        groupbox.setLayout(grid)

        # Colormap
        vbox_gamma = QtGui.QVBoxLayout()
        hbox_gamma1 = QtGui.QHBoxLayout()
        hbox_gamma2 = QtGui.QHBoxLayout()
        vbox_gamma.addLayout(hbox_gamma1)
        vbox_gamma.addLayout(hbox_gamma2)

        # Reset colormap button
        self.cb_reset_cmap = QtGui.QCheckBox('Reset on plot')
        self.cb_reset_cmap.setCheckState(QtCore.Qt.Checked)
        hbox_gamma1.addWidget(self.cb_reset_cmap)

        # Colormap combobox
        self.cb_cmaps = QtGui.QComboBox(self)
        self.cb_cmaps.activated.connect(self.on_cmap_change)

        path = os.path.dirname(os.path.realpath(__file__))

        path = os.path.join(path, 'colormaps')

        cmap_files = []
        for dir, _, files in os.walk(path):
            for filename in files:
                reldir = os.path.relpath(dir, path)
                relfile = os.path.join(reldir, filename)

                # Remove .\ for files in the root of the directory
                if relfile[:2] == '.\\':
                    relfile = relfile[2:]

                cmap_files.append(relfile)

        self.cb_cmaps.addItems(cmap_files)
        self.cb_cmaps.setMaxVisibleItems(25)

        hbox_gamma1.addWidget(self.cb_cmaps)

        # Colormap minimum text box
        self.le_min = QtGui.QLineEdit(self)
        self.le_min.setMaximumWidth(100)
        self.le_min.returnPressed.connect(self.on_min_max_entered)
        hbox_gamma2.addWidget(self.le_min)

        # Colormap minimum slider
        self.s_min = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.s_min.setMaximum(100)
        self.s_min.sliderMoved.connect(self.on_min_changed)
        hbox_gamma2.addWidget(self.s_min)

        # Colormap gamma slider
        self.s_gamma = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.s_gamma.setMinimum(-100)
        self.s_gamma.setMaximum(100)
        self.s_gamma.setValue(0)
        self.s_gamma.valueChanged.connect(self.on_gamma_changed)
        hbox_gamma2.addWidget(self.s_gamma)

        # Colormap maximum slider
        self.s_max = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.s_max.setMaximum(100)
        self.s_max.setValue(self.s_max.maximum())
        self.s_max.sliderMoved.connect(self.on_max_changed)
        hbox_gamma2.addWidget(self.s_max)

        # Colormap maximum text box
        self.le_max = QtGui.QLineEdit(self)
        self.le_max.setMaximumWidth(100)
        self.le_max.returnPressed.connect(self.on_min_max_entered)
        hbox_gamma2.addWidget(self.le_max)

        self.b_reset = QtGui.QPushButton('Reset')
        self.b_reset.clicked.connect(self.on_cm_reset)
        hbox_gamma1.addWidget(self.b_reset)

        groupbox_gamma = QtGui.QGroupBox('Colormap')
        groupbox_gamma.setLayout(vbox_gamma)

        # Bottom row buttons
        hbox4 = QtGui.QHBoxLayout()

        self.b_save_matrix = QtGui.QPushButton('Save data...')
        self.b_save_matrix.clicked.connect(self.on_save_matrix)
        hbox4.addWidget(self.b_save_matrix)

        self.b_settings = QtGui.QPushButton('Settings')
        self.b_settings.clicked.connect(self.settings.show_window)
        hbox4.addWidget(self.b_settings)

        # Main vertical box
        vbox = QtGui.QVBoxLayout(self.view_widget)
        vbox.addWidget(self.canvas.native)
        vbox.addLayout(hbox)
        vbox.addLayout(r_hbox)
        vbox.addWidget(groupbox)
        vbox.addWidget(groupbox_gamma)
        vbox.addLayout(hbox4)

        self.status_bar = QtGui.QStatusBar()
        self.l_position = QtGui.QLabel()
        self.status_bar.addWidget(self.l_position, 1)
        self.l_slope = QtGui.QLabel('Slope: -')
        self.status_bar.addWidget(self.l_slope)
        self.setStatusBar(self.status_bar)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.resize(500, 740)
        self.move(100, 100)

        self.setAcceptDrops(True)

        self.linecut.show()
        self.operations.show()
        self.show()

    def update_ui(self, reset=True, opening_state=False):
        """
        Update the user interface, typically called on loading new data (not
        on updating data).

        The x/y/z parameter selections are populated. If this is the first data
        that is loaded, it is checked if parameters set as default in the .ini
        are present, which are then immediately selected.
        """
        if self.name is not None:
            self.setWindowTitle(self.name)

        parameters = [''] + self.get_parameter_names()

        # Repopulate the combo boxes
        for cb in self.combo_boxes:
            i = cb.currentIndex()

            cb.clear()
            cb.addItems(parameters)
            cb.setCurrentIndex(i)

        # Load the series resistance
        if opening_state:
            R = self.profile_settings['sub_series_R']
            self.le_r.setText(R)

        # Set the selected parameters
        if reset and self.first_data_file:
            names = ['sub_series_V', 'sub_series_I', 'x', 'y', 'z']
            default_indices = [0, 0, 1, 1, 2, 2, 4]

            for i, cb in enumerate(self.combo_boxes):
                parameter = self.profile_settings[names[i]]

                index = cb.findText(parameter)

                cb.setCurrentIndex(index)

                if index == -1:
                    cb.setCurrentIndex(default_indices[i])

        # If the dataset is 1D; disable the y-parameter combobox
        if self.dat_file is not None:
            if self.dat_file.ndim == 1:
                self.cb_y.setCurrentIndex(0)
                self.cb_y.setEnabled(False)
            else:
                self.cb_y.setEnabled(True)

        # Set the colormap
        cmap = self.profile_settings['colormap']

        # The path that is saved in the profile can use either / or \\
        # as a path separator. Here we convert it to what the OS uses.
        if os.path.sep == '/':
            cmap = cmap.replace('\\', '/')
        elif os.path.sep == '\\':
            cmap = cmap.replace('/', '\\')

        index = self.cb_cmaps.findText(cmap)

        if index != -1:
            self.cb_cmaps.setCurrentIndex(index)
        else:
            logger.error('Could not find the colormap file %s' % cmap)

    def load_dat_file(self, filename):
        """
        Load a .dat file, it's .set file if present, update the GUI elements,
        and fire an on_data_change event to update the plots.
        """
        self.dat_file = DatFile(filename)
        self.settings.fill_tree()

        if filename != self.filename:
            path, self.name = os.path.split(filename)
            self.filename = filename
            self.abs_filename = os.path.abspath(filename)

            self.open_state(self.profile_ini_file)

            # self.update_ui()

        # self.on_data_change()

    def update_parameters(self):
        pass

    def save_default_profile(self, file):
        self.qtplot_ini.set('DEFAULT', 'default_profile', file)

        with open(self.qtplot_ini_file, 'w') as config_file:
            self.qtplot_ini.write(config_file)

    def save_state(self, filename):
        """
        Save the current qtplot state into a .ini file and the operations
        in a corresponding .json file.
        """
        profile_name = os.path.splitext(os.path.basename(filename))[0]

        operations_file = os.path.join(self.operations_dir,
                                       profile_name + '.json')

        self.operations.save(operations_file)

        state = OrderedDict((
            ('operations', operations_file),
            ('sub_series_V', str(self.cb_v.currentText())),
            ('sub_series_I', str(self.cb_i.currentText())),
            ('sub_series_R', str(self.le_r.text())),
            ('open_directory', self.profile_settings['open_directory']),
            ('save_directory', self.profile_settings['save_directory']),
            ('x', str(self.cb_x.currentText())),
            ('y', str(self.cb_y.currentText())),
            ('z', str(self.cb_z.currentText())),
            ('colormap', str(self.cb_cmaps.currentText())),
            ('title', str(self.export_widget.le_title.text())),
            ('DPI', str(self.export_widget.le_dpi.text())),
            ('rasterize', self.export_widget.cb_rasterize.isChecked()),
            ('x_label', str(self.export_widget.le_x_label.text())),
            ('y_label', str(self.export_widget.le_y_label.text())),
            ('z_label', str(self.export_widget.le_z_label.text())),
            ('x_format', str(self.export_widget.le_x_format.text())),
            ('y_format', str(self.export_widget.le_y_format.text())),
            ('z_format', str(self.export_widget.le_z_format.text())),
            ('x_div', str(self.export_widget.le_x_div.text())),
            ('y_div', str(self.export_widget.le_y_div.text())),
            ('z_div', str(self.export_widget.le_z_div.text())),
            ('font', str(self.export_widget.le_font.text())),
            ('font_size', str(self.export_widget.le_font_size.text())),
            ('width', str(self.export_widget.le_width.text())),
            ('height', str(self.export_widget.le_height.text())),
            ('cb_orient', str(self.export_widget.cb_cb_orient.currentText())),
            ('cb_pos', str(self.export_widget.le_cb_pos.text())),
            ('triangulation', self.export_widget.cb_triangulation.isChecked()),
            ('tripcolor', self.export_widget.cb_tripcolor.isChecked()),
            ('linecut', self.export_widget.cb_linecut.isChecked()),
            ('line_style', str(self.linecut.cb_linestyle.currentText())),
            ('line_width', str(self.linecut.le_linewidth.text())),
            ('marker_style', str(self.linecut.cb_markerstyle.currentText())),
            ('marker_size', str(self.linecut.le_markersize.text())),
        ))

        for option, value in state.items():
            # ConfigParser doesn't like single %
            value = str(value).replace('%', '%%')

            self.profile_ini.set('DEFAULT', option, value)

        path = os.path.join(self.profiles_dir, filename)

        with open(path, 'w') as config_file:
            self.profile_ini.write(config_file)

    def open_state(self, filename):
        """ Load all settings into the GUI """
        self.profile_ini_file = os.path.join(self.profiles_dir, filename)
        self.profile_name = os.path.splitext(os.path.basename(filename))[0]

        operations_file = os.path.join(self.operations_dir,
                                       self.profile_name + '.json')

        # Load the operations
        if os.path.exists(operations_file):
            self.operations.load(operations_file)
        else:
            logger.warning('No operations file present for selected profile')

        # Read the specified profile .ini
        self.profile_ini.read(self.profile_ini_file)

        # Load the profile into a dict
        for option in profile_defaults.keys():
            value = self.profile_ini.get('DEFAULT', option)

            if value in ['False', 'True']:
                value = self.profile_ini.getboolean('DEFAULT', option)

            self.profile_settings[option] = value

        try:
            R = float(self.profile_settings['sub_series_R'])

            self.sub_series_r(self.profile_settings['sub_series_V'],
                              self.profile_settings['sub_series_I'],
                              R)
        except ValueError:
            logger.warning('Could not parse resistance value in the profile')

        self.update_ui(opening_state=True)

        self.on_data_change()

        self.on_cmap_change()

        self.export_widget.populate_ui()
        self.linecut.populate_ui()

        # If we are viewing the export tab, update the plot
        if self.main_widget.currentWidget() == self.export_widget:
            self.export_widget.on_update()

    def get_parameter_names(self):
        if self.dat_file is not None:
            # return list(self.dat_file.df.columns.values)
            return sorted(self.dat_file.ids)
        elif self.data_set is not None:
            # qcodes data set
            return sorted(list(self.data_set.arrays))
        else:
            return []

    def set_data_set(self, data_set, update_ui=False):
        self.data_set = data_set

        if update_ui:
            self.update_ui()

        self.on_data_change()

    def on_data_change(self):
        """
        This is called when anything concerning the data has changed. This can
        consist of a new data file being loaded, a change in parameter to plot,
        or a change/addition of an Operation.

        A clean version of the Data2D is retrieved from the DatFile or DataSet,
        all the operations are applied to the data, and it is plotted.
        """
        if self.dat_file is None and self.data_set is None:
            return

        # Get the selected axes from the interface
        x_name, y_name, data_name = self.get_axis_names()

        # Update the Data2D from either a qtlab or qcodes dataset
        if self.dat_file is not None:
            self.data = self.dat_file.get_data(x_name, y_name, data_name)

            if self.data is None:
                return
        elif self.data_set is not None:
            # Create a Data2D object from qcodes DataSet
            x = self.data_set.arrays[x_name].array
            y = self.data_set.arrays[y_name].array
            z = self.data_set.arrays[data_name].array

            self.data = Data2D(x, y, z)

        # Apply the selected operations
        self.data = self.operations.apply_operations(self.data)

        # If we want to reset the colormap for each data update, do so
        if self.cb_reset_cmap.checkState() == QtCore.Qt.Checked:
            self.on_min_changed(0)
            self.s_gamma.setValue(0)
            self.on_max_changed(100)

        self.canvas.set_data(self.data)
        # Update the linecut
        self.canvas.draw_linecut(None, old_position=True)

        if np.isnan(self.data.z).any():
            logger.warning('The data contains NaN values')

    def get_axis_names(self):
        """ Get the parameters that are currently selected to be plotted """
        self.x_name = str(self.cb_x.currentText())
        self.y_name = str(self.cb_y.currentText())
        self.data_name = str(self.cb_z.currentText())

        return self.x_name, self.y_name, self.data_name

    def on_load_dat(self, event):
        open_directory = self.profile_settings['open_directory']
        filename = str(QtGui.QFileDialog.getOpenFileName(directory=open_directory,
                                                         filter='*.dat *.json'))

        if filename != "":
            self.load_dat_file(filename)

    def on_refresh(self, event):
        if self.filename:
            self.load_dat_file(self.filename)

            self.on_data_change()

    def on_swap_axes(self, event):
        x, y = self.cb_x.currentIndex(), self.cb_y.currentIndex()
        self.cb_x.setCurrentIndex(y)
        self.cb_y.setCurrentIndex(x)

        self.on_data_change()

    def sub_series_r(self, V_param, I_param, R):
        if self.dat_file is None:
            return

        if (V_param in self.dat_file.ids and I_param in self.dat_file.ids):
            voltages = self.dat_file.get_column(V_param)
            currents = self.dat_file.get_column(I_param)
            adjusted = voltages - currents * R

            self.dat_file.set_column(V_param + ' - Sub series R', adjusted)

    def on_sub_series_r(self, event=None):
        V_param = str(self.cb_v.currentText())

        self.sub_series_r(V_param,
                          str(self.cb_i.currentText()),
                          float(self.le_r.text()))

        self.update_ui(reset=False)

        x_col = str(self.cb_x.currentText())
        y_col = str(self.cb_y.currentText())

        # If the current x/y axis was the voltage axis to be corrected
        # then switch to the corrected values
        if V_param == x_col:
            self.cb_x.setCurrentIndex(self.cb_x.count() - 1)
        elif V_param == y_col:
            self.cb_y.setCurrentIndex(self.cb_y.count() - 1)

        self.on_data_change()

    def on_cmap_change(self, event=None):
        selected_cmap = str(self.cb_cmaps.currentText())

        path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(path, 'colormaps', selected_cmap)

        new_colormap = Colormap(path)

        new_colormap.min = self.canvas.colormap.min
        new_colormap.max = self.canvas.colormap.max

        self.canvas.colormap = new_colormap
        self.canvas.update()

    def on_min_max_entered(self):
        if self.data is not None:
            zmin, zmax = np.nanmin(self.data.z), np.nanmax(self.data.z)

            newmin = float(self.le_min.text())
            newmax = float(self.le_max.text())

            # Convert the entered bounds into slider positions (0 - 100)
            self.s_min.setValue((newmin - zmin) / ((zmax - zmin) / 100))
            self.s_max.setValue((newmax - zmin) / ((zmax - zmin) / 100))

            cm = self.canvas.colormap
            cm.min, cm.max = newmin, newmax

            self.canvas.update()

    def on_min_changed(self, value):
        if self.data is not None:
            min, max = np.nanmin(self.data.z), np.nanmax(self.data.z)

            newmin = min + (max - min) * (value / 99.0)
            self.le_min.setText('%.2e' % newmin)

            self.canvas.colormap.min = newmin
            self.canvas.update()

    def on_gamma_changed(self, value):
        if self.data is not None:
            gamma = 10.0**(value / 100.0)

            self.canvas.colormap.gamma = gamma
            self.canvas.update()

    def on_max_changed(self, value):
        if self.data is not None:
            min, max = np.nanmin(self.data.z), np.nanmax(self.data.z)

            # This stuff with the 99 is hacky, something is going on which
            # causes the highest values not to be rendered using the colormap.
            # The 99 makes the cm max a bit higher than the actual maximum
            newmax = min + (max - min) * (value / 99.0)
            self.le_max.setText('%.2e' % newmax)

            self.canvas.colormap.max = newmax
            self.canvas.update()

    def on_cm_reset(self):
        if self.data is not None:
            self.s_min.setValue(0)
            self.on_min_changed(0)
            self.s_gamma.setValue(0)
            self.s_max.setValue(100)
            self.on_max_changed(100)

    def on_save_matrix(self):
        save_directory = self.profile_settings['save_directory']

        filters = ('QTLab data format (*.dat);;'
                   'NumPy binary matrix format (*.npy);;'
                   'MATLAB matrix format (*.mat)')

        filename = QtGui.QFileDialog.getSaveFileName(self,
                                                     caption='Save file',
                                                     directory=save_directory,
                                                     filter=filters)
        filename = str(filename)

        if filename != '' and self.dat_file is not None:
            base = os.path.basename(filename)
            name, ext = os.path.splitext(base)

            self.data.save(filename)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            url = str(event.mimeData().urls()[0].toString())

            if url.endswith('.dat'):
                event.accept()

    def dropEvent(self, event):
        filepath = str(event.mimeData().urls()[0].toLocalFile())

        self.load_dat_file(filepath)

    def closeEvent(self, event):
        self.linecut.close()
        self.operations.close()
        self.settings.close()
        self.closed = True


def main():
    app = QtGui.QApplication(sys.argv)

    if len(sys.argv) > 1:
        QTPlot(filename=sys.argv[1])
    else:
        QTPlot()

    sys.exit(app.exec_())
