from __future__ import print_function

from six.moves import configparser
import matplotlib as mpl
import numpy as np
import os
import sys

from PyQt4 import QtGui, QtCore
from scipy import io

from .colormap import Colormap
from .data import DatFile, Data2D
from .export import ExportWidget
from .linecut import Linecut
from .operations import Operations
from .settings import Settings
from .canvas import Canvas


class MainWindow(QtGui.QMainWindow):
    """The main window of the qtplot application."""
    def __init__(self, filename=None):
        super(MainWindow, self).__init__(None)

        # Set some matplotlib font settings
        mpl.rcParams['mathtext.fontset'] = 'custom'
        mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

        # Load the open and save directories from qtplot.ini
        self.open_directory = self.read_from_ini('Settings', 'OpenDirectory')
        if self.open_directory is None:
            path = os.path.dirname(os.path.realpath(__file__))
            self.open_directory = path

        self.save_directory = self.read_from_ini('Settings', 'SaveDirectory')
        if self.save_directory is None:
            path = os.path.dirname(os.path.realpath(__file__))
            self.save_directory = path

        self.first_data_file = True
        self.name = None

        # In case of a .dat file
        self.filename = None
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

        if filename is not None:
            self.load_dat_file(filename)

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

        self.b_swap = QtGui.QPushButton('Swap dep. var.', self)
        self.b_swap.clicked.connect(self.on_swap_order)
        hbox.addWidget(self.b_swap)

        self.b_linecut = QtGui.QPushButton('Linecut')
        self.b_linecut.clicked.connect(self.linecut.show_window)
        hbox.addWidget(self.b_linecut)

        self.b_operations = QtGui.QPushButton('Operations')
        self.b_operations.clicked.connect(self.operations.show_window)
        hbox.addWidget(self.b_operations)

        # Subtracting series R
        r_hbox = QtGui.QHBoxLayout()

        lbl_sub = QtGui.QLabel('Sub series R:')
        r_hbox.addWidget(lbl_sub)

        lbl_v = QtGui.QLabel('V:')
        lbl_v.setMaximumWidth(20)
        r_hbox.addWidget(lbl_v)

        self.cb_v = QtGui.QComboBox(self)
        r_hbox.addWidget(self.cb_v)

        lbl_i = QtGui.QLabel('I:')
        lbl_i.setMaximumWidth(20)
        r_hbox.addWidget(lbl_i)

        self.cb_i = QtGui.QComboBox(self)
        r_hbox.addWidget(self.cb_i)

        lbl_r = QtGui.QLabel('R:')
        lbl_r.setMaximumWidth(20)
        r_hbox.addWidget(lbl_r)

        self.le_r = QtGui.QLineEdit(self)
        self.le_r.setMaximumWidth(100)
        self.le_r.returnPressed.connect(self.on_sub_series_r)
        r_hbox.addWidget(self.le_r)

        self.b_ok = QtGui.QPushButton('Ok', self)
        self.b_ok.clicked.connect(self.on_sub_series_r)
        self.b_ok.setMaximumWidth(50)
        r_hbox.addWidget(self.b_ok)

        # Selecting columns and orders
        #groupbox = QtGui.QGroupBox('Data')
        grid = QtGui.QGridLayout()
        #groupbox.setLayout(grid)

        lbl_x = QtGui.QLabel("X:", self)
        grid.addWidget(lbl_x, 1, 1)

        self.cb_x = QtGui.QComboBox(self)
        self.cb_x.activated.connect(self.on_data_change)
        grid.addWidget(self.cb_x, 1, 2)

        lbl_order_x = QtGui.QLabel('Dependent variable X: ', self)
        grid.addWidget(lbl_order_x, 1, 3)

        self.cb_order_x = QtGui.QComboBox(self)
        self.cb_order_x.activated.connect(self.on_data_change)
        grid.addWidget(self.cb_order_x, 1, 4)

        lbl_y = QtGui.QLabel("Y:", self)
        grid.addWidget(lbl_y, 2, 1)

        self.cb_y = QtGui.QComboBox(self)
        self.cb_y.activated.connect(self.on_data_change)
        grid.addWidget(self.cb_y, 2, 2)

        lbl_order_y = QtGui.QLabel('Dependent variable Y:', self)
        grid.addWidget(lbl_order_y, 2, 3)

        self.cb_order_y = QtGui.QComboBox(self)
        self.cb_order_y.activated.connect(self.on_data_change)
        grid.addWidget(self.cb_order_y, 2, 4)

        lbl_d = QtGui.QLabel("Data:", self)
        grid.addWidget(lbl_d, 3, 1)

        self.cb_z = QtGui.QComboBox(self)
        self.cb_z.activated.connect(self.on_data_change)
        grid.addWidget(self.cb_z, 3, 2)

        #self.cb_save_default = QtGui.QCheckBox('Remember columns')
        #grid.addWidget(self.cb_save_default, 3, 3)

        self.b_save_default = QtGui.QPushButton('Set as defaults')
        self.b_save_default.clicked.connect(self.on_save_default)
        grid.addWidget(self.b_save_default, 3, 4)

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
        #vbox.addWidget(self.toolbar)
        vbox.addWidget(self.canvas.native)
        vbox.addLayout(hbox)
        vbox.addLayout(r_hbox)
        #vbox.addLayout(grid)
        vbox.addWidget(groupbox)
        #vbox.addLayout(hbox_gamma)
        vbox.addWidget(groupbox_gamma)
        vbox.addLayout(hbox4)

        self.status_bar = QtGui.QStatusBar()
        self.setStatusBar(self.status_bar)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.resize(600, 700)
        self.move(100, 100)

        self.setAcceptDrops(True)

        self.linecut.show()
        self.operations.show()
        self.show()
        
    def update_ui(self, reset=True):
        """
        Update the user interface, typically called on loading new data (not
        on updating data).

        The x/y/z parameter selections are populated. If this is the first data
        that is loaded, it is checked if parameters set as default in the .ini
        are present, which are then immediately selected.
        """
        self.setWindowTitle(self.name)

        parameters = self.get_parameter_names()

        i = self.cb_v.currentIndex()
        self.cb_v.clear()
        self.cb_v.addItems(parameters)
        self.cb_v.setCurrentIndex(i)

        i = self.cb_i.currentIndex()
        self.cb_i.clear()
        self.cb_i.addItems(parameters)
        self.cb_i.setCurrentIndex(i)

        i = self.cb_x.currentIndex()
        self.cb_x.clear()
        self.cb_x.addItems(parameters)
        self.cb_x.setCurrentIndex(i)

        i = self.cb_y.currentIndex()
        self.cb_y.clear()
        self.cb_y.addItems(parameters)
        self.cb_y.setCurrentIndex(i)

        i = self.cb_z.currentIndex()
        self.cb_z.clear()
        self.cb_z.addItems(parameters)
        self.cb_z.setCurrentIndex(i)

        if self.dat_file is not None:
            i = self.cb_order_x.currentIndex()
            self.cb_order_x.clear()
            self.cb_order_x.addItems(parameters)
            self.cb_order_x.setCurrentIndex(i)

            i = self.cb_order_y.currentIndex()
            self.cb_order_y.clear()
            self.cb_order_y.addItems(parameters)
            self.cb_order_y.setCurrentIndex(i)

        if reset and self.first_data_file:
            self.first_data_file = False

            combo_boxes = [self.cb_x, self.cb_order_x, self.cb_y, self.cb_order_y, self.cb_z]
            names = ['X', 'X Order', 'Y', 'Y Order', 'Data']
            default_indices = [0, 0, 1, 1, 3]

            result = self.read_from_ini('Settings', names)

            if result is not None:
                # Check if the names in the ini file are present in
                # the loaded data otherwise, use the default index
                for i, cb in enumerate(combo_boxes):
                    index = cb.findText(result[i])

                    if index == -1:
                        cb.setCurrentIndex(default_indices[i])
                    else:
                        cb.setCurrentIndex(index)
            else:
                for i, cb in enumerate(combo_boxes):
                    cb.setCurrentIndex(default_indices[i])

    def get_parameter_names(self):
        if self.dat_file is not None:
            return self.dat_file.df.columns.values
        elif self.data_set is not None:
            # Sort in some kind of order?
            # Make property of DataSetLite?
            # TODO: Use full names/labels
            return list(self.data_set.arrays)
        else:
            return []

    def write_to_ini(self, section, keys_values):
        """ 
        Write settings to the qtplot.ini file. If the file is not present, a
        new one will be created.
        """
        path = os.path.dirname(os.path.realpath(__file__))
        filepath = os.path.join(path, '../qtplot.ini')

        config = configparser.ConfigParser()

        if os.path.isfile(filepath):
            config.read(filepath)

        if not config.has_section(section):
            config.add_section(section)

        for key, value in keys_values.items():
            config.set(section, key, value)

        with open(filepath, 'w') as config_file:
            config.write(config_file)

    def read_from_ini(self, section, options):
        """
        Read settings from the qtplot.ini file. If the file is not present,
        None will be returned.
        """
        path = os.path.dirname(os.path.realpath(__file__))
        filepath = os.path.join(path, '../qtplot.ini')

        config = configparser.ConfigParser()

        if os.path.isfile(filepath):
            config.read(filepath)

            if type(options) == str:
                options = [options]

            has_options = [config.has_option(section, option) for option in options]

            if False not in has_options:
                values = [config.get(section, option) for option in options]

                if len(values) == 1:
                    return values[0]
                else:
                    return values

        return None

    def load_dat_file(self, filename):
        """ 
        Load a .dat file, it's .set file if present, update the GUI elements,
        and fire an on_data_change event to update the plots.
        """
        self.dat_file = DatFile(filename)
        self.settings.load_file(filename)

        if filename != self.filename:
            path, self.name = os.path.split(filename)
            self.filename = filename

            self.update_ui()

        self.on_data_change()

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
        x_name, y_name, data_name, order_x, order_y = self.get_axis_names()

        self.export_widget.set_info(self.name, x_name, y_name, data_name)
        
        if self.dat_file is not None:
            not_found = self.dat_file.has_columns([x_name, y_name, data_name, order_x, order_y])
            if not_found is not None:
                self.status_bar.showMessage('ERROR: Could not find column \'' +
                    not_found + '\', try saving the correct one using \'Remember columns\'')

                return

            try:
                self.data = self.dat_file.get_data(x_name, y_name, data_name, order_x, order_y)
            except Exception:
                print('ERROR: Cannot pivot data into a matrix with these columns')

                return
        elif self.data_set is not None:
            x = self.data_set.arrays[x_name].array
            y = self.data_set.arrays[y_name].array
            z = self.data_set.arrays[data_name].array

            self.data = Data2D(x, y, z)

        self.data = self.operations.apply_operations(self.data)

        if self.cb_reset_cmap.checkState() == QtCore.Qt.Checked:
            self.on_min_changed(0)
            self.s_gamma.setValue(0)
            self.on_max_changed(100)

        self.canvas.set_data(self.data)
        #self.canvas.draw_linecut(None, old_position=True)
        self.canvas.update()

        if np.isnan(self.data.z).any():
            self.status_bar.showMessage("Warning: Data contains NaN values")
        else:
            self.status_bar.showMessage("")

    def get_axis_names(self):
        """ Get the parameters that are currently selected to be plotted """
        x_name = str(self.cb_x.currentText())
        y_name = str(self.cb_y.currentText())
        data_name = str(self.cb_z.currentText())
        order_x = str(self.cb_order_x.currentText())
        order_y = str(self.cb_order_y.currentText())

        return x_name, y_name, data_name, order_x, order_y

    def on_load_dat(self, event):
        filename = str(QtGui.QFileDialog.getOpenFileName(directory=self.open_directory,
                                                         filter='*.dat'))

        if filename != "":
            self.load_dat_file(filename)

    def on_refresh(self, event):
        if self.filename:
            self.load_dat_file(self.filename)

    def on_swap_axes(self, event):
        x, y = self.cb_x.currentIndex(), self.cb_y.currentIndex()
        self.cb_x.setCurrentIndex(y)
        self.cb_y.setCurrentIndex(x)

        self.on_swap_order(event)

    def on_swap_order(self, event):
        x, y = self.cb_order_x.currentIndex(), self.cb_order_y.currentIndex()
        self.cb_order_x.setCurrentIndex(y)
        self.cb_order_y.setCurrentIndex(x)

        self.on_data_change()

    def on_save_default(self, event):
        x_name, y_name, data_name, order_x, order_y = self.get_axis_names()

        axes = {
            'X':        x_name,
            'X order':  order_x,
            'Y':        y_name,
            'Y Order':  order_y,
            'Data':     data_name,
        }

        self.write_to_ini('Settings', axes)

    def on_sub_series_r(self, event=None):
        if self.dat_file is None:
            return

        V, I = str(self.cb_v.currentText()), str(self.cb_i.currentText())
        R = float(self.le_r.text())

        self.dat_file.df[V + ' - Sub series R'] = self.dat_file.df[V] - self.dat_file.df[I] * R

        self.update_ui(reset=False)

        x_col = str(self.cb_x.currentText())
        y_col = str(self.cb_y.currentText())

        if V == x_col:
            self.cb_x.setCurrentIndex(self.cb_x.count() - 1)
        elif V == y_col:
            self.cb_y.setCurrentIndex(self.cb_y.count() - 1)

        self.on_data_change()

    def on_cmap_change(self, event):
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

            newmin, newmax = float(self.le_min.text()), float(self.le_max.text())

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
        filename = QtGui.QFileDialog.getSaveFileName(self,
                                                     caption='Save file',
                                                     directory=self.save_directory,
                                                     filter='NumPy matrix format (*.npy);;MATLAB matrix format (*.mat)')
        filename = str(filename)

        if filename != '' and self.dat_file is not None:
            base = os.path.basename(filename)
            name, ext = os.path.splitext(base)

            mat = np.dstack((self.data.x.data, self.data.y.data, self.data.z.data))

            if ext == '.npy':
                np.save(filename, mat)
            elif ext == '.mat':
                io.savemat(filename, {'data': mat})

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
