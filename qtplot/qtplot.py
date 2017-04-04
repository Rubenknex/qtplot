import logging
import os
import sys

import numpy as np
from PyQt4 import QtCore, QtGui, uic

from .canvas import Canvas
from .export import Export
from .model import Model
from .linetrace import Linetrace
from .operations import Operations
from .settings import Settings
from .util import eng_format

logger = logging.getLogger(__name__)


class QTPlot(QtGui.QMainWindow):
    """
    This class contains all the logic behind the user interface and
    serves as the connection between the views (user interface) and
    the model (data).

    High priority:
    - Interp grid doesn't work
    - Subtract series resistance
        - make this work with easier syntax
        - data['new_col'] = data['V_bla'] - 350 * data['I_bla']
    - Arbitrary linetrace: update triangulation

    Low priority:
    - Proper linetrace speed
    - Update arbitrary trace on swap axes?
    """
    def __init__(self):
        super(QTPlot, self).__init__()

        self.model = Model()

        path = os.path.join(self.model.dir, 'ui/main.ui')
        uic.loadUi(path, self)

        self.canvas = Canvas()
        self.canvas_layout.addWidget(self.canvas.native)

        self.load_colormaps()

        center = QtGui.QApplication.desktop().screen().rect().center()
        self.move(center.x() - 550, center.y() - 400)

        self.show()

        self.linetrace = Linetrace(self, self.model)
        self.operations = Operations(self, self.model)
        self.settings = Settings(self, self.model)
        self.export = Export(self, self.model)

        self.tabs.addTab(self.export, 'Export')

        self.l_position = QtGui.QLabel()
        self.statusbar.addWidget(self.l_position, 1)
        self.l_slope = QtGui.QLabel()
        self.statusbar.addWidget(self.l_slope)

        self.cb_parameters = [self.cb_x, self.cb_y, self.cb_z]
        self.sliders = [self.s_cmap_min, self.s_cmap_gamma, self.s_cmap_max]

        self.bind()

        self.model.init_settings()

        self.set_keyboard_shortcuts(self.model.settings['shortcuts'])

        self.settings.initialize()

    def bind(self):
        """ Set up connections for GUI and model events """
        self.b_load.clicked.connect(self.on_load)
        self.b_save_data.clicked.connect(self.on_save)
        self.b_refresh.clicked.connect(self.model.refresh)
        self.b_swap_axes.clicked.connect(self.model.swap_axes)

        # Raising hidden windows
        self.b_linetrace.clicked.connect(self.linetrace.show_window)
        self.b_operations.clicked.connect(self.operations.show_window)
        self.b_settings.clicked.connect(self.settings.show_window)

        for cb in self.cb_parameters:
            cb.activated.connect(self.on_parameters_changed)

        # Colormap settings
        self.cb_colormap.currentIndexChanged.connect(self.on_cmap_chosen)
        self.b_reset_cmap.clicked.connect(self.on_cmap_reset)
        self.le_cmap_min.returnPressed.connect(self.on_cmap_edit_changed)
        self.le_cmap_max.returnPressed.connect(self.on_cmap_edit_changed)

        for s in self.sliders:
            s.sliderMoved.connect(self.on_cmap_slider_changed)

        # Canvas events
        self.canvas.events.mouse_press.connect(self.on_canvas_press)
        self.canvas.events.mouse_move.connect(self.on_canvas_move)

        # Model events
        self.model.profile_changed.connect(self.set_profile)
        self.model.data_file_changed.connect(self.on_data_file_changed)
        self.model.data2d_changed.connect(self.on_data2d_changed)
        self.model.cmap_changed.connect(self.on_cmap_changed)
        self.model.linetrace_changed.connect(self.on_linetrace_changed)

    def set_keyboard_shortcuts(self, shortcuts):
        """ Set up the keyboard shortcuts """
        # We have to set the shortcuts for every window separately
        for parent in [self, self.linetrace, self.settings, self.operations]:
            for key, value in shortcuts.items():
                try:
                    func = getattr(self.model, key)
                    QtGui.QShortcut(QtGui.QKeySequence(value), parent, func)
                except AttributeError:
                    print('Could not find shortcut function:', key)

    def get_profile(self):
        """ Create the dict that holds the current qtplot settings """
        state = {
            'x': str(self.cb_x.currentText()),
            'y': str(self.cb_y.currentText()),
            'z': str(self.cb_z.currentText()),
            'colormap': str(self.cb_colormap.currentText()),
        }

        linetrace_state = self.linetrace.get_state()
        export_state = self.export.get_state()

        profile = {}

        for d in [state, linetrace_state, export_state]:
            profile.update(d)

        return profile

    def set_profile(self, profile):
        """ Set the qtplot settings from a profile dict """
        cmap = profile['colormap']

        # The path that is saved in the profile can use either / or \\
        # as a path separator. Here we convert it to what the OS uses.
        if os.path.sep == '/':
            cmap = cmap.replace('\\', '/')
        elif os.path.sep == '\\':
            cmap = cmap.replace('/', '\\')

        index = self.cb_colormap.findText(cmap)

        if index != -1:
            self.cb_colormap.setCurrentIndex(index)

        if self.model.data_file is not None:
            names = ['x', 'y', 'z']
            default_indices = [1, 2, 4]

            # Try to find the parameter that is in the profile in the data
            for i, cb in enumerate(self.cb_parameters):
                idx = cb.findText(profile[names[i]])

                # If it's not there, use a default index
                if idx <= 0:
                    idx = default_indices[i]

                cb.setCurrentIndex(idx)

            self.on_parameters_changed()

    def load_colormaps(self):
        """ Populate the colormap combobox with all filenames """
        directory = os.path.dirname(os.path.realpath(__file__))

        directory = os.path.join(directory, 'colormaps')

        cmap_files = []
        for dir, _, files in os.walk(directory):
            for filename in files:
                reldir = os.path.relpath(dir, directory)
                relfile = os.path.join(reldir, filename)

                # Remove .\ for files in the root of the directory
                if relfile[:2] in ['.\\', './']:
                    relfile = relfile[2:]

                cmap_files.append(relfile)

        self.cb_colormap.addItems(cmap_files)
        self.cb_colormap.setMaxVisibleItems(25)

        self.canvas.colormap = self.model.colormap

    def on_load(self):
        """ Load a datafile """
        open_dir = self.model.profile['open_directory']
        filename = str(QtGui.QFileDialog.getOpenFileName(directory=open_dir,
                                                         filter='*.dat'))

        if filename != '':
            self.model.load_data_file(filename)

    def on_save(self):
        """ Save the current data """
        if self.data2d is None:
            return

        save_directory = self.model.profile['save_directory']

        filters = ('QTLab data format (*.dat);;'
                   'NumPy binary matrix format (*.npy);;'
                   'MATLAB matrix format (*.mat)')

        filename = QtGui.QFileDialog.getSaveFileName(caption='Save file',
                                                     directory=save_directory,
                                                     filter=filters)
        filename = str(filename)

        if filename != '' and self.model.data2d is not None:
            base = os.path.basename(filename)
            name, ext = os.path.splitext(base)

            self.data2d.save(filename)

    def on_parameters_changed(self):
        """ One of the parameters to plot has changed """
        if self.model.data_file is not None:
            params = [str(cb.currentText()) for cb in self.cb_parameters]
            self.model.select_parameters(*params)

    def on_cmap_chosen(self):
        """ A new colormap has been selected """
        cmap_name = str(self.cb_colormap.currentText())
        self.model.set_colormap(cmap_name)

    def on_cmap_reset(self):
        """ The colormap settings are reset """
        min, max = self.model.data2d.get_z_limits()

        self.model.set_colormap_settings(min, max, 1.0)

    def on_cmap_edit_changed(self):
        """ Update the colormap from the new min/max settings """
        new_min = float(self.le_cmap_min.text())
        new_max = float(self.le_cmap_max.text())

        gamma = self.model.colormap.gamma
        self.model.set_colormap_settings(new_min, new_max, gamma)

    def on_cmap_slider_changed(self, value):
        """ Update the colormap from the new slider positions """
        zmin, zmax = self.model.data2d.get_z_limits()

        min, max, gamma = self.model.colormap.get_settings()

        if self.s_cmap_min.isSliderDown():
            min = zmin + (zmax - zmin) * (value / 99.0)

        if self.s_cmap_gamma.isSliderDown():
            gamma = 10.0**(value / 100.0)

        if self.s_cmap_max.isSliderDown():
            max = zmin + (zmax - zmin) * (value / 99.0)

        self.model.set_colormap_settings(min, max, gamma)

    def on_canvas_press(self, event, initial_press=True):
        """ React to a mouse button press on the canvas """
        x, y = self.canvas.screen_to_data_coords(tuple(event.pos))

        type = {1: 'horizontal', 2: 'arbitrary', 3: 'vertical'}[event.button]

        incremental = self.linetrace.get_incremental()
        self.model.take_linetrace(x, y, type, incremental=incremental,
                                  initial_press=initial_press)

    def on_canvas_move(self, event):
        """ React to a mouse move on the canvas """
        # Update the mouse position in the statusbar
        if self.model.data2d is not None:
            x, y = self.canvas.screen_to_data_coords((event.pos[0],
                                                      event.pos[1]))

            if not np.isnan(x) and not np.isnan(y):
                # Show the coordinates in the statusbar
                text = 'X: %s\tY: %s' % (eng_format(x, 1), eng_format(y, 1))
                self.l_position.setText(text)

        # If a mouse button is held down, fire the press event
        if len(event.buttons) > 0:
            self.on_canvas_press(event, initial_press=False)

        # From here on handlers of changes in the model
    def on_data_file_changed(self, different_file):
        self.setWindowTitle(self.model.data_file.name)

        for cb in self.cb_parameters:
            cb.clear()
            cb.addItems([''] + self.model.data_file.columns)

        self.set_profile(self.model.profile)

    def on_data2d_changed(self, event=None):
        # Reset the colormap if required
        reset_cmap = self.cb_reset_on_plot.checkState() == QtCore.Qt.Checked

        if reset_cmap:
            min, max = self.model.data2d.get_z_limits()

            self.model.set_colormap_settings(min, max, 1.0)

        # Update the UI if necessary
        idx = self.cb_x.findText(self.model.x)
        self.cb_x.setCurrentIndex(idx)
        idx = self.cb_y.findText(self.model.y)
        self.cb_y.setCurrentIndex(idx)

        xq, yq = self.model.data2d.get_quadrilaterals()
        self.canvas.set_data(xq, yq, self.model.data2d.z)

        # Update the current linetrace
        self.model.update_linetrace()

    def on_cmap_changed(self):
        """ The colormap settings changed, so update the UI """
        if self.model.data2d is None:
            return

        min, max, gamma = self.model.colormap.get_settings()

        zmin, zmax = self.model.data2d.get_z_limits()

        # Convert into slider positions (0 - 100)
        # Don't adjust the slider if it's currently being used
        if not self.s_cmap_min.isSliderDown():
            new_val = (min - zmin) / ((zmax - zmin) / 100)
            self.s_cmap_min.setValue(new_val)

        if not self.s_cmap_gamma.isSliderDown():
            new_val = np.log10(gamma) * 100.0
            self.s_cmap_gamma.setValue(new_val)

        if not self.s_cmap_max.isSliderDown():
            new_val = (max - zmin) / ((zmax - zmin) / 100)
            self.s_cmap_max.setValue(new_val)

        self.le_cmap_min.setText('%.2e' % min)
        self.le_cmap_max.setText('%.2e' % max)

        # Update the colormap and plot
        self.canvas.colormap = self.model.colormap
        self.canvas.update()

    def on_linetrace_changed(self, event, redraw=False, line=None):
        if line is not None:
            # Let the canvas know how to draw the linetrace
            self.canvas.set_linetrace_data(*line.get_positions())

            # Display the slope of the arbitrary linetrace
            if line.type == 'arbitrary':
                slope = line.get_slope()
                text = 'Slope: {:.3e}\tInv: {:.3e}'.format(slope, 1 / slope)

                self.l_slope.setText(text)
            else:
                self.l_slope.setText('')


def main():
    """ Entry point for qtplot """
    logging.basicConfig(level=logging.WARNING)

    app = QtGui.QApplication(sys.argv)

    if len(sys.argv) > 1:
        c = QTPlot(filename=sys.argv[1])
    else:
        c = QTPlot()

    sys.exit(app.exec_())
