import os
import sys

import numpy as np
from PyQt4 import QtGui

from .model import Model
from .view import MainView, LineView, OperationsView, SettingsView


class Controller:
    """
    This class contains all the logic behind the user interface and
    serves as the connection between the views (user interface) and
    the model (data).
    """
    def __init__(self):
        self.model = Model()

        self.main_view = MainView()
        self.line_view = LineView(self.main_view)
        self.op_view = OperationsView(self.main_view)
        self.set_view = SettingsView(self.main_view)

        self.load_colormaps()

        self.setup_view_to_controller()
        self.setup_model_to_controller()

    def setup_view_to_controller(self):
        """
        Set up the connections listening for user interface events
        from the view.
        """
        self.main_view.b_load.clicked.connect(self.on_load)

        for cb in self.main_view.cb_parameters:
            cb.activated.connect(self.on_parameters_changed)

        self.main_view.cb_colormap.activated.connect(self.on_cmap_chosen)
        self.main_view.b_reset_cmap.clicked.connect(self.on_cmap_reset)

        self.main_view.le_cmap_min.returnPressed.connect(self.on_cmap_edit_changed)
        self.main_view.le_cmap_max.returnPressed.connect(self.on_cmap_edit_changed)

        for s in self.main_view.sliders:
            s.sliderMoved.connect(self.on_cmap_slider_changed)

    def setup_model_to_controller(self):
        """
        Set up the connections listening for changes fired by the model.
        """
        self.model.data_file_changed.connect(self.on_data_file_changed)
        self.model.data2d_changed.connect(self.on_data2d_changed)
        self.model.cmap_changed.connect(self.on_cmap_changed)

    def load_colormaps(self):
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

        self.main_view.cb_colormap.addItems(cmap_files)
        self.main_view.cb_colormap.setMaxVisibleItems(25)

        self.main_view.canvas.colormap = self.model.colormap

    def on_load(self):
        #open_directory = self.profile_settings['open_directory']
        #filename = str(QtGui.QFileDialog.getOpenFileName(directory=open_directory,
        #                                                 filter='*.dat'))

        filename = str(QtGui.QFileDialog.getOpenFileName(filter='*.dat'))

        if filename != '':
            self.model.load_data_file(filename)

    def on_parameters_changed(self):
        """ One of the parameters to plot has changed """
        self.model.select_parameters(*self.main_view.get_parameters())

    def on_cmap_chosen(self):
        """ A new colormap has been selected """
        self.model.set_colormap(self.main_view.get_cmap_name())

    def on_cmap_reset(self):
        """ The colormap settings are reset """
        min, max = self.model.data2d.get_z_limits()

        self.model.set_colormap_settings(min, max, 1.0)

    def on_cmap_edit_changed(self):
        """
        One of the min/max colormap text fields changed, so update colormap.
        """
        new_min = float(self.main_view.le_cmap_min.text())
        new_max = float(self.main_view.le_cmap_max.text())

        gamma = self.model.colormap.gamma
        self.model.set_colormap_settings(new_min, new_max, gamma)

    def on_cmap_slider_changed(self, value):
        """
        One of the colormap sliders moved, so update the colormap settings.
        """
        zmin, zmax = self.model.data2d.get_z_limits()

        min, max, gamma = self.model.colormap.get_settings()

        if self.main_view.s_cmap_min.isSliderDown():
            min = zmin + (zmax - zmin) * (value / 99.0)

        if self.main_view.s_cmap_gamma.isSliderDown():
            gamma = 10.0**(value / 100.0)

        if self.main_view.s_cmap_max.isSliderDown():
            max = zmin + (zmax - zmin) * (value / 99.0)

        self.model.set_colormap_settings(min, max, gamma)

    def on_data_file_changed(self):
        for cb in [self.main_view.cb_x,
                   self.main_view.cb_y,
                   self.main_view.cb_z]:
            cb.clear()
            cb.addItems([''] + self.model.data_file.ids)
            # set index

        # Temporary
        self.main_view.cb_x.setCurrentIndex(1)
        self.main_view.cb_y.setCurrentIndex(2)
        self.main_view.cb_z.setCurrentIndex(4)

    def on_data2d_changed(self):
        # Reset the colormap if required
        if self.main_view.get_reset_colormap():
            min, max = self.model.data2d.get_z_limits()

            self.model.set_colormap_settings(min, max, 1.0)

        self.main_view.canvas.set_data(self.model.data2d)

    def on_cmap_changed(self):
        """ The colormap settings changed, so update the UI """
        if self.model.data2d is None:
            return

        min, max, gamma = self.model.colormap.get_settings()

        zmin, zmax = self.model.data2d.get_z_limits()

        # Convert into slider positions (0 - 100)
        # Don't adjust the slider if it's currently being used
        if not self.main_view.s_cmap_min.isSliderDown():
            new_val = (min - zmin) / ((zmax - zmin) / 100)
            self.main_view.s_cmap_min.setValue(new_val)

        if not self.main_view.s_cmap_gamma.isSliderDown():
            new_val = np.log10(gamma) * 100.0
            self.main_view.s_cmap_gamma.setValue(new_val)

        if not self.main_view.s_cmap_max.isSliderDown():
            new_val = (max - zmin) / ((zmax - zmin) / 100)
            self.main_view.s_cmap_max.setValue(new_val)

        self.main_view.le_cmap_min.setText('%.2e' % min)
        self.main_view.le_cmap_max.setText('%.2e' % max)

        # Update the colormap and plot
        self.main_view.canvas.colormap = self.model.colormap
        self.main_view.canvas.update()


def main():
    """ Entry point for qtplot """
    app = QtGui.QApplication(sys.argv)

    if len(sys.argv) > 1:
        c = Controller(filename=sys.argv[1])
    else:
        c = Controller()

    sys.exit(app.exec_())
