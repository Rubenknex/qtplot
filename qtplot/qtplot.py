import os
import sys

import numpy as np
from PyQt4 import QtCore, QtGui, uic

from .canvas import Canvas
from .model import Model
from .linetrace import Linetrace
from .operations import Operations
from .settings import Settings


class QTPlot(QtGui.QMainWindow):
    """
    This class contains all the logic behind the user interface and
    serves as the connection between the views (user interface) and
    the model (data).
    """
    def __init__(self):
        super(QTPlot, self).__init__()

        directory = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(directory, 'ui/main.ui')
        uic.loadUi(path, self)

        self.canvas = Canvas()
        self.canvas_layout.addWidget(self.canvas.native)

        self.model = Model()

        self.load_colormaps()

        center = QtGui.QApplication.desktop().screen().rect().center()
        self.move(center.x() - 550, center.y() - 400)

        self.show()

        self.linetrace = Linetrace(self, self.model)
        self.operations = Operations(self, self.model)
        self.settings = Settings(self, self.model)

        self.cb_parameters = [self.cb_x, self.cb_y, self.cb_z]
        self.sliders = [self.s_cmap_min, self.s_cmap_gamma, self.s_cmap_max]

        self.bind()

    def bind(self):
        """
        Set up the connections listening for user interface events
        from the view.
        """
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

        self.cb_colormap.activated.connect(self.on_cmap_chosen)
        self.b_reset_cmap.clicked.connect(self.on_cmap_reset)

        self.le_cmap_min.returnPressed.connect(self.on_cmap_edit_changed)
        self.le_cmap_max.returnPressed.connect(self.on_cmap_edit_changed)

        for s in self.sliders:
            s.sliderMoved.connect(self.on_cmap_slider_changed)

        # Canvas events
        self.canvas.events.mouse_press.connect(self.on_canvas_press)
        self.canvas.events.mouse_move.connect(self.on_canvas_move)

        self.model.data_file_changed.connect(self.on_data_file_changed)
        self.model.data2d_changed.connect(self.on_data2d_changed)
        self.model.cmap_changed.connect(self.on_cmap_changed)

    def get_parameters(self):
        return [str(cb.currentText()) for cb in self.cb_parameters]

    def get_reset_colormap(self):
        return self.cb_reset_on_plot.checkState() == QtCore.Qt.Checked

    def get_cmap_name(self):
        return str(self.cb_colormap.currentText())

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

        self.cb_colormap.addItems(cmap_files)
        self.cb_colormap.setMaxVisibleItems(25)

        self.canvas.colormap = self.model.colormap

    def on_load(self):
        #open_directory = self.profile_settings['open_directory']
        #filename = str(QtGui.QFileDialog.getOpenFileName(directory=open_directory,
        #                                                 filter='*.dat'))

        filename = str(QtGui.QFileDialog.getOpenFileName(filter='*.dat'))

        if filename != '':
            self.model.load_data_file(filename)

    def on_save(self):
        #save_directory = self.profile_settings['save_directory']

        filters = ('QTLab data format (*.dat);;'
                   'NumPy binary matrix format (*.npy);;'
                   'MATLAB matrix format (*.mat)')

        filename = QtGui.QFileDialog.getSaveFileName(caption='Save file',
                                                     #directory=save_directory,
                                                     filter=filters)
        filename = str(filename)

        if filename != '' and self.model.data2d is not None:
            base = os.path.basename(filename)
            name, ext = os.path.splitext(base)

            self.data.save(filename)

    def on_parameters_changed(self):
        """ One of the parameters to plot has changed """
        self.model.select_parameters(*self.get_parameters())

    def on_cmap_chosen(self):
        """ A new colormap has been selected """
        self.model.set_colormap(self.get_cmap_name())

    def on_cmap_reset(self):
        """ The colormap settings are reset """
        min, max = self.model.data2d.get_z_limits()

        self.model.set_colormap_settings(min, max, 1.0)

    def on_cmap_edit_changed(self):
        """
        One of the min/max colormap text fields changed, so update colormap.
        """
        new_min = float(self.le_cmap_min.text())
        new_max = float(self.le_cmap_max.text())

        gamma = self.model.colormap.gamma
        self.model.set_colormap_settings(new_min, new_max, gamma)

    def on_cmap_slider_changed(self, value):
        """
        One of the colormap sliders moved, so update the colormap settings.
        """
        zmin, zmax = self.model.data2d.get_z_limits()

        min, max, gamma = self.model.colormap.get_settings()

        if self.s_cmap_min.isSliderDown():
            min = zmin + (zmax - zmin) * (value / 99.0)

        if self.s_cmap_gamma.isSliderDown():
            gamma = 10.0**(value / 100.0)

        if self.s_cmap_max.isSliderDown():
            max = zmin + (zmax - zmin) * (value / 99.0)

        self.model.set_colormap_settings(min, max, gamma)

    def on_canvas_press(self, event):
        x, y = self.canvas.screen_to_data_coords(tuple(event.pos))

        type = {1: 'horizontal', 2: 'arbitrary', 3: 'vertical'}[event.button]

        self.model.take_linetrace(x, y, type)

    def on_canvas_move(self, event):
        if len(event.buttons) > 0:
            self.on_canvas_press(event)

        # From here on handlers of changes in the model
    def on_data_file_changed(self):
        for cb in self.cb_parameters:
            cb.clear()
            cb.addItems([''] + self.model.data_file.ids)
            # set index

        # Temporary
        self.cb_x.setCurrentIndex(1)
        self.cb_y.setCurrentIndex(2)
        self.cb_z.setCurrentIndex(4)

    def on_data2d_changed(self):
        #print('data2d_changed')
        # Reset the colormap if required
        if self.get_reset_colormap():
            min, max = self.model.data2d.get_z_limits()

            self.model.set_colormap_settings(min, max, 1.0)

        # Update the UI if necessary
        idx = self.cb_x.findText(self.model.x)
        self.cb_x.setCurrentIndex(idx)
        idx = self.cb_y.findText(self.model.y)
        self.cb_y.setCurrentIndex(idx)

        self.canvas.set_data(self.model.data2d)

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


def main():
    """ Entry point for qtplot """
    app = QtGui.QApplication(sys.argv)

    if len(sys.argv) > 1:
        c = QTPlot(filename=sys.argv[1])
    else:
        c = QTPlot()

    sys.exit(app.exec_())
