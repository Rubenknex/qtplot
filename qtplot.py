import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import time

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.ticker import ScalarFormatter
from PyQt4 import QtGui, QtCore

from dat_file import DatFile, Data
from operations import Operations

"""
- Proper units on y-axis of linecut, now not properly formatted

- Integration into qtlab as real time plotting

- Use proper axis when plotting linecut and not averaging

- Warning message if NaN values are encountered

- Handle duplicate coordinate values correctly:
    Drop duplicates for the color plot
    Duplicates are ok for linecut plots

- Prevent data from sorting or add option
    Is this expected behavior? A colorplot is a top down view and measurement order is lost
    Should the linecut retain information about the order of the measurements?
"""

class FixedOrderFormatter(ScalarFormatter):
    """Format numbers using engineering notation."""
    def __init__(self, significance=0):
        ScalarFormatter.__init__(self, useOffset=None, useMathText=None)
        self.format = '%.' + str(significance) + 'f'

    def __call__(self, x, pos=None):
        if x == 0:
            return '0'

        exp = self.orderOfMagnitude

        return self.format % (x / (10 ** exp))

    def _set_orderOfMagnitude(self, range):
        exp = math.floor(math.log10(range))
        self.orderOfMagnitude = exp - (exp % 3)

class Window(QtGui.QMainWindow):
    """The main window of the qtplot application."""
    def __init__(self, lc_window, op_window, filename=None):
        QtGui.QMainWindow.__init__(self)

        self.filename = None
        
        self.linecut = lc_window
        self.operations = op_window

        self.fig, self.ax = plt.subplots()
        self.cb = None

        self.line = None
        self.linecut_type = None
        self.linecut_coord = None

        self.cmap_change = False

        self.dat_file = None
        self.data = None
        self.pcolor_data = None

        self.init_ui()

        if filename is not None:
            self.load_file(filename)
            self.update_ui()

    def init_ui(self):
        self.setWindowTitle('qtplot')

        self.main_widget = QtGui.QWidget(self)

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_motion)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        self.b_load = QtGui.QPushButton('Load DAT...')
        self.b_load.clicked.connect(self.on_load_dat)
        self.b_refresh = QtGui.QPushButton('Refresh')
        self.b_refresh.clicked.connect(self.on_refresh)
        self.b_swap_axes = QtGui.QPushButton('Swap axes', self)
        self.b_swap_axes.clicked.connect(self.on_swap_axes)
        self.b_swap = QtGui.QPushButton('Swap order', self)
        self.b_swap.clicked.connect(self.on_swap_order)
        self.b_linecut = QtGui.QPushButton('Linecut')
        self.b_linecut.clicked.connect(self.linecut.raise_)
        self.b_operations = QtGui.QPushButton('Operations')
        self.b_operations.clicked.connect(self.operations.raise_)

        lbl_x = QtGui.QLabel("X:", self)
        self.cb_x = QtGui.QComboBox(self)
        self.cb_x.activated.connect(self.on_data_change)
        lbl_order_x = QtGui.QLabel('X Order: ', self)
        self.cb_order_x = QtGui.QComboBox(self)

        lbl_y = QtGui.QLabel("Y:", self)
        self.cb_y = QtGui.QComboBox(self)
        self.cb_y.activated.connect(self.on_data_change)
        lbl_order_y = QtGui.QLabel('Y Order: ', self)
        self.cb_order_y = QtGui.QComboBox(self)

        lbl_d = QtGui.QLabel("Data:", self)
        self.cb_z = QtGui.QComboBox(self)
        self.cb_z.activated.connect(self.on_data_change)

        self.le_min = QtGui.QLineEdit(self)
        self.le_min.setMaximumWidth(100)
        self.le_min.returnPressed.connect(self.on_cmap_changed)

        self.s_gamma = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.s_gamma.setMinimum(0)
        self.s_gamma.setMaximum(100)
        self.s_gamma.setValue(50)
        self.s_gamma.valueChanged.connect(self.on_cmap_changed)

        self.le_max = QtGui.QLineEdit(self)
        self.le_max.setMaximumWidth(100)
        self.le_max.returnPressed.connect(self.on_cmap_changed)

        self.b_copy_colorplot = QtGui.QPushButton('Copy colorplot to clipboard', self)
        self.b_copy_colorplot.clicked.connect(self.on_copy_colorplot)
        self.b_copy_linecut = QtGui.QPushButton('Copy linecut to clipboard', self)
        self.b_copy_linecut.clicked.connect(self.on_copy_linecut)

        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.b_load)
        hbox.addWidget(self.b_refresh)
        hbox.addWidget(self.b_swap_axes)
        hbox.addWidget(self.b_swap)
        hbox.addWidget(self.b_linecut)
        hbox.addWidget(self.b_operations)

        grid = QtGui.QGridLayout()
        grid.addWidget(lbl_x, 1, 1)
        grid.addWidget(self.cb_x, 1, 2)
        grid.addWidget(lbl_order_x, 1, 3)
        grid.addWidget(self.cb_order_x, 1, 4)

        grid.addWidget(lbl_y, 2, 1)
        grid.addWidget(self.cb_y, 2, 2)
        grid.addWidget(lbl_order_y, 2, 3)
        grid.addWidget(self.cb_order_y, 2, 4)

        grid.addWidget(lbl_d, 3, 1)
        grid.addWidget(self.cb_z, 3, 2)

        hbox_gamma = QtGui.QHBoxLayout()
        hbox_gamma.addWidget(self.le_min)
        hbox_gamma.addWidget(self.s_gamma)
        hbox_gamma.addWidget(self.le_max)

        hbox4 = QtGui.QHBoxLayout()
        hbox4.addWidget(self.b_copy_colorplot)
        hbox4.addWidget(self.b_copy_linecut)

        vbox = QtGui.QVBoxLayout(self.main_widget)
        vbox.addWidget(self.toolbar)
        vbox.addWidget(self.canvas)
        vbox.addLayout(hbox)
        vbox.addLayout(grid)
        vbox.addLayout(hbox_gamma)
        vbox.addLayout(hbox4)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.move(100, 100)

    def update_ui(self):
        self.setWindowTitle(self.name)

        self.cb_x.clear()
        self.cb_x.addItems(self.dat_file.columns)
        self.cb_x.setCurrentIndex(0)

        self.cb_order_x.clear()
        self.cb_order_x.addItems(self.dat_file.columns)
        self.cb_order_x.setCurrentIndex(0)

        self.cb_y.clear()
        self.cb_y.addItems(self.dat_file.columns)
        self.cb_y.setCurrentIndex(1)

        self.cb_order_y.clear()
        self.cb_order_y.addItems(self.dat_file.columns)
        self.cb_order_y.setCurrentIndex(1)

        self.cb_z.clear()
        self.cb_z.addItems(self.dat_file.columns)
        self.cb_z.setCurrentIndex(3)

    def load_file(self, filename):
        self.dat_file = DatFile(filename)

        if filename != self.filename:
            path, self.name = os.path.split(filename)
            self.filename = filename

            self.line = None
            self.linecut_type = None
            self.linecut_coord = None

            self.update_ui()

        self.data = None
        self.pcolor_data = None

        self.on_data_change()

    def on_load_dat(self, event):
        filename = str(QtGui.QFileDialog.getOpenFileName(filter='*.dat'))

        if filename != "":
            self.load_file(filename)

    def on_refresh(self, event):
        if self.filename:
            self.load_file(self.filename)

    def on_swap_axes(self, event):
        x, y = self.cb_x.currentIndex(), self.cb_y.currentIndex()
        self.cb_x.setCurrentIndex(y)
        self.cb_y.setCurrentIndex(x)

        if len(self.ax.lines) > 0:
            self.ax.lines.pop(0)

        if self.linecut_type == 'horizontal':
            self.linecut_type = 'vertical'
        else:
            self.linecut_type = 'horizontal'

        self.on_swap_order(event)

    def on_swap_order(self, event):
        x, y = self.cb_order_x.currentIndex(), self.cb_order_y.currentIndex()
        self.cb_order_x.setCurrentIndex(y)
        self.cb_order_y.setCurrentIndex(x)

        self.on_data_change()

    def on_cmap_changed(self):
        self.cmap_change = True
        self.plot_2d_data()

    def on_mouse_click(self, event):
        if not event.inaxes or self.dat_file == None:
            return

        if event.button == 1:
            # Get the row closest to the mouse Y
            self.linecut_coord = self.data.get_closest_y(event.ydata)
            self.linecut_type = 'horizontal'
        elif event.button == 2:
            # Get the column closest to the mouse X
            self.linecut_coord = self.data.get_closest_x(event.xdata)
            self.linecut_type = 'vertical'

        self.plot_linecut()

    def on_mouse_motion(self, event):
        if event.button != None:
            self.on_mouse_click(event)

    def on_data_change(self):
        if self.dat_file is not None:
            self.generate_data()
            self.plot_2d_data()
            self.plot_linecut()

    def on_copy_colorplot(self):
        path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(path, 'test.png')
        self.fig.savefig(path)

        img = QtGui.QImage(path)
        QtGui.QApplication.clipboard().setImage(img)

    def on_copy_linecut(self):
        path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(path, 'test.png')
        self.linecut.fig.savefig(path)

        img = QtGui.QImage(path)
        QtGui.QApplication.clipboard().setImage(img)

    def get_axis_names(self):
        x_name = str(self.cb_x.currentText())
        y_name = str(self.cb_y.currentText())
        data_name = str(self.cb_z.currentText())
        order_x = str(self.cb_order_x.currentText())
        order_y = str(self.cb_order_y.currentText())

        return x_name, y_name, data_name, order_x, order_y

    def generate_data(self):
        x_name, y_name, data_name, order_x, order_y = self.get_axis_names()

        self.data = self.dat_file.get_data(x_name, y_name, data_name, order_x, order_y)
        self.data = self.operations.apply_operations(self.data)
        self.pcolor_data = self.data.get_pcolor()

    def plot_2d_data(self):
        if self.dat_file is None:
            return

        cmap = mpl.cm.get_cmap('seismic')
        value = (self.s_gamma.value() / 100.0)
        cmap.set_gamma(math.exp(value * 5) / 10.0)

        self.ax.clear()

        x_name, y_name, data_name, order_x, order_y = self.get_axis_names()

        if self.pcolor_data == None:
            self.generate_data()
        
        # Set the axis range to increase upwards or to the left, and reverse if necessary
        self.ax.set_xlim(sorted(self.ax.get_xlim()))
        self.ax.set_ylim(sorted(self.ax.get_ylim()))

        x_flip, y_flip = self.data.is_flipped()
        if x_flip:
            self.ax.invert_xaxis()
        if y_flip:
            self.ax.invert_yaxis()

        self.quadmesh = self.ax.pcolormesh(*self.pcolor_data, cmap=cmap)

        if self.cmap_change:
            self.quadmesh.set_clim(vmin=float(self.le_min.text()), vmax=float(self.le_max.text()))
        else:
            cm_min, cm_max = self.quadmesh.get_clim()
            self.le_min.setText('%.2e' % cm_min)
            self.le_max.setText('%.2e' % cm_max)

            self.s_gamma.setValue(50)
        
        self.ax.axis('tight')

        # Create a colorbar, if there is already one draw it in the existing place
        if self.cb:
            self.cb.ax.clear()
            self.cb = self.fig.colorbar(self.quadmesh, cax=self.cb.ax)
        else:
            self.cb = self.fig.colorbar(self.quadmesh)

        self.cb.set_label(data_name)
        self.cb.formatter = FixedOrderFormatter(1)
        self.cb.update_ticks()

        # Set the various labels
        self.ax.set_title(self.name)
        self.ax.set_xlabel(x_name)
        self.ax.set_ylabel(y_name)
        self.ax.xaxis.set_major_formatter(FixedOrderFormatter())
        self.ax.yaxis.set_major_formatter(FixedOrderFormatter())
        self.ax.set_aspect('auto')

        self.fig.tight_layout()
        
        self.canvas.draw()

        self.background = self.canvas.copy_from_bbox(self.ax.bbox)

        self.cmap_change = False

    def plot_linecut(self):
        if self.dat_file == None or self.linecut_type == None:
            return

        lc = self.linecut
        x_name, y_name, data_name, order_x, order_y = self.get_axis_names()

        if len(self.ax.lines) == 0:
            self.line = self.ax.axvline(color='red')

        if len(lc.ax.lines) > 0:
            lc.ax.lines.pop(0)

        if self.linecut_type == 'horizontal':
            self.line.set_transform(self.ax.get_yaxis_transform())
            self.line.set_xdata([0, 1])
            self.line.set_ydata([self.linecut_coord, self.linecut_coord])

            x, y = self.data.get_row_at(self.linecut_coord)
            lc.ax.plot(x, y, color='red', linewidth=0.5)
            lc.ax.set_xlabel(x_name)
        elif self.linecut_type == 'vertical':
            self.line.set_transform(self.ax.get_xaxis_transform())
            self.line.set_xdata([self.linecut_coord, self.linecut_coord])
            self.line.set_ydata([0, 1])

            x, y = self.data.get_column_at(self.linecut_coord)
            lc.ax.plot(x, y, color='red', linewidth=0.5)
            lc.ax.set_xlabel(y_name)

        lc.ax.set_title(self.name)
        lc.ax.set_ylabel(data_name)
        lc.ax.xaxis.set_major_formatter(FixedOrderFormatter())
        lc.ax.yaxis.set_major_formatter(FixedOrderFormatter())

        # Make the figure fit the data
        lc.ax.relim()
        lc.ax.autoscale_view()
        lc.fig.tight_layout()

        # Redraw both plots to update them
        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)

        lc.canvas.draw()

    def resizeEvent(self, event):
        if len(self.ax.lines) > 0:
            self.ax.lines.pop(0)

        # Very slow, maybe search for faster way
        # http://stackoverflow.com/questions/13552345/how-to-disable-multiple-auto-redrawing-at-resizing-widgets-in-pyqt
        self.plot_2d_data()

    def closeEvent(self, event):
        self.linecut.close()
        self.operations.close()

class Linecut(QtGui.QDialog):
    def __init__(self, parent=None):
        super(Linecut, self).__init__(parent)

        self.fig, self.ax = plt.subplots()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Linecut")

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.move(800, 100)

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    linecut = Linecut()
    operations = Operations()
    main = Window(linecut, operations)
    
    if len(sys.argv) > 1:
        main = Window(linecut, operations, filename=sys.argv[1])

    linecut.main = main
    operations.main = main

    linecut.show()
    operations.show()
    main.show()
    
    sys.exit(app.exec_())