import ConfigParser
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
from scipy import interpolate, spatial, io
from scipy.interpolate import griddata
from scipy.spatial import qhull, delaunay_plot_2d

from data import DatFile, Data
from export import ExportWidget
from linecut import Linecut, FixedOrderFormatter
from operations import Operations
from settings import Settings
from canvas import Canvas

class Window(QtGui.QMainWindow):
    """The main window of the qtplot application."""
    def __init__(self, lc_window, op_window, filename=None):
        QtGui.QMainWindow.__init__(self)

        self.filename = None
        
        self.linecut = lc_window
        self.operations = op_window
        self.settings = Settings()

        self.dat_file = None
        self.data = None

        self.init_ui()

        if filename is not None:
            self.load_file(filename)

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
        
        self.b_load = QtGui.QPushButton('Load DAT...')
        self.b_load.clicked.connect(self.on_load_dat)
        hbox.addWidget(self.b_load)

        self.b_refresh = QtGui.QPushButton('Refresh')
        self.b_refresh.clicked.connect(self.on_refresh)
        hbox.addWidget(self.b_refresh)

        self.b_swap_axes = QtGui.QPushButton('Swap axes', self)
        self.b_swap_axes.clicked.connect(self.on_swap_axes)
        hbox.addWidget(self.b_swap_axes)

        self.b_swap = QtGui.QPushButton('Swap order', self)
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
        r_hbox.addWidget(self.le_r)

        self.b_ok = QtGui.QPushButton('Ok', self)
        self.b_ok.clicked.connect(self.on_sub_series_r)
        self.b_ok.setMaximumWidth(50)
        r_hbox.addWidget(self.b_ok)

        # Selecting columns and orders
        grid = QtGui.QGridLayout()

        lbl_x = QtGui.QLabel("X:", self)
        grid.addWidget(lbl_x, 1, 1)

        self.cb_x = QtGui.QComboBox(self)
        self.cb_x.activated.connect(self.on_data_change)
        grid.addWidget(self.cb_x, 1, 2)

        lbl_order_x = QtGui.QLabel('X Order: ', self)
        grid.addWidget(lbl_order_x, 1, 3)

        self.cb_order_x = QtGui.QComboBox(self)
        self.cb_order_x.activated.connect(self.on_data_change)
        grid.addWidget(self.cb_order_x, 1, 4)

        lbl_y = QtGui.QLabel("Y:", self)
        grid.addWidget(lbl_y, 2, 1)

        self.cb_y = QtGui.QComboBox(self)
        self.cb_y.activated.connect(self.on_data_change)
        grid.addWidget(self.cb_y, 2, 2)

        lbl_order_y = QtGui.QLabel('Y Order: ', self)
        grid.addWidget(lbl_order_y, 2, 3)

        self.cb_order_y = QtGui.QComboBox(self)
        self.cb_order_y.activated.connect(self.on_data_change)
        grid.addWidget(self.cb_order_y, 2, 4)

        lbl_d = QtGui.QLabel("Data:", self)
        grid.addWidget(lbl_d, 3, 1)

        self.cb_z = QtGui.QComboBox(self)
        self.cb_z.activated.connect(self.on_data_change)
        grid.addWidget(self.cb_z, 3, 2)

        self.cb_save_default = QtGui.QCheckBox('Save as default columns')
        grid.addWidget(self.cb_save_default, 3, 3)

        # Colormap
        hbox_gamma = QtGui.QHBoxLayout()
        
        self.cb_reset_cmap = QtGui.QCheckBox('Reset on plot')
        self.cb_reset_cmap.setCheckState(QtCore.Qt.Checked)
        hbox_gamma.addWidget(self.cb_reset_cmap)

        self.le_min = QtGui.QLineEdit(self)
        self.le_min.setMaximumWidth(100)
        self.le_min.returnPressed.connect(self.on_min_max_entered)
        hbox_gamma.addWidget(self.le_min)

        self.s_min = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.s_min.sliderMoved.connect(self.on_min_changed)
        hbox_gamma.addWidget(self.s_min)

        self.s_gamma = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.s_gamma.setMinimum(-100)
        self.s_gamma.setMaximum(100)
        self.s_gamma.setValue(0)
        self.s_gamma.valueChanged.connect(self.on_gamma_changed)
        hbox_gamma.addWidget(self.s_gamma)

        self.s_max = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.s_max.setValue(self.s_max.maximum())
        self.s_max.sliderMoved.connect(self.on_max_changed)
        hbox_gamma.addWidget(self.s_max)

        self.le_max = QtGui.QLineEdit(self)
        self.le_max.setMaximumWidth(100)
        self.le_max.returnPressed.connect(self.on_min_max_entered)
        hbox_gamma.addWidget(self.le_max)

        self.b_reset = QtGui.QPushButton('Reset')
        self.b_reset.clicked.connect(self.on_cm_reset)
        hbox_gamma.addWidget(self.b_reset)

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
        vbox.addLayout(grid)
        vbox.addLayout(hbox_gamma)
        vbox.addLayout(hbox4)

        self.status_bar = QtGui.QStatusBar()
        self.setStatusBar(self.status_bar)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.resize(600, 700)
        self.move(100, 100)

    def update_ui(self, reset=True):
        self.setWindowTitle(self.name)

        columns = self.dat_file.df.columns.values

        self.cb_v.clear()
        self.cb_v.addItems(columns)

        self.cb_i.clear()
        self.cb_i.addItems(columns)

        i = self.cb_x.currentIndex()
        self.cb_x.clear()
        self.cb_x.addItems(columns)
        self.cb_x.setCurrentIndex(i)

        i = self.cb_order_x.currentIndex()
        self.cb_order_x.clear()
        self.cb_order_x.addItems(columns)
        self.cb_order_x.setCurrentIndex(i)

        i = self.cb_y.currentIndex()
        self.cb_y.clear()
        self.cb_y.addItems(columns)
        self.cb_y.setCurrentIndex(i)

        i = self.cb_order_y.currentIndex()
        self.cb_order_y.clear()
        self.cb_order_y.addItems(columns)
        self.cb_order_y.setCurrentIndex(i)

        i = self.cb_z.currentIndex()
        self.cb_z.clear()
        self.cb_z.addItems(columns)
        self.cb_z.setCurrentIndex(i)

        if reset:
            combo_boxes = [self.cb_x, self.cb_order_x, self.cb_y, self.cb_order_y, self.cb_z]
            names = ['X', 'X Order', 'Y', 'Y Order', 'Data']
            default_indices = [0, 0, 1, 1, 3]

            if os.path.isfile('qtplot.config'):
                config = ConfigParser.RawConfigParser()
                path = os.path.dirname(os.path.realpath(__file__))
                config.read(os.path.join(path, 'qtplot.config'))

                if config.has_section('Settings'):
                    indices = [cb.findText(config.get('Settings', names[i])) for i,cb in enumerate(combo_boxes)]

                    for i, index in enumerate(indices):
                        if index != -1:
                            combo_boxes[i].setCurrentIndex(index)
                        else:
                            combo_boxes[i].setCurrentIndex(default_indices[i])
            else:
                for i, index in enumerate(default_indices):
                    combo_boxes[i].setCurrentIndex(index)

    def on_load_dat(self, event):
        filename = str(QtGui.QFileDialog.getOpenFileName(filter='*.dat'))

        if filename != "":
            self.load_file(filename)

    def load_file(self, filename):
        self.dat_file = DatFile(filename)
        self.settings.load_file(filename)

        if filename != self.filename:
            path, self.name = os.path.split(filename)
            self.filename = filename

            self.update_ui()

        self.on_data_change()  

    def on_refresh(self, event):
        if self.filename:
            self.load_file(self.filename)

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
    
    def on_data_change(self):
        x_name, y_name, data_name, order_x, order_y = self.get_axis_names()

        self.export_widget.set_info(self.name, x_name, y_name, data_name)

        t0 = time.clock()
        self.data = self.dat_file.get_data(x_name, y_name, data_name, order_x, order_y)
        t1 = time.clock()
        #print 'get_data:', t1-t0
        self.data = self.operations.apply_operations(self.data)
        #print 'operations:', time.clock()-t1

        if self.cb_reset_cmap.checkState() == QtCore.Qt.Checked:
            self.on_min_changed(0)
            self.s_gamma.setValue(0)
            self.on_max_changed(100)

        self.canvas.set_data(self.data)
        self.canvas.draw_linecut(None, old_position=True)
        self.canvas.update()

        """
        if self.data.values.mask.any():
            self.status_bar.showMessage("Warning: Data contains NaN values")
        else:
            self.status_bar.showMessage("")
        """

    def on_sub_series_r(self, event):
        if self.dat_file == None:
            return

        V, I = str(self.cb_v.currentText()), str(self.cb_i.currentText())
        R = float(self.le_r.text())

        self.dat_file.df['SUB SERIES R'] = self.dat_file.df[V] - self.dat_file.df[I] * R

        self.update_ui(reset=False)

    def on_min_max_entered(self):
        if self.data != None:
            zmin, zmax = np.nanmin(self.data.values), np.nanmax(self.data.values)

            newmin, newmax = float(self.le_min.text()), float(self.le_max.text())

            # Convert the entered bounds into slider positions (0 - 100)
            self.s_min.setValue((newmin - zmin) / ((zmax - zmin) / 100))
            self.s_max.setValue((newmax - zmin) / ((zmax - zmin) / 100))

            cm = self.canvas.colormap
            cm.min, cm.max = newmin, newmax

            self.canvas.update()

    def on_min_changed(self, value):
        if self.data != None:
            min, max = np.nanmin(self.data.values), np.nanmax(self.data.values)

            newmin = min + ((max - min) / 100.0) * value
            self.le_min.setText('%.2e' % newmin)

            self.canvas.colormap.min = newmin
            self.canvas.update()

    def on_gamma_changed(self, value):
        if self.data != None:
            gamma = 10.0**(value / 100.0)

            self.canvas.colormap.gamma = gamma
            self.canvas.update()

    def on_max_changed(self, value):
        if self.data != None:
            min, max = np.nanmin(self.data.values), np.nanmax(self.data.values)

            newmax = min + ((max - min) / 100.0) * value
            self.le_max.setText('%.2e' % newmax)

            self.canvas.colormap.max = newmax
            self.canvas.update()

    def on_cm_reset(self):
        if self.data != None:
            zmin, zmax = np.nanmin(self.data.values), np.nanmax(self.data.values)

            self.s_min.setValue(0)
            self.on_min_changed(0)
            self.s_gamma.setValue(0)
            self.s_max.setValue(100)
            self.on_max_changed(100)

    def on_save_matrix(self):
        path = os.path.dirname(os.path.realpath(__file__))
        filename = QtGui.QFileDialog.getSaveFileName(self, 'Save file', path, 'NumPy matrix format (*.npy);;MATLAB matrix format (*.mat)')
        filename = str(filename)

        if filename != '' and self.dat_file != None:
            base = os.path.basename(filename)
            name, ext = os.path.splitext(base)

            mat = np.dstack((self.data.x_coords, self.data.y_coords, self.data.values))

            if ext == '.npy':
                np.save(filename, mat)
                #mat.dump(filename)
            elif ext == '.mat':
                io.savemat(filename, {'data':mat})

    def get_axis_names(self):
        x_name = str(self.cb_x.currentText())
        y_name = str(self.cb_y.currentText())
        data_name = str(self.cb_z.currentText())
        order_x = str(self.cb_order_x.currentText())
        order_y = str(self.cb_order_y.currentText())

        return x_name, y_name, data_name, order_x, order_y

    def closeEvent(self, event):
        self.linecut.close()
        self.operations.close()
        self.settings.close()

        if self.cb_save_default.checkState() == QtCore.Qt.Checked:
            config = ConfigParser.RawConfigParser()
            config.add_section('Settings')
            x_name, y_name, data_name, order_x, order_y = self.get_axis_names()
            config.set('Settings', 'X', x_name)
            config.set('Settings', 'X Order', order_x)
            config.set('Settings', 'Y', y_name)
            config.set('Settings', 'Y Order', order_y)
            config.set('Settings', 'Data', data_name)

            path = os.path.dirname(os.path.realpath(__file__))
            with open(os.path.join(path, 'qtplot.config'), 'wb') as config_file:
                config.write(config_file)

if __name__ == '__main__':
    mpl.rcParams['mathtext.fontset'] = 'custom'
    mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

    app = QtGui.QApplication(sys.argv)

    linecut = Linecut()
    operations = Operations()
    
    if len(sys.argv) > 1:
        main = Window(linecut, operations, filename=sys.argv[1])
    else:
        main = Window(linecut, operations)

    linecut.main = main
    operations.main = main

    linecut.show()
    operations.show()
    main.show()
    
    sys.exit(app.exec_())