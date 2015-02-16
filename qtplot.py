import os
import sys
import time

from PyQt4 import QtGui, QtCore

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.image import NonUniformImage

import numpy as np
import pandas as pd

from pptx import Presentation
from pptx.util import Inches

from dat_file import DatFile
from operations import Operations

class Window(QtGui.QDialog):
    def __init__(self, lc_window, op_window, filename, parent=None):
        super(Window, self).__init__(parent)
        
        self.linecut = lc_window
        self.operations = op_window
        self.data_file = DatFile(filename)
        self.filename = filename
        path, self.name = os.path.split(self.data_file.filename)

        self.fig, self.ax = plt.subplots()
        self.cb = None

        self.linecut_type = None
        self.linecut_coord = None

        self.ppt = None
        self.slide = None

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.filename)

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_motion)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        self.x_lbl = QtGui.QLabel("X", self)
        self.x_combo = QtGui.QComboBox(self)
        self.x_combo.addItems(self.data_file.columns)
        self.x_combo.setCurrentIndex(5)

        self.y_lbl = QtGui.QLabel("Y", self)
        self.y_combo = QtGui.QComboBox(self)
        self.y_combo.addItems(self.data_file.columns)
        self.y_combo.setCurrentIndex(8)

        self.d_lbl = QtGui.QLabel("Data", self)
        self.d_combo = QtGui.QComboBox(self)
        self.d_combo.addItems(self.data_file.columns)
        self.d_combo.setCurrentIndex(3)

        self.button = QtGui.QPushButton('Plot')
        self.button.clicked.connect(self.plot_2d_data)

        self.le_lbl = QtGui.QLabel("PPT File", self)
        self.ppt_browse = QtGui.QPushButton("Browse", self)
        self.ppt_browse.clicked.connect(self.browse_ppt)
        self.le_ppt = QtGui.QLineEdit("data.pptx", self)

        self.b_ppt1 = QtGui.QPushButton('Add 2D Data to PPT')
        self.b_ppt1.clicked.connect(self.add_2d_data)

        self.b_ppt2 = QtGui.QPushButton('Add Linecut to PPT')
        self.b_ppt2.clicked.connect(self.add_linecut)

        hbox1 = QtGui.QHBoxLayout()
        hbox1.addWidget(self.x_lbl)
        hbox1.addWidget(self.x_combo)

        hbox2 = QtGui.QHBoxLayout()
        hbox2.addWidget(self.y_lbl)
        hbox2.addWidget(self.y_combo)

        hbox3 = QtGui.QHBoxLayout()
        hbox3.addWidget(self.d_lbl)
        hbox3.addWidget(self.d_combo)

        hbox4 = QtGui.QHBoxLayout()
        hbox4.addWidget(self.le_lbl)
        hbox4.addWidget(self.ppt_browse)
        hbox4.addWidget(self.le_ppt)

        hbox5 = QtGui.QHBoxLayout()
        hbox5.addWidget(self.b_ppt1)
        hbox5.addWidget(self.b_ppt2)

        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.toolbar)
        vbox.addWidget(self.canvas)
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox3)
        vbox.addWidget(self.button)
        vbox.addLayout(hbox4)
        vbox.addLayout(hbox5)

        self.setLayout(vbox)

    def plot_2d_data(self):
        data = self.data_file.df.copy()
        columns = self.data_file.columns

        # Get the column names to use for the x, y and data
        self.x_lbl = str(self.x_combo.currentText())
        self.y_lbl = str(self.y_combo.currentText())
        self.data_lbl = str(self.d_combo.currentText())

        # Average the measurement columns which are related to the DAC values
        for col in columns:
            if self.x_lbl == col or self.y_lbl == col:
                if col in columns[3:7]:
                    data[col] = data.groupby(columns[1])[col].transform(np.average)

                if col in columns[7:11]:
                    data[col] = data.groupby(columns[0])[col].transform(np.average)

        # Pivot the data into an x and y axis, and values
        data = data.pivot(self.y_lbl, self.x_lbl, self.data_lbl)

        data = self.operations.perform_operation(data)

        # Clear the figure
        self.ax.clear()

        x = np.array(data.columns)
        y = np.array(data.index)

        xc = x[:-1] + np.diff(x) / 2.0
        yc = y[:-1] + np.diff(y) / 2.0

        xc = np.append(xc[0] - (x[1] - x[0]), xc)
        xc = np.append(xc, xc[-1] + (x[-1] - x[-2]))

        yc = np.append(yc[0] - (x[1] - x[0]), yc)
        yc = np.append(yc, yc[-1] + (x[-1] - x[-2]))

        masked = np.ma.masked_where(np.isnan(data.values), data.values)
        quadmesh = self.ax.pcolormesh(xc, yc, masked, cmap='seismic')
        
        self.ax.axis('tight')

        # Create a colorbar, if there is already one draw it in the existing place
        if self.cb:
            self.cb.ax.clear()
            self.cb = self.fig.colorbar(quadmesh, cax=self.cb.ax)
        else:
            self.cb = self.fig.colorbar(quadmesh)

        self.cb.set_label(self.data_lbl)

        # Set the various labels
        self.ax.set_title(self.name)
        self.ax.set_xlabel(self.x_lbl)
        self.ax.set_ylabel(self.y_lbl)
        self.ax.ticklabel_format(style='sci', scilimits=(-3, 3))
        self.ax.set_aspect('auto')

        self.fig.tight_layout()
        
        self.canvas.draw()

        del data

    def on_mouse_motion(self, event):
        if event.button != None:
            self.on_mouse_click(event)

    def on_mouse_click(self, event):
        # Don't do anything if we are not within the axes of the figure
        if not event.inaxes:
            return

        lc = self.linecut
        
        # If there is already a line in the figure, remove it for the new one
        if len(self.ax.lines) > 0:
            self.ax.lines.pop(0)

        if len(lc.ax.lines) > 0:
            lc.ax.lines.pop(0)

        if event.button == 1:
            # Get the row closest to the mouse Y
            self.linecut_coord = min(self.piv.index, key=lambda x:abs(x - event.ydata))
            self.linecut_type = 'horizontal'

            # Draw a horizontal line
            self.ax.axhline(y=self.linecut_coord, color='red')

            lc.ax.plot(self.piv.columns, self.piv.loc[self.linecut_coord], color='red')
            lc.ax.set_xlabel(self.x_lbl)
        elif event.button == 2:
            # Get the column closest to the mouse X
            self.linecut_coord = min(self.piv.columns, key=lambda x:abs(x - event.xdata))
            self.linecut_type = 'vertical'

            # Draw a vertical line
            self.ax.axvline(x=self.linecut_coord, color='red')

            lc.ax.plot(self.piv.index, self.piv[self.linecut_coord], color='red')
            lc.ax.set_xlabel(self.y_lbl)

        lc.ax.set_title(self.name)
        lc.ax.set_ylabel(self.data_lbl)
        lc.ax.ticklabel_format(style='sci', scilimits=(-3, 3))

        # Make the figure fit the data
        lc.ax.relim()
        lc.ax.autoscale_view()
        lc.fig.tight_layout()

        # Redraw both plots to update them
        self.canvas.draw()
        lc.canvas.draw()

    def add_2d_data(self, event):
        if self.ppt is None:
            self.ppt = Presentation(str(self.le_ppt.text()))

            title_slide_layout = self.ppt.slide_layouts[6]
            self.slide = self.ppt.slides.add_slide(title_slide_layout)

        self.fig.savefig("test.png")
        self.slide.shapes.add_picture("test.png", Inches(1), Inches(1))

    def add_linecut(self, event):
        if self.ppt is None:
            self.ppt = Presentation(str(self.le_ppt.text()))

            title_slide_layout = self.ppt.slide_layouts[6]
            self.slide = self.ppt.slides.add_slide(title_slide_layout)

        self.linecut.fig.savefig("test.png")
        self.slide.shapes.add_picture("test.png", Inches(1), Inches(1))

    def browse_ppt(self, event):
        filename = QtGui.QFileDialog.getOpenFileName(self)
        self.le_ppt.setText(filename)

    def closeEvent(self, event):
        if self.ppt:
            try:
                self.ppt.save(str(self.le_ppt.text()))
            except IOError as e:
                path, filename = os.path.split(str(self.le_ppt.text()))
                name, ext = os.path.splitext(filename)
                new = os.path.join(path, (name + time.asctime()) + ext)
                print new
                self.ppt.save(new)

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

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    main = None
    linecut = Linecut()
    operations = Operations()
    main = Window(linecut, operations, "test_data/Dev1_42.dat")
    
    linecut.main = main
    operations.main = main

    if len(sys.argv) > 1:
        main = Window(linecut, sys.argv[1])
    else:
        #filename = str(QtGui.QFileDialog.getOpenFileName(filter='*.dat'))
        #main = Window(linecut, filename)
        pass

    linecut.show()
    main.show()

    operations.show()

    sys.exit(app.exec_())