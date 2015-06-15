import math
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
import numpy as np
from PyQt4 import QtGui, QtCore
import os
import pandas as pd

from util import FixedOrderFormatter

class Linecut(QtGui.QDialog):
    def __init__(self, parent=None):
        super(Linecut, self).__init__(parent)

        self.fig, self.ax = plt.subplots()
        self.x, self.y = None, None
        self.line = self.ax.plot(0, 0, color='red', linewidth=0.5)[0]
        self.lines = []
        self.total_offset = 0

        self.ax.xaxis.set_major_formatter(FixedOrderFormatter())
        self.ax.yaxis.set_major_formatter(FixedOrderFormatter())

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Linecut")

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        grid = QtGui.QGridLayout()

        self.cb_reset_cmap = QtGui.QCheckBox('Reset on plot')
        self.cb_reset_cmap.setCheckState(QtCore.Qt.Checked)
        grid.addWidget(self.cb_reset_cmap, 1, 1)

        self.b_save = QtGui.QPushButton('Data to clipboard', self)
        self.b_save.clicked.connect(self.on_clipboard)
        grid.addWidget(self.b_save, 1, 2)

        self.b_save_dat = QtGui.QPushButton('Save data...', self)
        self.b_save_dat.clicked.connect(self.on_save)
        grid.addWidget(self.b_save_dat, 1, 3)

        self.b_copy = QtGui.QPushButton('Figure to clipboard', self)
        self.b_copy.clicked.connect(self.on_copy_figure)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+C"), self, self.on_copy_figure)
        grid.addWidget(self.b_copy, 1, 4)

        self.cb_incremental = QtGui.QCheckBox('Incremental')
        self.cb_incremental.setCheckState(QtCore.Qt.Unchecked)
        grid.addWidget(self.cb_incremental, 2, 1)

        grid.addWidget(QtGui.QLabel('Offset:'), 2, 2)

        self.le_offset = QtGui.QLineEdit('0', self)
        grid.addWidget(self.le_offset, 2, 3)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addLayout(grid)
        self.setLayout(layout)

        self.resize(500, 500)
        self.move(720, 100)

    def on_reset(self):
        if self.x != None and self.y != None:
            minx, maxx = np.min(self.x), np.max(self.x)
            miny, maxy = np.min(self.y), np.max(self.y)

            xdiff = (maxx - minx) * .1
            ydiff = (maxy - miny) * .1

            self.ax.axis([minx - xdiff, maxx + xdiff, miny - ydiff, maxy + ydiff])
            self.canvas.draw()

    def on_clipboard(self):
        if self.x == None or self.y == None:
            return

        data = pd.DataFrame(np.column_stack((self.x, self.y)), columns=[self.xlabel, self.ylabel])
        data.to_clipboard(index=False)

    def on_save(self):
        if self.x == None or self.y == None:
            return

        path = os.path.dirname(os.path.realpath(__file__))
        filename = QtGui.QFileDialog.getSaveFileName(self, 'Save file', path, '.dat')

        if filename != '':
            data = pd.DataFrame(np.column_stack((self.x, self.y)), columns=[self.xlabel, self.ylabel])
            data.to_csv(filename, sep='\t', index=False)

    def on_copy_figure(self):
        path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(path, 'test.png')
        self.fig.savefig(path, bbox_inches='tight')

        img = QtGui.QImage(path)
        QtGui.QApplication.clipboard().setImage(img)
    
    def plot_linecut(self, x, y, title, xlabel, ylabel):
        self.xlabel, self.ylabel = xlabel, ylabel
        self.x, self.y = x, y

        # Remove all the existing lines and only plot one if we uncheck the incremental box
        # Else, add a new line to the collection
        if self.cb_incremental.checkState() == QtCore.Qt.Unchecked:
            for line in self.lines:
                self.ax.lines.remove(line)
                
            self.lines = []

            self.line.set_xdata(x)
            self.line.set_ydata(y)

            self.total_offset = 0
        else:
            self.line.set_xdata([])
            self.line.set_ydata([])

            line = self.ax.plot(x, y + self.total_offset)[0]
            self.lines.append(line)

            self.total_offset += float(self.le_offset.text())

        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

        if self.cb_reset_cmap.checkState() == QtCore.Qt.Checked:
            x, y = np.ma.masked_invalid(x), np.ma.masked_invalid(y)
            minx, maxx = np.min(x), np.max(x)
            miny, maxy = np.min(y), np.max(y)

            xdiff = (maxx - minx) * .05
            ydiff = (maxy - miny) * .05

            self.ax.axis([minx - xdiff, maxx + xdiff, miny - ydiff, maxy + ydiff])

        self.ax.set_aspect('auto')
        self.fig.tight_layout()

        self.fig.canvas.draw()

    def resizeEvent(self, event):
        self.fig.tight_layout()
        self.canvas.draw()

    def show_window(self):
        self.show()
        self.raise_()

    def closeEvent(self, event):
        self.hide()
        event.ignore()