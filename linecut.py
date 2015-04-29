import math
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
import numpy as np
from PyQt4 import QtGui, QtCore
import os
import pandas as pd

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

class Linecut(QtGui.QDialog):
    def __init__(self, parent=None):
        super(Linecut, self).__init__(parent)

        self.fig, self.ax = plt.subplots()
        self.x, self.y = None, None
        self.line = self.ax.plot(0, 0, color='red', linewidth=0.5)[0]

        self.ax.xaxis.set_major_formatter(FixedOrderFormatter())
        self.ax.yaxis.set_major_formatter(FixedOrderFormatter())

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Linecut")

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        hbox = QtGui.QHBoxLayout()

        self.cb_reset_cmap = QtGui.QCheckBox('Reset on plot')
        self.cb_reset_cmap.setCheckState(QtCore.Qt.Checked)
        hbox.addWidget(self.cb_reset_cmap)

        self.b_save = QtGui.QPushButton('Copy data to clipboard', self)
        self.b_save.clicked.connect(self.on_clipboard)
        hbox.addWidget(self.b_save)

        self.b_save_dat = QtGui.QPushButton('Save data...', self)
        self.b_save_dat.clicked.connect(self.on_save)
        hbox.addWidget(self.b_save_dat)

        self.b_copy = QtGui.QPushButton('Copy figure to clipboard (Ctrl+C)', self)
        self.b_copy.clicked.connect(self.on_copy_figure)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+C"), self, self.on_copy_figure)
        hbox.addWidget(self.b_copy)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addLayout(hbox)
        self.setLayout(layout)

        self.move(800, 100)

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

        self.line.set_xdata(x)
        self.line.set_ydata(y)

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

        self.canvas.draw()

    def resizeEvent(self, event):
        self.fig.tight_layout()
        self.canvas.draw()

    def show_window(self):
        self.show()
        self.raise_()

    def closeEvent(self, event):
        self.hide()
        event.ignore()