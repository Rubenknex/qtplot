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

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Linecut")

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        hbox = QtGui.QHBoxLayout()

        self.b_save = QtGui.QPushButton('Copy data to clipboard', self)
        self.b_save.clicked.connect(self.on_save_data)
        hbox.addWidget(self.b_save)

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

    def on_save_data(self):
        if self.x == None or self.y == None:
            return

        data = pd.DataFrame(np.column_stack((self.x, self.y)), columns=[self.xlabel, self.ylabel])
        data.to_clipboard(index=False)
        """
        path = os.path.dirname(os.path.realpath(__file__))
        filename = QtGui.QFileDialog.getSaveFileName(self, 'Save file', path, '.operations')

        if filename != '':
            data.to_csv(filename)
        """

    def on_copy_figure(self):
        path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(path, 'test.png')
        self.fig.savefig(path, bbox_inches='tight')

        img = QtGui.QImage(path)
        QtGui.QApplication.clipboard().setImage(img)
    
    def plot_linecut(self, x, y, title, xlabel, ylabel):
        self.xlabel, self.ylabel = xlabel, ylabel
        self.x, self.y = x, y

        if len(self.ax.lines) > 0:
            self.ax.lines.pop(0)

        self.ax.plot(x, y, color='red', linewidth=0.5)

        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

        self.ax.xaxis.set_major_formatter(FixedOrderFormatter())
        self.ax.yaxis.set_major_formatter(FixedOrderFormatter())

        self.ax.relim()
        self.ax.autoscale_view()
        
        self.ax.set_aspect('auto')
        self.fig.tight_layout()

        self.canvas.draw()

    def resizeEvent(self, event):
        self.fig.tight_layout()
        self.canvas.draw()

    def closeEvent(self, event):
        self.hide()
        event.ignore()