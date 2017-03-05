import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg, \
    NavigationToolbar2QT
from PyQt4 import QtCore, QtGui, uic


class Linetrace(QtGui.QDialog):
    def __init__(self, parent, model):
        super(Linetrace, self).__init__(parent)

        self.model = model

        directory = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(directory, 'ui/linetrace.ui')
        uic.loadUi(path, self)

        self.fig, self.ax = plt.subplots()

        self.canvas = FigureCanvasQTAgg(self.fig)
        #self.canvas.mpl_connect('pick_event', self.on_pick)
        #self.canvas.mpl_connect('button_press_event', self.on_press)

        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        self.layout().insertWidget(0, self.canvas)
        self.layout().insertWidget(0, self.toolbar)

        pos = parent.mapToGlobal(parent.rect().topRight())
        self.move(pos.x() + 3, pos.y() - 25)

        self.show()

    def bind(self):
        self.model.linetrace_changed.connect(self.on_linetrace_changed)

    def on_linetrace_changed(self):
        # Use set_data
        self.ax.clear()

        self.ax.plot(*self.model.linetrace.get_matplotlib())

        self.ax.set_aspect('auto')
        self.fig.tight_layout()

        self.fig.canvas.draw()

    def show_window(self):
        self.show()
        self.raise_()

    def closeEvent(self, event):
        self.hide()
        event.ignore()
