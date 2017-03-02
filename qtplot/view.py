import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg, NavigationToolbar2QT
from PyQt4 import QtCore, QtGui, uic

from .canvas import Canvas


class MainView(QtGui.QMainWindow):
    def __init__(self):
        super(MainView, self).__init__()

        directory = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(directory, 'ui/main.ui')
        uic.loadUi(path, self)

        self.cb_parameters = [self.cb_x, self.cb_y, self.cb_z]
        self.sliders = [self.s_cmap_min, self.s_cmap_gamma, self.s_cmap_max]

        self.canvas = Canvas()
        self.canvas_layout.addWidget(self.canvas.native)

        center = QtGui.QApplication.desktop().screen().rect().center()
        self.move(center.x() - 550, center.y() - 400)

        self.show()

    def get_parameters(self):
        return [str(cb.currentText()) for cb in self.cb_parameters]

    def get_reset_colormap(self):
        return self.cb_reset_on_plot.checkState() == QtCore.Qt.Checked

    def get_cmap_name(self):
        return str(self.cb_colormap.currentText())


class LineView(QtGui.QDialog):
    def __init__(self, parent):
        super(LineView, self).__init__(parent)

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
        self.move(pos.x() + 3, pos.y()  - 25)

        self.show()

class OperationsView(QtGui.QDialog):
    def __init__(self, parent):
        super(OperationsView, self).__init__(parent)

        directory = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(directory, 'ui/operations.ui')
        uic.loadUi(path, self)

        pos = parent.mapToGlobal(parent.rect().topRight())
        self.move(pos.x() + 3, pos.y() + 450)

        self.show()


class SettingsView(QtGui.QDialog):
    def __init__(self, parent):
        super(SettingsView, self).__init__(parent)

        directory = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(directory, 'ui/settings.ui')
        uic.loadUi(path, self)
