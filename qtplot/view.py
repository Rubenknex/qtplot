import os

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

    def get_parameters(self):
        return [str(cb.currentText()) for cb in self.cb_parameters]

    def get_reset_colormap(self):
        return self.cb_reset_on_plot.checkState() == QtCore.Qt.Checked

    def get_cmap_name(self):
        return str(self.cb_colormap.currentText())


class LineView(QtGui.QDialog):
    def __init__(self):
        super(LineView, self).__init__()

        directory = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(directory, 'ui/linetrace.ui')
        uic.loadUi(path, self)


class OperationsView(QtGui.QDialog):
    def __init__(self):
        super(OperationsView, self).__init__()

        directory = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(directory, 'ui/operations.ui')
        uic.loadUi(path, self)


class SettingsView(QtGui.QDialog):
    def __init__(self):
        super(SettingsView, self).__init__()

        directory = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(directory, 'ui/settings.ui')
        uic.loadUi(path, self)
