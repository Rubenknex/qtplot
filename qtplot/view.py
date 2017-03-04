import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg, \
     NavigationToolbar2QT
from PyQt4 import QtCore, QtGui, uic

from .canvas import Canvas
from .data import Data2D


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

    def show_window(self):
        self.show()
        self.raise_()

    def closeEvent(self, event):
        self.hide()
        event.ignore()


class OperationWidget(QtGui.QWidget):
    def __init__(self, parameters):
        super(OperationWidget, self).__init__(None)

        self.widgets = {}
        self.types = {}

        layout = QtGui.QGridLayout(self)
        height = 1

        # For every parameter in the Operation widget, create the appropriate
        # parameter widget depending on the data type
        for parameter, value in parameters.items():
            if type(value) is bool:
                checkbox = QtGui.QCheckBox(parameter)
                checkbox.setChecked(value)
                #checkbox.stateChanged.connect(self.main.on_data_change)
                layout.addWidget(checkbox, height, 2)

                self.widgets[parameter] = checkbox
            elif type(value) is int or type(value) is float:
                lineedit = QtGui.QLineEdit(str(value))
                lineedit.setValidator(QtGui.QDoubleValidator())
                layout.addWidget(QtGui.QLabel(parameter), height, 1)
                layout.addWidget(lineedit, height, 2)

                self.widgets[parameter] = lineedit
            elif type(value) is list:
                layout.addWidget(QtGui.QLabel(parameter), height, 1)
                combobox = QtGui.QComboBox()
                #combobox.activated.connect(self.main.on_data_change)
                combobox.addItems(value)
                layout.addWidget(combobox, height, 2)

                self.widgets[parameter] = combobox

            self.types[parameter] = type(value)

            height += 1

    def get_parameter(self, name):
        """ Return the casted value of a property. """
        if name in self.widgets:
            widget = self.widgets[name]
            cast = self.types[name]

            if type(widget) is QtGui.QCheckBox:
                return cast(widget.isChecked())
            elif type(widget) is QtGui.QLineEdit:
                return cast(str(widget.text()))
            elif type(widget) is QtGui.QComboBox:
                return str(widget.currentText())

    def get_parameters(self):
        """ Returna dict of the parameters """
        params = {name: self.get_parameter(name) for name in self.widgets}

        return params

    def set_parameter(self, name, value):
        """ Set a parameter to a value. """
        if name in self.items:
            widget = self.items[name]

            if type(widget) is QtGui.QCheckBox:
                widget.setChecked(bool(value))
            elif type(widget) is QtGui.QLineEdit:
                widget.setText(str(value))
            elif type(widget) is QtGui.QComboBox:
                index = widget.findText(value)
                widget.setCurrentIndex(index)

    def set_parameters(self, parameters):
        for name, value in params.items():
            self.set_parameter(name, value)


class OperationsView(QtGui.QDialog):
    def __init__(self, parent):
        super(OperationsView, self).__init__(parent)

        directory = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(directory, 'ui/operations.ui')
        uic.loadUi(path, self)

        # For some reason the StackedWidget has 3 default wigets
        # here we remove them
        for i in range(3):
            self.stack.removeWidget(self.stack.widget(0))

        pos = parent.mapToGlobal(parent.rect().topRight())
        self.move(pos.x() + 3, pos.y() + 450)

        self.show()

    def get_operation(self):
        return str(self.lw_operations.currentItem().text())

    def show_window(self):
        self.show()
        self.raise_()

    def closeEvent(self, event):
        self.hide()
        event.ignore()


class SettingsView(QtGui.QDialog):
    def __init__(self, parent):
        super(SettingsView, self).__init__(parent)

        directory = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(directory, 'ui/settings.ui')
        uic.loadUi(path, self)

    def show_window(self):
        self.show()
        self.raise_()

    def closeEvent(self, event):
        self.hide()
        event.ignore()
