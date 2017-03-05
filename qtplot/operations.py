import json
import os

from PyQt4 import QtCore, QtGui, uic


class OperationWidget(QtGui.QWidget):
    def __init__(self, parameters, callback):
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
                checkbox.stateChanged.connect(callback)
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
                combobox.activated.connect(callback)
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
        for name, value in parameters.items():
            self.set_parameter(name, value)


class Operations(QtGui.QDialog):
    def __init__(self, parent, model):
        super(Operations, self).__init__(parent)

        self.model = model

        directory = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(directory, 'ui/operations.ui')
        uic.loadUi(path, self)

        self.load_operations()

        # For some reason the StackedWidget has 3 default wigets
        # here we remove them
        for i in range(3):
            self.stack.removeWidget(self.stack.widget(0))

        pos = parent.mapToGlobal(parent.rect().topRight())
        self.move(pos.x() + 3, pos.y() + 450)

        self.show()

    def bind(self):
        self.b_add.clicked.connect(self.on_add_operation)
        self.lw_queue.currentRowChanged.connect(self.stack.setCurrentIndex)

        self.model.operations_changed.connect(self.on_operations_changed)

    def load_operations(self):
        directory = os.path.dirname(os.path.realpath(__file__))

        path = os.path.join(directory, 'operation_defaults.json')
        with open(path) as f:
            self.operation_defaults = json.load(f)

        self.lw_operations.addItems(sorted(self.operation_defaults.keys()))

    def get_operation(self):
        return str(self.lw_operations.currentItem().text())

    def get_current_parameters(self):
        index = self.stack.currentIndex()
        params = self.stack.currentWidget().get_parameters()

        return index, params

    def on_operation_changed(self):
        index, params = self.op_view.get_current_parameters()

        self.model.set_operation_parameters(index, params)

    def on_add_operation(self):
        name = self.op_view.get_operation()

        self.model.add_operation(name, **self.operation_defaults[name])

    def on_operations_changed(self, event, operation=None):
        if event == 'add':
            item = QtGui.QListWidgetItem(operation.name)

            if operation.enabled:
                item.setCheckState(QtCore.Qt.Checked)
            else:
                item.setCheckState(QtCore.Qt.Unchecked)

            widget = OperationWidget(operation.parameters,
                                     self.on_operation_changed)
            self.stack.addWidget(widget)

            self.lw_queue.addItem(item)
            self.lw_queue.setCurrentItem(item)
        elif event == 'values':
            pass

        self.model.apply_operations()

    def show_window(self):
        self.show()
        self.raise_()

    def closeEvent(self, event):
        self.hide()
        event.ignore()
