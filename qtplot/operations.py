import json
import os

from PyQt4 import QtCore, QtGui, uic

from .data import Data2D


class OperationWidget(QtGui.QWidget):
    """
    This widget has a grid layout of all the GUI elements needed to adjust
    the settings of an operation. It is used in a QStackedWidget.
    """
    def __init__(self, parameters, callback):
        super(OperationWidget, self).__init__(None)

        self.widgets = {}
        self.types = {}

        layout = QtGui.QGridLayout(self)
        height = 1

        # For every parameter in the Operation widget, create the appropriate
        # parameter widget depending on the data type
        for parameter, value in sorted(parameters.items()):
            if type(value) is bool:
                checkbox = QtGui.QCheckBox(parameter)
                checkbox.setChecked(value)
                checkbox.stateChanged.connect(callback)
                layout.addWidget(checkbox, height, 2)

                self.widgets[parameter] = checkbox
            elif type(value) in [int, float] or value is None:
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

            """
            self.types[parameter] = type(value)

            if value is None:
                self.types[parameter] = float
            else:
                self.types[parameter] = type(value)
            """

            self.types[parameter] = float if value is None else type(value)

            height += 1

    def get_parameter(self, name):
        """ Return the casted value of a parameter. """
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
        if name in self.widgets:
            widget = self.widgets[name]

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
    """
    This window contains a list of operations to choose from, and a list of
    the current active operations.
    """
    def __init__(self, parent, model):
        super(Operations, self).__init__(parent)

        self.model = model

        path = os.path.join(self.model.dir, 'ui/operations.ui')
        uic.loadUi(path, self)

        self.load_operations()

        # For some reason the StackedWidget has 3 default wigets
        # here we remove them
        for i in range(3):
            self.stack.removeWidget(self.stack.widget(0))

        self.bind()

        pos = parent.mapToGlobal(parent.rect().topRight())
        self.move(pos.x() + 3, pos.y() + 450)

        self.show()

    def bind(self):
        self.b_add.clicked.connect(self.on_add)
        self.b_up.clicked.connect(self.on_up)
        self.b_down.clicked.connect(self.on_down)
        self.b_remove.clicked.connect(self.on_remove)
        self.b_clear.clicked.connect(self.on_clear)
        self.b_load.clicked.connect(self.on_load)
        self.b_save.clicked.connect(self.on_save)

        self.lw_operations.currentItemChanged.connect(self.on_select_operation)
        self.lw_queue.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.lw_queue.itemClicked.connect(self.on_item_clicked)

        self.model.operations_changed.connect(self.on_operations_changed)

    def load_operations(self):
        directory = os.path.dirname(os.path.realpath(__file__))

        path = os.path.join(directory, 'default_operations.json')
        with open(path) as f:
            self.operation_defaults = json.load(f)

        self.lw_operations.addItems(sorted(self.operation_defaults.keys()))

    def get_current_parameters(self):
        index = self.stack.currentIndex()
        params = self.stack.currentWidget().get_parameters()

        return index, params

    def on_operation_changed(self):
        index, params = self.get_current_parameters()

        self.model.set_operation_parameters(index, params)

    def on_add(self):
        name = str(self.lw_operations.currentItem().text())

        self.model.add_operation(name)

    def on_up(self):
        index = self.get_current_parameters()[0]

        if index > 0:
            self.model.swap_operations(index - 1)

    def on_down(self):
        index = self.get_current_parameters()[0]

        if 0 <= index < self.lw_queue.count() - 1:
            self.model.swap_operations(index)

    def on_remove(self):
        index = self.get_current_parameters()[0]

        if index != -1:
            self.model.remove_operation(index)

    def on_clear(self):
        self.model.clear_operations()

    def on_load(self):
        filename = str(QtGui.QFileDialog.getOpenFileName(self,
                                                         'Open file',
                                                         #path,
                                                         '*.json'))

        if filename != '':
            self.model.load_operations(filename)

    def on_save(self):
        filename = QtGui.QFileDialog.getSaveFileName(self,
                                                     'Save file',
                                                     #path,
                                                     '.json')

        if filename != '':
            self.model.save_operations(filename)

    def on_select_operation(self, current, previous):
        if current:
            # Use the function documentation as description
            description = getattr(Data2D, str(current.text())).__doc__
            self.le_description.setText(description)

    def on_item_clicked(self, item):
        # If an item is clicked, update it's enabled state
        index = self.lw_queue.row(item)
        enabled = item.checkState() == QtCore.Qt.Checked

        self.model.set_operation_enabled(index, enabled)

    def on_operations_changed(self, event=None, op=None, index=None):
        """
        Handle when something in the operation list has changed
        This is mostly to update the UI
        """
        if event == 'add':
            item = QtGui.QListWidgetItem(op.name)

            if op.enabled:
                item.setCheckState(QtCore.Qt.Checked)
            else:
                item.setCheckState(QtCore.Qt.Unchecked)

            widget = OperationWidget(self.operation_defaults[op.name],
                                     self.on_operation_changed)

            # First update the widget with any parameters that are set
            widget.set_parameters(op.parameters)

            # Get all parameters from the widget
            op.parameters = widget.get_parameters()
            self.stack.addWidget(widget)

            self.lw_queue.addItem(item)
            self.lw_queue.setCurrentItem(item)
        elif event == 'values':
            pass
        elif event == 'swap':
            current = self.lw_queue.takeItem(index)
            self.lw_queue.insertItem(index + 1, current)

            current = self.stack.widget(index)
            self.stack.removeWidget(current)
            self.stack.insertWidget(index + 1, current)
        elif event == 'remove':
            self.lw_queue.takeItem(index)
            self.stack.removeWidget(self.stack.widget(index))
        elif event == 'clear':
            self.lw_queue.clear()

            while self.stack.count() > 0:
                self.stack.removeWidget(self.stack.widget(0))

        self.model.apply_operations()

    def keyPressEvent(self, event):
        # Listening to the Return pressed event in a LineEdit does not work
        # properly so we say the operations have changed when Return is
        # pressed anywhere in this window
        if event.key() == QtCore.Qt.Key_Return:
            self.on_operation_changed()

    def show_window(self):
        self.show()
        self.raise_()

    def closeEvent(self, event):
        self.hide()
        event.ignore()
