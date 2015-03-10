from collections import OrderedDict
import numpy as np
import os
import pandas as pd
import json

from PyQt4 import QtGui, QtCore
from scipy import ndimage

from data import Data

class Operation(QtGui.QWidget):
    """Contains the name and GUI widgets for the parameters of an operation."""
    def __init__(self, name, func, widgets=[]):
        super(Operation, self).__init__(None)

        layout = QtGui.QGridLayout(self)
        self.name = name
        self.func = func
        self.items = {}

        height = 1

        for widget in widgets:
            typ, name, data = widget

            if typ == 'checkbox':
                checkbox = QtGui.QCheckBox(name)
                checkbox.setChecked(data)
                layout.addWidget(checkbox, height, 2)

                self.items[name] = checkbox
            elif typ == 'textbox':
                lineedit = QtGui.QLineEdit(data)
                lineedit.setValidator(QtGui.QDoubleValidator())
                layout.addWidget(QtGui.QLabel(name), height, 1)
                layout.addWidget(lineedit, height, 2)

                self.items[name] = lineedit
            elif typ == 'combobox':
                layout.addWidget(QtGui.QLabel(name), height, 1)
                combobox = QtGui.QComboBox()
                combobox.addItems(data)
                layout.addWidget(combobox, height, 2)

                self.items[name] = combobox

            height += 1

    def get_parameter(self, name, cast=str):
        """Return the casted value of a property."""
        if name in self.items:
            widget = self.items[name]

            if type(widget) is QtGui.QCheckBox:
                return widget.isChecked()
            elif type(widget) is QtGui.QLineEdit:
                return cast(str(widget.text()))
            elif type(widget) is QtGui.QComboBox:
                return str(widget.currentText())

    def set_parameter(self, name, value):
        """Set a property to a value."""
        if name in self.items:
            widget = self.items[name]

            if type(widget) is QtGui.QCheckBox:
                widget.setChecked(bool(value))
            elif type(widget) is QtGui.QLineEdit:
                widget.setText(value)
            elif type(widget) is QtGui.QComboBox:
                index = widget.findText(value)
                widget.setCurrentIndex(index)

    def get_parameters(self):
        """Return a tuple of the name of the operation and a dict of the parameters."""
        params = {name: self.get_parameter(name) for name in self.items}

        return self.name, params

    def set_parameters(self, params):
        """Set all the parameters with a dict containing them."""
        for name, value in params.iteritems():
            self.set_parameter(name, value)

class Operations(QtGui.QDialog):
    def __init__(self, parent=None):
        super(Operations, self).__init__(parent)

        self.columns = None

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Operations")

        self.items = {
            'abs':          [Data.abs],
            'autoflip':     [Data.autoflip],
            'crop':         [Data.crop, [('textbox', 'Left', '0'), ('textbox', 'Right', '-1'), ('textbox', 'Bottom', '0'), ('textbox', 'Top', '-1')]],
            'dderiv':       [Data.dderiv, [('textbox', 'Theta', '0')]],
            'equalize':     [Data.equalize],
            'even odd':     [Data.even_odd],
            'flip':         [Data.flip, [('checkbox', 'X Axis', False), ('checkbox', 'Y Axis', False)]],
            'gradmag':      [Data.gradmag],
            'highpass':     [Data.highpass, [('textbox', 'X Width', '3'), ('textbox', 'Y Height', '3'), ('combobox', 'Type', ['Gaussian', 'Lorentzian', 'Exponential', 'Thermal'])]],
            'hist2d':       [Data.hist2d, [('combobox', 'Axis', ['Horizontal', 'Vertical']), ('textbox', 'Min', '0'), ('textbox', 'Max', '-1'), ('textbox', 'Bins', '20')]],
            'interp grid':  [Data.interp_grid],
            'log':          [Data.log],
            'lowpass':      [Data.lowpass, [('textbox', 'X Width', '3'), ('textbox', 'Y Height', '3'), ('combobox', 'Type', ['Gaussian', 'Lorentzian', 'Exponential', 'Thermal'])]],
            'neg':          [Data.neg],
            'norm columns': [Data.norm_columns],
            'norm rows':    [Data.norm_rows],
            'offset':       [Data.offset, [('textbox', 'Offset', '0')]],
            'offset axes':  [Data.offset_axes, [('textbox', 'X Offset', '0'), ('textbox', 'Y Offset', '0')]],
            'power':        [Data.power, [('textbox', 'Power', '1')]],
            'scale axes':   [Data.scale_axes, [('textbox', 'X Scale', '1'), ('textbox', 'Y Scale', '1')]],
            'scale data':   [Data.scale_data, [('textbox', 'Factor', '1')]],
            'sub linecut':  [Data.sub_linecut],
            'sub plane':    [Data.sub_plane, [('textbox', 'X Slope', '0'), ('textbox', 'Y Slope', '0')]],
            'xderiv':       [Data.xderiv, [('combobox', 'Method', ['midpoint', '2nd order central diff'])]],
            'yderiv':       [Data.yderiv, [('combobox', 'Method', ['midpoint', '2nd order central diff'])]],
        }

        self.options = QtGui.QListWidget(self)
        self.options.addItems(sorted(self.items.keys()))
        self.options.currentItemChanged.connect(self.on_select_option)

        self.b_add = QtGui.QPushButton('Add')
        self.b_add.clicked.connect(self.on_add)

        self.b_up = QtGui.QPushButton('Up')
        self.b_up.clicked.connect(self.on_up)

        self.b_down = QtGui.QPushButton('Down')
        self.b_down.clicked.connect(self.on_down)

        self.b_remove = QtGui.QPushButton('Remove')
        self.b_remove.clicked.connect(self.on_remove)

        self.b_clear = QtGui.QPushButton('Clear')
        self.b_clear.clicked.connect(self.on_clear)

        self.b_update = QtGui.QPushButton('Update')
        self.b_update.clicked.connect(self.on_update)

        self.b_load = QtGui.QPushButton('Load...')
        self.b_load.clicked.connect(self.on_load)

        self.b_save = QtGui.QPushButton('Save...')
        self.b_save.clicked.connect(self.on_save)

        self.queue = QtGui.QListWidget(self)
        self.queue.currentItemChanged.connect(self.on_selected_changed)
        self.queue.itemClicked.connect(self.on_item_clicked)

        self.le_help = QtGui.QLineEdit(self)
        self.le_help.setReadOnly(True)

        main_vbox = QtGui.QVBoxLayout()

        hbox = QtGui.QHBoxLayout()

        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.b_add)
        vbox.addWidget(self.b_up)
        vbox.addWidget(self.b_down)
        vbox.addWidget(self.b_remove)
        vbox.addWidget(self.b_clear)
        vbox.addWidget(self.b_update)
        vbox.addWidget(self.b_load)
        vbox.addWidget(self.b_save)

        vbox2 = QtGui.QVBoxLayout()
        vbox2.addWidget(self.queue)
        self.stack = QtGui.QStackedWidget()
        vbox2.addWidget(self.stack)

        hbox.addWidget(self.options)
        hbox.addLayout(vbox)
        hbox.addLayout(vbox2)

        main_vbox.addLayout(hbox)
        main_vbox.addWidget(self.le_help)
        
        self.setLayout(main_vbox)

        self.setGeometry(800, 700, 400, 200)

    def update_plot(func):
        def wrapper(self):
            func(self)
            self.main.on_data_change()

        return wrapper
    
    @update_plot
    def on_add(self):
        if self.options.currentItem():
            name = str(self.options.currentItem().text())

            item = QtGui.QListWidgetItem(name)
            item.setCheckState(QtCore.Qt.Checked)
            operation = Operation(name, *self.items[name])
            item.setData(QtCore.Qt.UserRole, QtCore.QVariant(operation))
            self.stack.addWidget(operation)

            self.queue.addItem(item)
            self.queue.setCurrentItem(item)

    @update_plot
    def on_up(self):
        selected_row = self.queue.currentRow()
        current = self.queue.takeItem(selected_row)
        self.queue.insertItem(selected_row - 1, current)
        self.queue.setCurrentRow(selected_row - 1)

    @update_plot
    def on_down(self):
        selected_row = self.queue.currentRow()
        current = self.queue.takeItem(selected_row)
        self.queue.insertItem(selected_row + 1, current)
        self.queue.setCurrentRow(selected_row + 1)

    @update_plot
    def on_remove(self):
        self.queue.takeItem(self.queue.currentRow())

    @update_plot
    def on_clear(self):
        self.queue.clear()

    @update_plot
    def on_update(self):
        pass

    @update_plot
    def on_load(self):
        path = os.path.dirname(os.path.realpath(__file__))
        filename = str(QtGui.QFileDialog.getOpenFileName(self, 'Open file', path, '*.operations'))

        if filename == '':
            return

        self.queue.clear()

        with open(filename) as f:
            operations = json.load(f, object_pairs_hook=OrderedDict)

        for name, operation in operations.iteritems():
            item = QtGui.QListWidgetItem(name)
            op = Operation(name, *self.items[name])
            op.set_parameters(operation)
            item.setData(QtCore.Qt.UserRole, QtCore.QVariant(op))
            self.stack.addWidget(op)

            self.queue.addItem(item)
            self.queue.setCurrentItem(item)
    
    def on_save(self):
        path = os.path.dirname(os.path.realpath(__file__))
        filename = QtGui.QFileDialog.getSaveFileName(self, 'Save file', path, '.operations')

        if filename == '':
            return

        operations = OrderedDict()
        for i in xrange(self.queue.count()):
                operation = self.queue.item(i).data(QtCore.Qt.UserRole).toPyObject()
                
                name, params = operation.get_parameters()
                operations[name] = params

        with open(filename, 'w') as f:
            f.write(json.dumps(operations, indent=4))

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Return:
            self.main.on_data_change()

    def on_select_option(self, current, previous):
        if current:
            description = self.items[str(current.text())][0].__doc__
            self.le_help.setText(description)

    def on_selected_changed(self, current, previous):
        if current:
            widget = current.data(QtCore.Qt.UserRole).toPyObject()
            self.stack.addWidget(widget)
            self.stack.setCurrentWidget(widget)

    def on_item_clicked(self, item):
        self.main.on_data_change()

    def apply_operations(self, data):
        copy = data.copy()

        for i in xrange(self.queue.count()):
            item = self.queue.item(i)

            if item.checkState() == QtCore.Qt.Unchecked:
                continue

            operation = item.data(QtCore.Qt.UserRole).toPyObject()

            kwargs = operation.get_parameters()[1]
            kwargs['linecut_type'] = self.main.line_type
            kwargs['linecut_coord'] = self.main.line_coord

            copy = operation.func(copy, **kwargs)

        return copy

    def show_window(self):
        self.show()
        self.raise_()

    def closeEvent(self, event):
        self.hide()
        event.ignore()