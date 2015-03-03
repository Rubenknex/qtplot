from collections import OrderedDict
import numpy as np
import os
import pandas as pd
import json

from PyQt4 import QtGui, QtCore
from scipy import ndimage

from dat_file import Data

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

    def get_property(self, name, cast=str):
        if name in self.items:
            widget = self.items[name]

            if type(widget) is QtGui.QCheckBox:
                return widget.isChecked()
            elif type(widget) is QtGui.QLineEdit:
                return cast(str(widget.text()))
            elif type(widget) is QtGui.QComboBox:
                return str(widget.currentText())

    def set_property(self, name, value):
        if name in self.items:
            widget = self.items[name]

            if type(widget) is QtGui.QCheckBox:
                widget.setChecked(bool(value))
            elif type(widget) is QtGui.QLineEdit:
                widget.setText(value)
            elif type(widget) is QtGui.QComboBox:
                index = widget.findText(value)
                widget.setCurrentIndex(index)

    def get_formatted(self):
        params = {name: self.get_property(name) for name in self.items}

        return self.name, params

    def set_params(self, params):
        for name, value in params.iteritems():
            self.set_property(name, value)

class Operations(QtGui.QDialog):
    def __init__(self, parent=None):
        super(Operations, self).__init__(parent)

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Operations")

        self.items = {
            'abs':          [Data.abs],
            'autoflip':     [Data.autoflip],
            'crop':         [Data.crop, [('textbox', 'Left', '0'), ('textbox', 'Right', '0'), ('textbox', 'Bottom', '0'), ('textbox', 'Top', '0')]],
            'dderiv':       [Data.dderiv, [('textbox', 'Theta', '0')]],
            'equalize':     [Data.equalize],
            'even odd':     [Data.even_odd],
            'flip':         [Data.flip, [('checkbox', 'X Axis', False), ('checkbox', 'Y Axis', False)]],
            'gradmag':      [Data.gradmag],
            'highpass':     [Data.highpass, [('textbox', 'X Width', '3'), ('textbox', 'Y Height', '3'), ('combobox', 'Type', ['Gaussian', 'Lorentzian', 'Exponential', 'Thermal'])]],
            'hist2d':       [Data.hist2d],
            'log':          [Data.log],
            'lowpass':      [Data.lowpass, [('textbox', 'X Width', '3'), ('textbox', 'Y Height', '3'), ('combobox', 'Type', ['Gaussian', 'Lorentzian', 'Exponential', 'Thermal'])]],
            'neg':          [Data.neg],
            'normalize':    [Data.normalize],
            'offset':       [Data.offset, [('textbox', 'Offset', '0')]],
            'offset axes':  [Data.offset_axes, [('textbox', 'X Offset', '0'), ('textbox', 'Y Offset', '0')]],
            'power':        [Data.power, [('textbox', 'Power', '1')]],
            'scale axes':   [Data.scale_axes, [('textbox', 'X Scale', '1'), ('textbox', 'Y Scale', '1')]],
            'scale data':   [Data.scale_data, [('textbox', 'Factor', '1')]],
            'sub linecut':  [Data.sub_linecut],
            'xderiv':       [Data.xderiv],
            'yderiv':       [Data.yderiv],
        }

        self.options = QtGui.QListWidget(self)
        self.options.addItems(sorted(self.items.keys()))

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
        
        self.setLayout(hbox)

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
            op.set_params(operation)
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
                
                name, params = operation.get_formatted()
                operations[name] = params

        with open(filename, 'w') as f:
            f.write(json.dumps(operations, indent=4))

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Return:
            self.main.on_data_change()

    def on_selected_changed(self, current, previous):
        if current:
            widget = current.data(QtCore.Qt.UserRole).toPyObject()
            self.stack.addWidget(widget)
            self.stack.setCurrentWidget(widget)

    def apply_operations(self, data):
        ops = []

        copy = data.copy()

        for i in xrange(self.queue.count()):
            item = self.queue.item(i)
            operation = item.data(QtCore.Qt.UserRole).toPyObject()
            name = str(self.queue.item(i).text())

            kwargs = operation.get_formatted()[1]
            kwargs['linecut_type'] = self.main.linecut_type
            kwargs['linecut_coord'] = self.main.linecut_coord

            copy = operation.func(copy, **kwargs)

        return copy