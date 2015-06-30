from collections import OrderedDict
import numpy as np
import os
import pandas as pd
import json
import math

from PyQt4 import QtGui, QtCore
from scipy import ndimage

from data import Data2D

class Operation(QtGui.QWidget):
    """Contains the name and GUI widgets for the parameters of an operation."""
    def __init__(self, name, main, func, widgets=[]):
        super(Operation, self).__init__(None)

        layout = QtGui.QGridLayout(self)
        self.name = name
        self.main = main
        self.func = func
        self.items = {}
        self.types = {}

        height = 1

        for widget in widgets:
            w_name, data = widget

            if type(data) == bool:
                checkbox = QtGui.QCheckBox(w_name)
                checkbox.setChecked(data)
                checkbox.stateChanged.connect(self.main.on_data_change)
                layout.addWidget(checkbox, height, 2)

                self.items[w_name] = checkbox
            elif type(data) == int or type(data) == float:
                lineedit = QtGui.QLineEdit(str(data))
                lineedit.setValidator(QtGui.QDoubleValidator())
                layout.addWidget(QtGui.QLabel(w_name), height, 1)
                layout.addWidget(lineedit, height, 2)

                self.items[w_name] = lineedit
            elif type(data) == list:
                layout.addWidget(QtGui.QLabel(w_name), height, 1)
                combobox = QtGui.QComboBox()
                combobox.activated.connect(self.main.on_data_change)
                combobox.addItems(data)
                layout.addWidget(combobox, height, 2)

                self.items[w_name] = combobox

            self.types[w_name] = type(data)

            height += 1

        if name == 'sub linecut' or name == 'sub linecut avg':
            b_current = QtGui.QPushButton('Current linecut')
            b_current.clicked.connect(self.on_current_linecut)
            layout.addWidget(b_current, height, 2)

    def on_current_linecut(self):
        index = self.items['type'].findText(self.main.canvas.line_type)
        self.items['type'].setCurrentIndex(index)
        self.items['position'].setText(str(self.main.canvas.line_coord))

    def get_parameter(self, name):
        """Return the casted value of a property."""
        if name in self.items:
            widget = self.items[name]
            cast = self.types[name]

            if type(widget) is QtGui.QCheckBox:
                return cast(widget.isChecked())
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
                widget.setText(str(value))
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
            'abs':          [Data2D.abs],
            'autoflip':     [Data2D.autoflip],
            'crop':         [Data2D.crop,         [('left', 0), ('right', -1), ('bottom', 0), ('top', -1)]],
            'dderiv':       [Data2D.dderiv,       [('theta', 0), ('method', ['midpoint', '2nd order central diff'])]],
            'equalize':     [Data2D.equalize],
            'even odd':     [Data2D.even_odd,     [('even', True)]],
            'flip':         [Data2D.flip,         [('x_flip', False), ('y_flip', False)]],
            'gradmag':      [Data2D.gradmag,      [('method', ['midpoint', '2nd order central diff'])]],
            'highpass':     [Data2D.highpass,     [('x_width', 3.0), ('y_height', 3.0), ('method', ['gaussian', 'lorentzian', 'exponential', 'thermal'])]],
            'hist2d':       [Data2D.hist2d,       [('min', 0.0), ('max', 0.0), ('bins', 0)]],
            'interp grid':  [Data2D.interp_grid,  [('width', 100), ('height', 100)]],
            'interp x':     [Data2D.interp_x,     [('points', 100)]],
            'interp y':     [Data2D.interp_y,     [('points', 100)]],
            'log':          [Data2D.log,          [('subtract', False), ('min', 0.0001)]],
            'lowpass':      [Data2D.lowpass,      [('x_width', 3.0), ('y_height', 3.0), ('method', ['gaussian', 'lorentzian', 'exponential', 'thermal'])]],
            'negate':       [Data2D.negate],
            'norm y':       [Data2D.norm_columns],
            'norm x':       [Data2D.norm_rows],
            'offset':       [Data2D.offset,       [('offset', 0.0)]],
            'offset axes':  [Data2D.offset_axes,  [('x_offset', 0.0), ('y_offset', 0.0)]],
            'power':        [Data2D.power,        [('power', 1.0)]],
            'scale axes':   [Data2D.scale_axes,   [('x_scale', 1.0), ('y_scale', 1.0)]],
            'scale data':   [Data2D.scale_data,   [('factor', 1.0)]],
            'sub linecut':  [Data2D.sub_linecut,  [('type', ['horizontal', 'vertical']), ('position', float('nan'))]],
            'sub linecut avg': [Data2D.sub_linecut_avg, [('type', ['horizontal', 'vertical']), ('position', float('nan')), ('size', 3)]],
            'sub plane':    [Data2D.sub_plane,    [('x_slope', 0.0), ('y_slope', 0.0)]],
            'xderiv':       [Data2D.xderiv,       [('method', ['midpoint', '2nd order central diff'])]],
            'yderiv':       [Data2D.yderiv,       [('method', ['midpoint', '2nd order central diff'])]],
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

        self.resize(400, 200)
        self.move(720, 640)

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
            operation = Operation(name, self.main, *self.items[name])
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
            item.setCheckState(QtCore.Qt.Checked)
            op = Operation(name, self.main, *self.items[name])
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

            if operation.name == 'hist2d':
                if operation.get_parameter('bins') == 0:
                    bins = np.round(np.sqrt(copy.z.shape[0]))
                    operation.set_parameter('bins', int(bins))

                if operation.get_parameter('min') == 0:
                    min, max = np.nanmin(copy.z), np.nanmax(copy.z)
                    operation.set_parameter('min', min)
                    operation.set_parameter('max', max)
            elif operation.name == 'sub linecut' or operation.name == 'sub linecut avg':
                if self.main.canvas.line_coord != None and self.main.canvas.line_type != None:
                    if math.isnan(operation.get_parameter('position')):
                        operation.set_parameter('type', self.main.canvas.line_type)
                        operation.set_parameter('position', self.main.canvas.line_coord)

            kwargs = operation.get_parameters()[1]
            #copy = operation.func(copy, **kwargs)

            operation.func(copy, **kwargs)

        return copy

    def show_window(self):
        self.show()
        self.raise_()

    def closeEvent(self, event):
        self.hide()
        event.ignore()