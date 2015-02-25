import numpy as np
import pandas as pd
from PyQt4 import QtGui, QtCore

"""
Operation

Parameters:
- Textboxes
- Listview
- Checkboxes
"""

class Operation(QtGui.QWidget):
    def __init__(self, func, widgets=[]):
        super(Operation, self).__init__(None)

        layout = QtGui.QGridLayout(self)
        self.func = func
        self.items = {}

        height = 1

        for widget in widgets:
            typ, name, data = widget

            if typ == 'checkbox':
                checkbox = QtGui.QCheckBox(name)
                checkbox.setChecked(data)
                layout.addWidget(QtGui.QCheckBox(name), height, 2)

                self.items[name] = checkbox
            elif typ == 'textbox':
                lineedit = QtGui.QLineEdit(data)
                layout.addWidget(QtGui.QLabel(name), height, 1)
                layout.addWidget(lineedit, height, 2)

                self.items[name] = lineedit
            elif typ == 'combobox':
                combobox = QtGui.QComboBox()
                combobox.addItems(data)
                layout.addWidget(combobox, height, 2)

                self.items[name] = combobox

            height += 1

    def get_property(self, name, cast):
        if name in self.items:
            widget = self.items[name]

            if type(widget) is QtGui.QCheckBox:
                return widget.isChecked()
            elif type(widget) is QtGui.QLineEdit:
                return cast(str(widget.text()))
            elif type(widget) is QtGui.QComboBox:
                return str(widget.currentText())

class Operations(QtGui.QDialog):
    def __init__(self, parent=None):
        super(Operations, self).__init__(parent)

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Operations")

        self.items = {
            'abs': (self.abs, []),
            'autoflip': None,
            'crop': (self.crop, [('textbox', 'Left', '0'), ('textbox', 'Right', '0'), ('textbox', 'Bottom', '0'), ('textbox', 'Top', '0')]),
            'dderiv': None,
            'equalize': None,
            'flip': None,
            'gradmag': None,
            'highpass': None,
            'hist2d': None,
            'log': None,
            'lowpass': None,
            'neg': (self.neg, []),
            'offset': None,
            'offset axes': (self.offset_axes, []),
            'power': None,
            'rotate ccw': None,
            'rotate cw': (self.rotate_cw, []),
            'scale axes': (self.scale_axes, []),
            'scale data': (self.scale_data, [('textbox', 'Factor', '1')]),
            'sub linecut': (self.sub_linecut, []),
            'xderiv': (self.xderiv, []),
            'yderiv': (self.yderiv, []),
        }

        self.options = QtGui.QListWidget(self)
        self.options.addItems(sorted(self.items.keys()))

        self.b_add = QtGui.QPushButton('Add')
        self.b_add.clicked.connect(self.add)

        self.b_up = QtGui.QPushButton('Up')
        self.b_up.clicked.connect(self.up)

        self.b_down = QtGui.QPushButton('Down')
        self.b_down.clicked.connect(self.down)

        self.b_remove = QtGui.QPushButton('Remove')
        self.b_remove.clicked.connect(self.remove)

        self.b_clear = QtGui.QPushButton('Clear')
        self.b_clear.clicked.connect(self.clear)

        self.b_update = QtGui.QPushButton('Update')
        self.b_update.clicked.connect(self.update)

        self.queue = QtGui.QListWidget(self)
        self.queue.currentItemChanged.connect(self.selected_changed)

        hbox = QtGui.QHBoxLayout()

        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.b_add)
        vbox.addWidget(self.b_up)
        vbox.addWidget(self.b_down)
        vbox.addWidget(self.b_remove)
        vbox.addWidget(self.b_clear)
        vbox.addWidget(self.b_update)

        vbox2 = QtGui.QVBoxLayout()
        vbox2.addWidget(self.queue)
        self.stack = QtGui.QStackedWidget()
        vbox2.addWidget(self.stack)

        hbox.addWidget(self.options)
        hbox.addLayout(vbox)
        hbox.addLayout(vbox2)
        
        self.setLayout(hbox)

        self.setGeometry(800, 700, 300, 200)

    def update_plot(func):
        def wrapper(self):
            func(self)
            self.main.on_axis_changed(None)

        return wrapper
    
    @update_plot
    def add(self):
        if self.options.currentItem():
            name = str(self.options.currentItem().text())

            item = QtGui.QListWidgetItem(name)
            operation = Operation(*self.items[name])
            item.setData(QtCore.Qt.UserRole, QtCore.QVariant(operation))
            self.stack.addWidget(operation)
            self.queue.addItem(item)

    @update_plot
    def up(self):
        selected_row = self.queue.currentRow()
        current = self.queue.takeItem(selected_row)
        self.queue.insertItem(selected_row - 1, current)
        self.queue.setCurrentRow(selected_row - 1)

    @update_plot
    def down(self):
        selected_row = self.queue.currentRow()
        current = self.queue.takeItem(selected_row)
        self.queue.insertItem(selected_row + 1, current)
        self.queue.setCurrentRow(selected_row + 1)

    @update_plot
    def remove(self):
        self.queue.takeItem(self.queue.currentRow())

    @update_plot
    def clear(self):
        self.queue.clear()

    @update_plot
    def update(self):
        pass

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Return:
            self.main.on_axis_changed(None)

    def selected_changed(self, current, previous):
        if current:
            widget = current.data(QtCore.Qt.UserRole).toPyObject()
            self.stack.addWidget(widget)
            self.stack.setCurrentWidget(widget)

    def apply_operations(self, data):
        ops = []

        for i in xrange(self.queue.count()):
            item = self.queue.item(i)
            operation = item.data(QtCore.Qt.UserRole).toPyObject()
            name = str(self.queue.item(i).text())
            data = operation.func(data, operation)

        return data

    def abs(self, data, item):
        return data.applymap(np.absolute)

    def crop(self, data, op):
        x1, x2 = op.get_property('Left', int), op.get_property('Right', int)
        y1, y2 = op.get_property('Bottom', int), op.get_property('Top', int)
        return data.iloc[x1:x2, y1:y2]

    def neg(self, data, item):
        return data.applymap(np.negative)

    def offset_axes(self, data, item):
        x_off, y_off = [float(x) for x in str(item.data(QtCore.Qt.UserRole).toPyObject()).split()]
        return pd.DataFrame(data.values, index=data.index + y_off, columns=data.columns + x_off)

    def rotate_cw(self, data, item):
        return pd.DataFrame(data.values, index=data.index, columns=data.columns)

    def scale_axes(self, data, item):
        x_sc, y_sc = [float(x) for x in str(item.data(QtCore.Qt.UserRole).toPyObject()).split()]
        return pd.DataFrame(data.values, index=data.index * y_sc, columns=data.columns * x_sc)

    def scale_data(self, data, op):
        factor = op.get_property('Factor', float)
        return pd.DataFrame(data.values * factor, index=data.index, columns=data.columns)

    def sub_linecut(self, data, item):
        if self.main.linecut_type == None:
            return data

        if self.main.linecut_type == 'horizontal':
            lc_data = np.array(data.loc[self.main.linecut_coord])
        elif self.main.linecut_type == 'vertical':
            lc_data = np.array(data[self.main.linecut_coord])
            lc_data = lc_data[:,np.newaxis]

        return pd.DataFrame(data.values - lc_data, index=data.index, columns=data.columns)

    def xderiv(self, data, item):
        return pd.DataFrame(np.gradient(data.values)[1], index=data.index, columns=data.columns)

    def yderiv(self, data, item):
        return pd.DataFrame(np.gradient(data.values)[0], index=data.index, columns=data.columns)