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

class Operations(QtGui.QDialog):
    def __init__(self, parent=None):
        super(Operations, self).__init__(parent)

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Operations")

        self.items = {
            'abs': self.abs,
            'autoflip': None,
            'crop': self.crop,
            'dderiv': None,
            'equalize': None,
            'flip': None,
            'gradmag': None,
            'highpass': None,
            'hist2d': None,
            'log': None,
            'lowpass': None,
            'neg': self.neg,
            'offset': None,
            'offset axes': self.offset_axes,
            'power': None,
            'rotate ccw': None,
            'rotate cw': None,
            'scale axes': self.scale_axes,
            'scale data': self.scale_data,
            'sub linecut': self.sub_linecut,
            'xderiv': self.xderiv,
            'yderiv': self.yderiv,
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

        self.queue = QtGui.QListWidget(self)
        self.queue.currentItemChanged.connect(self.selected_changed)

        self.le_op = QtGui.QLineEdit(self)
        self.le_op.textEdited.connect(self.text_changed)
        self.le_op.editingFinished.connect(self.return_pressed)

        hbox = QtGui.QHBoxLayout()

        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.b_add)
        vbox.addWidget(self.b_up)
        vbox.addWidget(self.b_down)
        vbox.addWidget(self.b_remove)
        vbox.addWidget(self.b_clear)

        vbox2 = QtGui.QVBoxLayout()
        vbox2.addWidget(self.queue)
        vbox2.addWidget(self.le_op)

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
            name = self.options.currentItem().text()

            data = ''
            if name == 'crop':
                data = 'x1 x2 y1 y2'

            item = QtGui.QListWidgetItem(name)
            item.setData(QtCore.Qt.UserRole, QtCore.QVariant(data))
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

    def selected_changed(self, current, previous):
        if current:
            data = current.data(QtCore.Qt.UserRole).toPyObject()
            self.le_op.setText(data)

    def text_changed(self, text):
        if self.queue.currentItem():
            self.queue.currentItem().setData(QtCore.Qt.UserRole, QtCore.QVariant(str(text)))

    @update_plot
    def return_pressed(self):
        pass

    def abs(self, data, item):
        return data.applymap(np.absolute)

    def crop(self, data, item):
        coords = [int(x) for x in str(item.data(QtCore.Qt.UserRole).toPyObject()).split()]
        return data.iloc[coords[0]:coords[1], coords[2]:coords[3]]

    def neg(self, data, item):
        return data.applymap(np.negative)

    def offset_axes(self, data, item):
        x_off, y_off = [float(x) for x in str(item.data(QtCore.Qt.UserRole).toPyObject()).split()]
        return pd.DataFrame(data.values, index=data.index + y_off, columns=data.columns + x_off)

    def scale_axes(self, data, item):
        x_sc, y_sc = [float(x) for x in str(item.data(QtCore.Qt.UserRole).toPyObject()).split()]
        return pd.DataFrame(data.values, index=data.index * y_sc, columns=data.columns * x_sc)

    def scale_data(self, data, item):
        factor = float(str(item.data(QtCore.Qt.UserRole).toPyObject()))
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

    def apply_operations(self, data):
        ops = []

        for i in xrange(self.queue.count()):
            item = self.queue.item(i)
            name = str(self.queue.item(i).text())
            data = self.items[name](data, item)

        return data