import numpy as np
import pandas as pd
from PyQt4 import QtGui, QtCore

class Operations(QtGui.QDialog):
    def __init__(self, parent=None):
        super(Operations, self).__init__(parent)

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Operations")

        self.items = {
            'abs': self.abs,
            'crop': self.crop,
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

        hbox = QtGui.QHBoxLayout()

        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.b_add)
        vbox.addWidget(self.b_up)
        vbox.addWidget(self.b_down)
        vbox.addWidget(self.b_remove)
        vbox.addWidget(self.b_clear)

        hbox.addWidget(self.options)
        hbox.addLayout(vbox)
        hbox.addWidget(self.queue)
        
        self.setLayout(hbox)

        self.setGeometry(800, 700, 300, 200)
        
    def add(self):
        if self.options.currentItem():
            self.queue.addItem(self.options.currentItem().text())

    def up(self):
        selected_row = self.queue.currentRow()
        current = self.queue.takeItem(selected_row)
        self.queue.insertItem(selected_row - 1, current)
        self.queue.setCurrentRow(selected_row - 1)

    def down(self):
        selected_row = self.queue.currentRow()
        current = self.queue.takeItem(selected_row)
        self.queue.insertItem(selected_row + 1, current)
        self.queue.setCurrentRow(selected_row + 1)

    def remove(self):
        self.queue.takeItem(self.queue.currentRow())

    def clear(self):
        self.queue.clear()

    def abs(self, data):
        return data.applymap(np.absolute)

    def crop(self, data):
        return data

    def sub_linecut(self, data):
        if self.main.linecut_type == None:
            return data

        values = data.values
        lc_data = None

        if self.main.linecut_type == 'horizontal':
            lc_data = np.array(data.loc[self.main.linecut_coord])
        elif self.main.linecut_type == 'vertical':
            lc_data = np.array(data[self.main.linecut_coord])
            lc_data = lc_data[:,np.newaxis]

        values -= lc_data

        return pd.DataFrame(values, index=data.index, columns=data.columns)

    def xderiv(self, data):
        return data

    def yderiv(self, data):
        return data

    def perform_operation(self, data):
        ops = []
        for i in xrange(self.queue.count()):
            ops.append(str(self.queue.item(i).text()))

        for op in ops:
            data = self.items[op](data)

        return data