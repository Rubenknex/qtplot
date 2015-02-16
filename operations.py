import numpy as np
import pandas as pd
from PyQt4 import QtGui, QtCore

class Operations(QtGui.QDialog):
    def __init__(self, parent=None):
        super(Operations, self).__init__(parent)

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Operations")

        items = ['abs', 'autoflip', 'crop', 'sub linecut', 'sub plane', 'xderiv', 'yderiv']

        self.options = QtGui.QListWidget(self)
        self.options.addItems(items)

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

    def sub_linecut(self, data):
        pass

    def perform_operation(self, data):
        return data