import math
import numpy as np
from PyQt4 import QtGui, QtCore
import os
import pandas as pd

class Settings(QtGui.QDialog):
    def __init__(self, parent=None):
        super(Settings, self).__init__(parent)

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Settings")

        hbox = QtGui.QHBoxLayout()

        self.tree = QtGui.QTreeWidget(self)
        self.tree.setHeaderLabels(['Name', 'Value'])
        self.tree.setColumnWidth(0, 200)
        self.tree.itemChanged.connect(self.on_item_changed)

        self.b_copy = QtGui.QPushButton('Copy')
        self.b_copy.clicked.connect(self.on_copy)

        hbox.addWidget(self.tree)
        hbox.addWidget(self.b_copy)

        layout = QtGui.QVBoxLayout()
        layout.addLayout(hbox)
        self.setLayout(layout)

        self.setGeometry(900, 300, 400, 600)

    def load_file(self, filename):
        path, ext = os.path.splitext(filename)
        settings_file = path + '.set'

        if os.path.exists(settings_file):
            with open(settings_file) as f:
                self.lines = f.readlines()
            
            self.fill_tree(self.lines)
        else:
            pass

    def fill_tree(self, lines):
        widgets = []

        for line in lines:
            line = line.rstrip('\n\t\r')

            if line == '':
                continue

            if not line.startswith('\t'):
                name, value = line.split(': ', 1)
                
                parent = QtGui.QTreeWidgetItem(None, [name, value])
                parent.setCheckState(0, QtCore.Qt.Unchecked)
                widgets.append(parent)
            else:
                name, value = line.split(': ', 1)

                child = QtGui.QTreeWidgetItem(parent, [name.strip(), value])
                child.setCheckState(0, QtCore.Qt.Unchecked)

        self.tree.insertTopLevelItems(0, widgets)

    def on_item_changed(self, widget):
        state = widget.checkState(0)

        for i in range(widget.childCount()):
            child = widget.child(i)
            child.setCheckState(0, state)

    def on_copy(self):
        text = ''

        root = self.tree.invisibleRootItem()
        count = root.childCount()

        for i in range(count):
            parent = root.child(i)

            header = False

            if parent.checkState(0) == QtCore.Qt.Checked:
                text += str(parent.text(0)) + ': ' + str(parent.text(1)) + '\n'
                header = True

            for j in range(parent.childCount()):
                child = parent.child(j)

                if child.checkState(0) == QtCore.Qt.Checked:
                    if not header:
                        text += str(parent.text(0)) + ': ' + str(parent.text(1)) + '\n'
                        header = True

                    text += '  ' + str(child.text(0)) + ': ' + str(child.text(1)) + '\n'

        QtGui.QApplication.clipboard().setText(text)

    def show_window(self):
        self.show()
        self.raise_()

    def closeEvent(self, event):
        self.hide()
        event.ignore()