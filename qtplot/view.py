import os

from PyQt4 import QtGui, uic

from .canvas import Canvas


class MainView(QtGui.QMainWindow):
    def __init__(self):
        super(MainView, self).__init__()

        directory = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(directory, 'ui/main.ui')
        uic.loadUi(path, self)

        self.canvas = Canvas()
        self.canvas_layout.addWidget(self.canvas.native)


class LinecutView:
    def __init__(self):
        pass
