import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from itertools import cycle

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg, NavigationToolbar2QT

from PyQt4 import QtGui, QtCore

from .util import FixedOrderFormatter, eng_format


class Linetrace(plt.Line2D):
    """
    Represents a linetrace from the data. The purpose of this class is
    to be able to store incremental linetraces in an array.

    x/y:        Arrays containing x and y data
    type:       Type of linetrace, 'horizontal' or 'vertical'
    position:   The x/y coordinate at which the linetrace was taken
    """
    def __init__(self, x, y, type, position):
        plt.Line2D.__init__(self, x, y, color='red', linewidth=0.5)

        self.type = type
        self.position = position


class Linecut(QtGui.QDialog):
    def __init__(self, parent=None):
        super(Linecut, self).__init__(None)

        self.fig, self.ax = plt.subplots()
        self.x, self.y = None, None
        self.linetraces = []
        self.points = []
        self.markers = []
        self.colors = cycle('bgrcmykw')

        self.ax.xaxis.set_major_formatter(FixedOrderFormatter())
        self.ax.yaxis.set_major_formatter(FixedOrderFormatter())

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Linecut")

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        grid = QtGui.QGridLayout()

        self.cb_reset_cmap = QtGui.QCheckBox('Reset on plot')
        self.cb_reset_cmap.setCheckState(QtCore.Qt.Checked)
        grid.addWidget(self.cb_reset_cmap, 1, 1)

        self.b_save = QtGui.QPushButton('Data to clipboard', self)
        self.b_save.clicked.connect(self.on_clipboard)
        grid.addWidget(self.b_save, 1, 2)

        self.b_save_dat = QtGui.QPushButton('Save data...', self)
        self.b_save_dat.clicked.connect(self.on_save)
        grid.addWidget(self.b_save_dat, 1, 3)

        self.b_copy = QtGui.QPushButton('Figure to clipboard', self)
        self.b_copy.clicked.connect(self.on_copy_figure)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+C"), self, self.on_copy_figure)
        grid.addWidget(self.b_copy, 1, 4)

        self.cb_include_z = QtGui.QCheckBox('Include Z')
        self.cb_include_z.setCheckState(QtCore.Qt.Checked)
        grid.addWidget(self.cb_include_z, 1, 5)

        grid.addWidget(QtGui.QLabel('Linecuts'), 2, 1)

        self.cb_incremental = QtGui.QCheckBox('Incremental')
        self.cb_incremental.setCheckState(QtCore.Qt.Unchecked)
        grid.addWidget(self.cb_incremental, 2, 2)

        grid.addWidget(QtGui.QLabel('Offset:'), 2, 3)

        self.le_offset = QtGui.QLineEdit('0', self)
        grid.addWidget(self.le_offset, 2, 4)

        self.b_clear_lines = QtGui.QPushButton('Clear', self)
        self.b_clear_lines.clicked.connect(self.on_clear_lines)
        grid.addWidget(self.b_clear_lines, 2, 5)

        grid.addWidget(QtGui.QLabel('Points'), 3, 1)
        self.cb_point = QtGui.QComboBox(self)
        self.cb_point.addItems(['X,Y', 'X', 'Y'])
        grid.addWidget(self.cb_point, 3, 2)

        grid.addWidget(QtGui.QLabel('Significance'))
        self.sb_significance = QtGui.QSpinBox(self)
        grid.addWidget(self.sb_significance, 3, 4)

        self.b_clear_points = QtGui.QPushButton('Clear', self)
        self.b_clear_points.clicked.connect(self.on_clear_points)
        grid.addWidget(self.b_clear_points, 3, 5)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addLayout(grid)
        self.setLayout(layout)

        self.resize(500, 500)
        self.move(630, 100)

    def on_reset(self):
        if self.x is not None and self.y is not None:
            minx, maxx = np.min(self.x), np.max(self.x)
            miny, maxy = np.min(self.y), np.max(self.y)

            xdiff = (maxx - minx) * .1
            ydiff = (maxy - miny) * .1

            self.ax.axis([minx - xdiff, maxx + xdiff,
                         miny - ydiff, maxy + ydiff])

            self.canvas.draw()

    def on_click(self, event):
        if event.inaxes and event.button == 2:
            significance = self.sb_significance.value()
            point_type = str(self.cb_point.currentText())

            if point_type == 'X':
                coords = eng_format(event.xdata, significance)
            elif point_type == 'Y':
                coords = eng_format(event.ydata, significance)
            elif point_type == 'X,Y':
                coords = '%s, %s' % (eng_format(event.xdata, significance),
                                     eng_format(event.ydata, significance))
            else:
                coords = ''

            marker = self.ax.plot(event.xdata, event.ydata, '+', c='k')[0]
            self.markers.append(marker)
            text = self.ax.annotate(coords,
                                    xy=(event.xdata, event.ydata),
                                    xycoords='data',
                                    xytext=(3, 3),
                                    textcoords='offset points')

            self.points.append(text)

            self.fig.canvas.draw()

    def on_clipboard(self):
        if self.x is None or self.y is None:
            return

        data = pd.DataFrame(np.column_stack((self.x, self.y)),
                            columns=[self.xlabel, self.ylabel])

        data.to_clipboard(index=False)

    def on_save(self):
        if self.x is None or self.y is None:
            return

        path = os.path.dirname(os.path.realpath(__file__))
        filename = QtGui.QFileDialog.getSaveFileName(self,
                                                     'Save file',
                                                     path,
                                                     '.dat')

        if filename != '':
            data = pd.DataFrame(np.column_stack((self.x, self.y)),
                                columns=[self.xlabel, self.ylabel])

            data.to_csv(filename, sep='\t', index=False)

    def on_copy_figure(self):
        path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(path, 'test.png')
        self.fig.savefig(path, bbox_inches='tight')

        img = QtGui.QImage(path)
        QtGui.QApplication.clipboard().setImage(img)

    def on_clear_lines(self):
        for line in self.linetraces:
            line.remove()

        self.linetraces = []

        self.fig.canvas.draw()

    def on_clear_points(self):
        for point in self.points:
            point.remove()

        self.points = []

        for marker in self.markers:
            marker.remove()

        self.markers = []

        self.fig.canvas.draw()

    def plot_linetrace(self, x, y, z, type, position, title,
                       xlabel, ylabel, otherlabel):
        # Don't draw lines consisting of one point
        if np.count_nonzero(~np.isnan(y)) < 2:
            return

        self.xlabel, self.ylabel, self.otherlabel = xlabel, ylabel, otherlabel
        self.title = title
        self.x, self.y, self.z = x, y, z

        if self.cb_include_z.checkState() == QtCore.Qt.Checked:
            title = '{0}\n{1} = {2}'.format(title, otherlabel, eng_format(z, 1))

        self.ax.set_title(title)

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

        # Remove all the existing lines and only plot one if we uncheck
        # the incremental box. Else, add a new line to the collection
        if self.cb_incremental.checkState() == QtCore.Qt.Unchecked:
            for line in self.linetraces:
                line.remove()

            self.linetraces = []

            line = Linetrace(x, y, type, position)
            self.linetraces.append(line)
            self.ax.add_line(line)

            self.total_offset = 0
        else:
            if len(self.ax.lines) > 0:
                if self.linetraces[-1].position == position:
                    return

            index = len(self.linetraces) - 1

            offset = float(self.le_offset.text())
            line = Linetrace(x, y + index * offset, type, position)
            line.set_color(self.colors.next())

            self.linetraces.append(line)
            self.ax.add_line(line)

        if self.cb_reset_cmap.checkState() == QtCore.Qt.Checked:
            x, y = np.ma.masked_invalid(x), np.ma.masked_invalid(y)
            minx, maxx = np.min(x), np.max(x)
            miny, maxy = np.min(y), np.max(y)

            xdiff = (maxx - minx) * .05
            ydiff = (maxy - miny) * .05

            self.ax.axis([minx - xdiff, maxx + xdiff,
                          miny - ydiff, maxy + ydiff])

        self.ax.set_aspect('auto')
        self.fig.tight_layout()

        self.fig.canvas.draw()

    def resizeEvent(self, event):
        self.fig.tight_layout()
        self.canvas.draw()

    def show_window(self):
        self.show()
        self.raise_()

    def closeEvent(self, event):
        self.hide()
        event.ignore()
