import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import textwrap
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

    def __init__(self, x, y, type, position, **kwargs):
        plt.Line2D.__init__(self, x, y, **kwargs)

        self.type = type
        self.position = position


class Linecut(QtGui.QDialog):
    def __init__(self, main=None):
        super(Linecut, self).__init__(None)

        self.main = main

        self.fig, self.ax = plt.subplots()
        self.x, self.y = None, None
        self.linetraces = []
        self.marker = None
        self.colors = cycle('bgrcmykw')

        self.ax.xaxis.set_major_formatter(FixedOrderFormatter())
        self.ax.yaxis.set_major_formatter(FixedOrderFormatter())

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Linecut")

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.mpl_connect('pick_event', self.on_pick)
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
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+C"),
                        self, self.on_copy_figure)
        grid.addWidget(self.b_copy, 1, 4)

        self.cb_include_z = QtGui.QCheckBox('Include Z')
        self.cb_include_z.setCheckState(QtCore.Qt.Checked)
        grid.addWidget(self.cb_include_z, 1, 5)

        # Linecuts
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

        # Lines
        grid.addWidget(QtGui.QLabel('Line style'), 3, 1)
        self.cb_linestyle = QtGui.QComboBox(self)
        self.cb_linestyle.addItems(['None', 'solid', 'dashed', 'dotted'])
        grid.addWidget(self.cb_linestyle, 3, 2)

        grid.addWidget(QtGui.QLabel('Linewidth'), 3, 3)
        self.le_linewidth = QtGui.QLineEdit('0.5', self)
        grid.addWidget(self.le_linewidth, 3, 4)

        # Markers
        grid.addWidget(QtGui.QLabel('Marker style'), 4, 1)
        self.cb_markerstyle = QtGui.QComboBox(self)
        self.cb_markerstyle.addItems(['None', '.', 'o', 'x'])
        grid.addWidget(self.cb_markerstyle, 4, 2)

        grid.addWidget(QtGui.QLabel('Size'), 4, 3)
        self.le_markersize = QtGui.QLineEdit('0.5', self)
        grid.addWidget(self.le_markersize, 4, 4)

        self.row_tree = QtGui.QTreeWidget(self)
        self.row_tree.setHeaderLabels(['Parameter', 'Value'])
        self.row_tree.setColumnWidth(0, 100)
        #grid.addWidget(self.row_tree, 4, 5)

        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.canvas)
        hbox.addWidget(self.row_tree)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addLayout(hbox)
        layout.addLayout(grid)
        self.setLayout(layout)

        self.resize(700, 500)
        self.move(630, 100)

    def populate_ui(self):
        profile = self.main.profile_settings

        idx = self.cb_linestyle.findText(profile['line_style'])
        self.cb_linestyle.setCurrentIndex(idx)
        self.le_linewidth.setText(profile['line_width'])

        idx = self.cb_markerstyle.findText(profile['marker_style'])
        self.cb_markerstyle.setCurrentIndex(idx)
        self.le_markersize.setText(profile['marker_size'])

    def get_line_kwargs(self):
        return {
            'linestyle': str(self.cb_linestyle.currentText()),
            'linewidth': float(self.le_linewidth.text()),
            'marker': str(self.cb_markerstyle.currentText()),
            'markersize': float(self.le_markersize.text()),
        }

    def on_reset(self):
        if self.x is not None and self.y is not None:
            minx, maxx = np.min(self.x), np.max(self.x)
            miny, maxy = np.min(self.y), np.max(self.y)

            xdiff = (maxx - minx) * .1
            ydiff = (maxy - miny) * .1

            self.ax.axis([minx - xdiff, maxx + xdiff,
                          miny - ydiff, maxy + ydiff])

            self.canvas.draw()

    def on_pick(self, event):
        line = self.linetraces[0]

        ind = event.ind[int(len(event.ind) / 2)]
        x = line.get_xdata()[ind]
        y = line.get_ydata()[ind]

        # Find the other data in this datapoint's row
        data = self.main.dat_file.find_row({self.xlabel: x, self.ylabel: y})

        # Fill the treeview with data
        self.row_tree.clear()
        widgets = []
        for name, value in data.items():
            val = '{:.3e}'.format(value)
            val = eng_format(value, 1)
            widgets.append(QtGui.QTreeWidgetItem(None, [name, val]))

        self.row_tree.insertTopLevelItems(0, widgets)

        # Remove the previous datapoint marker
        if self.marker is not None:
            self.marker.remove()

        # Plot a new datapoint marker
        self.marker = self.ax.plot(x, y, '.', markersize=15, color='black')[0]

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

        title = '\n'.join(textwrap.wrap(title, 40, replace_whitespace=False))
        self.ax.set_title(title)

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

        # Remove all the existing lines and only plot one if we uncheck
        # the incremental box. Else, add a new line to the collection
        if self.cb_incremental.checkState() == QtCore.Qt.Unchecked:
            for line in self.linetraces:
                line.remove()

            self.linetraces = []

            line = Linetrace(x, y, type, position,
                             color='red',
                             picker=5,
                             **self.get_line_kwargs())

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
            line.set_color(next(self.colors))

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
